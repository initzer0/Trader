from __future__ import annotations

import argparse
import json
import signal
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import optuna
from optuna.trial import TrialState

# Allow running without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trader.backtest.switching import backtest_long_or_cash
from trader.ml.data import (
    DataConfig,
    default_btc_1m_csv,
    load_ohlc_close_series,
    make_supervised_samples,
    time_split_samples,
)
from trader.ml.models import LSTMReturnRegressor
from trader.ml.torch_dataset import ReturnSequenceTorchDataset
from trader.ml.train import TrainConfig, predict_regressor, train_regressor


_STOP_REQUESTED = False


def _request_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (Path,)):
        return str(v)
    return v


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _cleanup_failed_trials(study: optuna.Study) -> int:
    """Delete failed trials from a persisted Optuna study.

    This is useful when resuming from the same sqlite db: failed trials can pile up
    and make it harder to sift through results.

    Best-effort: if the storage doesn't support deletion or Optuna internals change,
    we skip deletion rather than failing the whole run.
    """

    trials = study.get_trials(deepcopy=False)
    failed = [t for t in trials if t.state == TrialState.FAIL]
    if not failed:
        return 0

    storage = getattr(study, "_storage", None)
    delete_trial = getattr(storage, "delete_trial", None)
    if delete_trial is None:
        return 0

    deleted = 0
    for t in failed:
        trial_id = getattr(t, "_trial_id", None)
        if trial_id is None:
            continue
        try:
            delete_trial(trial_id)
            deleted += 1
        except Exception:
            # Keep going even if a particular trial can't be deleted.
            continue

    return deleted


def _plot_test_trades(
    *,
    out_path: Path,
    times: np.ndarray,
    close: np.ndarray,
    decision_times: np.ndarray,
    decision_close: np.ndarray,
    positions_btc: np.ndarray,
    title: str,
    max_points: int = 50_000,
) -> None:
    """Plot BTC close over test period with a position overlay.

    - Blue line: BTC close price
    - Orange step: model position (0=USDT, 1=BTC) (scaled to right axis)
    """

    if times.size == 0 or decision_times.size == 0:
        return

    # Downsample for plotting performance.
    def downsample(
        t: np.ndarray, y: np.ndarray, nmax: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if t.size <= nmax:
            return t, y
        step = max(1, int(np.ceil(t.size / nmax)))
        return t[::step], y[::step]

    pt_t, pt_close = downsample(times, close, max_points)
    dt_t, dt_pos = downsample(decision_times, positions_btc.astype(float), max_points)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(pt_t, pt_close, linewidth=1.0, label="BTC close")
    ax1.set_xlabel("Datetime (UTC)")
    ax1.set_ylabel("BTC close")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.step(
        dt_t,
        dt_pos,
        where="post",
        alpha=0.35,
        label="Position (BTC=1)",
        color="tab:orange",
    )
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Position")

    # Mark trade events (switches) but keep it bounded.
    switches = np.flatnonzero(np.diff(positions_btc.astype(int)) != 0) + 1
    if switches.size > 0:
        max_markers = 2000
        if switches.size > max_markers:
            step = int(np.ceil(switches.size / max_markers))
            switches = switches[::step]
        ax1.vlines(
            decision_times[switches],
            ymin=np.min(pt_close),
            ymax=np.max(pt_close),
            alpha=0.06,
            linewidth=0.5,
        )

    # Legend (combine both axes)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_hist_with_stats(
    *,
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    bins: int = 120,
) -> None:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    ax.hist(data, bins=bins, color="tab:blue", alpha=0.75)
    mean = float(np.mean(data))
    var = float(np.var(data))
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.text(
        0.98,
        0.95,
        f"mean={mean:.6g}\nvar={var:.6g}\nN={data.size}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )


def _plot_input_output_distributions(
    *,
    out_dir: Path,
    split: "object",  # SplitSamples
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        x_train = split.train.x.reshape(-1)
        x_val = split.val.x.reshape(-1)
        x_test = split.test.x.reshape(-1)

        y_train = split.train.y.reshape(-1)
        y_val = split.val.y.reshape(-1)
        y_test = split.test.y.reshape(-1)

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        _plot_hist_with_stats(ax=axes[0], data=x_train, title="Input (train)")
        _plot_hist_with_stats(ax=axes[1], data=x_val, title="Input (val)")
        _plot_hist_with_stats(ax=axes[2], data=x_test, title="Input (test)")
        axes[-1].set_xlabel("input value")
        fig.tight_layout()
        fig.savefig(out_dir / "input_distribution.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        _plot_hist_with_stats(ax=axes[0], data=y_train, title="Output (train)")
        _plot_hist_with_stats(ax=axes[1], data=y_val, title="Output (val)")
        _plot_hist_with_stats(ax=axes[2], data=y_test, title="Output (test)")
        axes[-1].set_xlabel("output value")
        fig.tight_layout()
        fig.savefig(out_dir / "output_distribution.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


def objective(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    fee_rate: float,
    device: str,
    run_dir: Path,
    artifact_mode: str,
) -> float:
    downsample_every = trial.suggest_categorical(
        "downsample_every", [1, 5, 15, 60, 240, 1440]
    )
    lookback = trial.suggest_int("lookback", 10, 120)
    prediction_horizon = trial.suggest_categorical(
        "prediction_horizon", [1, 3, 5, 15, 60]
    )

    # How often we make a prediction / decision / potential trade.
    # This is distinct from prediction_horizon: e.g. predict 12h ahead but decide every 6h.
    decision_every = trial.suggest_categorical(
        "decision_every",
        [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 120, 240, 360, 720, 1440],
    )

    # For this backtest, trade cadence equals decision cadence (we can rebalance each decision).
    trade_horizon = decision_every

    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_categorical(
        "weight_decay", [0.0, 1e-8, 1e-6, 1e-4, 1e-3]
    )
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    epochs = trial.suggest_int("epochs", 3, 10)

    try:
        series = load_ohlc_close_series(csv_path, downsample_every=downsample_every)
        samples = make_supervised_samples(
            series,
            lookback=lookback,
            prediction_horizon=prediction_horizon,
            sample_every=decision_every,
        )

        split = time_split_samples(
            series,
            samples,
            split_dt=datetime(2025, 1, 1, tzinfo=timezone.utc),
            val_fraction=0.1,
            prediction_horizon=prediction_horizon,
        )
    except (ValueError, RuntimeError) as exc:
        # Some hyperparameter combos (e.g. massive downsampling + long horizon) can
        # make the pre-2025 window too small. Treat these as invalid trials.
        trial.set_user_attr("pruned_reason", str(exc))
        raise optuna.TrialPruned(str(exc))

    train_ds = ReturnSequenceTorchDataset(split.train.x, split.train.y)
    val_ds = ReturnSequenceTorchDataset(split.val.x, split.val.y)
    test_x = split.test.x

    model = LSTMReturnRegressor(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    train_metrics = train_regressor(
        model, train_ds=train_ds, val_ds=val_ds, config=train_cfg
    )

    # Evaluate on validation with a trading rule
    val_pred = predict_regressor(
        model,
        x=split.val.x,
        device=device,
        batch_size=train_cfg.batch_size,
    )
    val_bt = backtest_long_or_cash(
        close=series.close,
        base_index=split.val.base_index,
        pred_log_return=val_pred,
        trade_horizon=trade_horizon,
        fee_rate=fee_rate,
    )

    # Also compute test metrics for reference
    test_pred = predict_regressor(
        model,
        x=test_x,
        device=device,
        batch_size=train_cfg.batch_size,
    )
    test_bt = backtest_long_or_cash(
        close=series.close,
        base_index=split.test.base_index,
        pred_log_return=test_pred,
        trade_horizon=trade_horizon,
        fee_rate=fee_rate,
    )

    trial.set_user_attr("val_mse", train_metrics["val_mse"])
    trial.set_user_attr("best_epoch", train_metrics.get("best_epoch", -1))
    trial.set_user_attr(
        "train_mse_last", train_metrics.get("train_mse_last", float("nan"))
    )
    trial.set_user_attr("val_mse_last", train_metrics.get("val_mse_last", float("nan")))
    trial.set_user_attr("val_return_gross", val_bt.total_return_gross)
    trial.set_user_attr("val_return_net", val_bt.total_return_net)
    trial.set_user_attr("val_final_eur_gross", val_bt.final_value_gross)
    trial.set_user_attr("val_final_eur_net", val_bt.final_value_net)
    trial.set_user_attr("val_trades", val_bt.n_trades)

    trial.set_user_attr("test_return_gross", test_bt.total_return_gross)
    trial.set_user_attr("test_return_net", test_bt.total_return_net)
    trial.set_user_attr("test_final_eur_gross", test_bt.final_value_gross)
    trial.set_user_attr("test_final_eur_net", test_bt.final_value_net)
    trial.set_user_attr("test_trades", test_bt.n_trades)

    # Per-trial artifacts
    if artifact_mode.lower() == "all":
        trial_dir = run_dir / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Data distribution plots (post-preprocessing)
        _plot_input_output_distributions(out_dir=trial_dir, split=split)

        cfg = {
            "csv": str(csv_path),
            "fee_rate": fee_rate,
            "device": device,
            "downsample_every": downsample_every,
            "lookback": lookback,
            "prediction_horizon": prediction_horizon,
            "decision_every": decision_every,
            "trade_horizon": trade_horizon,
            "model": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            "train": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            "split_dt": "2025-01-01T00:00:00+00:00",
        }
        _write_json(trial_dir / "config.json", cfg)

        # Add test window information (interpreting equity curves as € from 1€ initial).
        test_base = split.test.base_index
        test_start_time = series.times[int(test_base.min())]
        test_end_time = series.times[
            int(min(series.close.size - 1, test_base.max() + trade_horizon))
        ]

        eval_payload = {
            "trial_number": trial.number,
            "initial_eur": 1.0,
            "test_window": {
                "start": test_start_time.isoformat(),
                "end": test_end_time.isoformat(),
            },
            "val": {
                "mse": train_metrics["val_mse"],
                "return_gross": val_bt.total_return_gross,
                "return_net": val_bt.total_return_net,
                "final_eur_gross": val_bt.final_value_gross,
                "final_eur_net": val_bt.final_value_net,
                "n_trades": val_bt.n_trades,
            },
            "test": {
                "return_gross": test_bt.total_return_gross,
                "return_net": test_bt.total_return_net,
                "final_eur_gross": test_bt.final_value_gross,
                "final_eur_net": test_bt.final_value_net,
                "n_trades": test_bt.n_trades,
            },
        }
        _write_json(trial_dir / "eval.json", eval_payload)

        # Save loss curves
        try:
            loss_payload = {
                "train_mse_history": train_metrics.get("train_mse_history", []),
                "val_mse_history": train_metrics.get("val_mse_history", []),
                "best_epoch": train_metrics.get("best_epoch", -1),
            }
            _write_json(trial_dir / "loss.json", loss_payload)

            train_hist = np.asarray(loss_payload["train_mse_history"], dtype=float)
            val_hist = np.asarray(loss_payload["val_mse_history"], dtype=float)
            epochs = np.arange(1, max(train_hist.size, val_hist.size) + 1)

            fig, ax = plt.subplots(figsize=(10, 4))
            if train_hist.size:
                ax.plot(epochs[: train_hist.size], train_hist, label="train MSE")
            if val_hist.size:
                ax.plot(epochs[: val_hist.size], val_hist, label="val MSE")
            be = int(loss_payload.get("best_epoch", -1))
            if be >= 0:
                ax.axvline(
                    be + 1, color="gray", alpha=0.3, linestyle="--", label="best epoch"
                )
            ax.set_title("Training loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE")
            ax.legend()
            fig.tight_layout()
            fig.savefig(trial_dir / "loss.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

        # Save model
        try:
            import torch

            torch.save(model.state_dict(), trial_dir / "model_state.pt")
        except Exception:
            pass

        # Save raw arrays for evaluation/debug
        try:
            np.save(
                trial_dir / "test_pred_log_return.npy", test_pred.astype(np.float32)
            )
            np.save(
                trial_dir / "test_true_log_return.npy", split.test.y.astype(np.float32)
            )
            np.save(
                trial_dir / "test_equity_curve_gross.npy",
                test_bt.equity_curve_gross.astype(np.float64),
            )
            np.save(
                trial_dir / "test_equity_curve_net.npy",
                test_bt.equity_curve_net.astype(np.float64),
            )
        except Exception:
            pass

        # Save decision table (CSV) for exact trade inspection
        try:
            positions_btc = test_pred > 0.0
            decision_times = series.times[test_base]
            decision_close = series.close[test_base]
            trade_event = np.zeros_like(positions_btc, dtype=bool)
            trade_event[1:] = positions_btc[1:] != positions_btc[:-1]
            out_csv = trial_dir / "test_decisions.csv"
            header = (
                "time,close,pred_log_return,true_log_return,position_btc,trade_event\n"
            )
            with out_csv.open("w", encoding="utf-8") as f:
                f.write(header)
                for t, c, p, ytrue, pos, te in zip(
                    decision_times.astype(str).to_numpy(),
                    decision_close,
                    test_pred,
                    split.test.y,
                    positions_btc,
                    trade_event,
                ):
                    f.write(
                        f"{t},{c:.10g},{p:.10g},{float(ytrue):.10g},{int(pos)},{int(te)}\n"
                    )
        except Exception:
            pass

        # Save quick plot for test set
        try:
            dec_times = series.times[test_base].to_numpy()
            dec_close = series.close[test_base]

            start_i = int(test_base.min())
            end_i = int(min(series.close.size - 1, test_base.max() + trade_horizon))
            full_times = series.times[start_i : end_i + 1].to_numpy()
            full_close = series.close[start_i : end_i + 1]

            _plot_test_trades(
                out_path=trial_dir / "test_trades.png",
                times=full_times,
                close=full_close,
                decision_times=dec_times,
                decision_close=dec_close,
                positions_btc=(test_pred > 0.0),
                title=f"Test trades (trial {trial.number}) | net={test_bt.final_value_net:.3f}€ gross={test_bt.final_value_gross:.3f}€",
            )
        except Exception:
            pass

    # We mainly optimize net-of-fee returns.
    return float(val_bt.total_return_net)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optuna search: LSTM predicts next log return; trade BTC if pred > 0."
    )
    parser.add_argument(
        "--storage", default=str(PROJECT_ROOT / "experiment" / "optuna.db")
    )
    parser.add_argument("--study", default="lstm_btc_returns")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--fee", type=float, default=0.001, help="Trading fee rate (0.001 = 0.1%)"
    )
    parser.add_argument(
        "--device", default="cpu", help="torch device, e.g. cpu or cuda"
    )
    parser.add_argument("--csv", default=str(default_btc_1m_csv(PROJECT_ROOT)))
    parser.add_argument(
        "--cleanup-failed-trials",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete FAILED trials from the Optuna DB at startup (recommended when resuming).",
    )
    parser.add_argument(
        "--artifacts",
        default="all",
        choices=["none", "best", "all"],
        help="What artifacts to write. all=per-trial dirs, best=only best at end, none=write nothing.",
    )
    parser.add_argument(
        "--graceful-stop",
        action="store_true",
        default=True,
        help="Stop after the current trial when Ctrl+C is pressed (resumable with same --storage/--study).",
    )
    args = parser.parse_args()

    storage_url = f"sqlite:///{Path(args.storage).as_posix()}"
    study = optuna.create_study(
        study_name=args.study,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )

    if args.cleanup_failed_trials:
        deleted = _cleanup_failed_trials(study)
        if deleted:
            print(f"Deleted {deleted} failed trial(s) from storage.")

    csv_path = Path(args.csv)

    if args.graceful_stop:
        try:
            signal.signal(signal.SIGINT, _request_stop)
        except Exception:
            # Some environments may not allow installing signal handlers.
            pass

    # Run directory for this invocation
    run_root = PROJECT_ROOT / "experiment" / "runs" / args.study
    run_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run_config.json",
        {
            "study": args.study,
            "storage": f"sqlite:///{Path(args.storage).as_posix()}",
            "csv": str(csv_path),
            "fee": float(args.fee),
            "device": args.device,
            "trials": int(args.trials),
            "artifacts": args.artifacts,
            "cleanup_failed_trials": bool(args.cleanup_failed_trials),
        },
    )

    def _stop_callback(
        study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:  # noqa: ARG001
        if _STOP_REQUESTED:
            study.stop()

    try:
        study.optimize(
            lambda t: objective(
                t,
                csv_path=csv_path,
                fee_rate=float(args.fee),
                device=args.device,
                run_dir=run_dir,
                artifact_mode=args.artifacts,
            ),
            n_trials=args.trials,
            callbacks=[_stop_callback] if args.graceful_stop else None,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        # If Ctrl+C happens mid-trial, we still exit cleanly. Re-run with the same
        # --storage/--study to resume from the last completed trial.
        print("\nInterrupted. Resume by rerunning with the same --storage and --study.")
        return 130

    best = study.best_trial
    print("Best trial:")
    print(f"  value (val net return): {best.value:.6f}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    out_path = run_dir / "best.txt"

    lines = [
        f"study={args.study}",
        f"storage={storage_url}",
        f"best_value={best.value}",
        "params=" + repr(best.params),
        "user_attrs=" + repr(best.user_attrs),
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")

    # Optionally write artifacts for best trial (even if per-trial artifacts are disabled)
    if args.artifacts in {"best"}:
        # Reconstruct and save best trial artifacts in a stable folder
        best_dir = run_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            best_dir / "best_trial.json",
            {"value": best.value, "params": best.params, "user_attrs": best.user_attrs},
        )

    print()
    print("Optuna dashboard:")
    print(f"  optuna-dashboard {storage_url} --study-name {args.study}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
