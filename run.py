# run_experiments.py
import argparse, os, json, time, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import builtins, inspect
import random,torch

from fairbench.utils.logging_utils import setup_logger, Timer, Heartbeat
from fairbench.datasets import load_dataset 
from fairbench.methods import run_method
from fairbench.metrics import compute_metrics, compute_metrics_train


# Print to log
def setup_logging(save_dir, method, dataset, seed):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{method}_{dataset}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt="%(asctime)s %(levelname).1s %(name)s:%(lineno)d - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt)
    root.handlers = [sh, fh]
    logging.captureWarnings(True) 

_orig_print = builtins.print

def _print_to_log(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    msg = sep.join(str(a) for a in args)
    frame = inspect.currentframe().f_back
    mod = frame.f_globals.get("__name__", "?")
    logging.getLogger(mod).info(f"{msg}")

builtins.print = _print_to_log

# Parse
def parse_args():
    p = argparse.ArgumentParser(description="Subgroup Fairness Bench")

    # Dataset
    p.add_argument("--dataset", type=str, required=True, default="adult")
    p.add_argument("--data_dir", type=str, default="data/raw/")
    p.add_argument("--sparse_n_groups", type=int, default=5)

    # Common
    p.add_argument("--x_sensitive", type=str, default="concat", choices=["concat", "drop"],
                   help="drop: f(x), concat: f(x,s)")
    p.add_argument("--sens_keys", type=str, default=None,
                   help=("sensitive attribute list'"))
    p.add_argument("--sens_thresh", type=float, default=0.5,
                   help="sensitive attribute threshold")
    p.add_argument("--base_model", type=str, default="mlp", choices=["mlp", "linear"])

    # Parameters
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)

    # Method selection
    p.add_argument("--method", type=str, required=True, default="dr")

    # DRAF
    p.add_argument("--lambda_fair", type=float, default=0.0,
                   help="DR fairness weight")
    p.add_argument("--n_low_frac", type=float, default=None,
                    help="minimum subgroup sample number")
    p.add_argument("--af_max_order", type=int, nargs='+', default=None)

    # Results/logging
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--logfile", type=str, default="experiment.log")
    p.add_argument("--save_dir", type=str, default="results")

    # apriori
    p.add_argument("--union_mode", type=str, default="apriori_forward",
                   choices=["all", "apriori_forward"])

    return p.parse_args()

# main
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    try: torch.manual_seed(args.seed); 
    except: pass

    setup_logging(save_dir=args.save_dir, method=args.method, dataset=args.dataset, seed=args.seed)

    log = setup_logger("fair", args.logfile or None)
    hb = Heartbeat(log, 0)
    hb.start()
    for k, v in vars(args).items():
        print(f"  {k} = {v}")

    exp_name = args.exp_name or f"{args.dataset}_{args.method}_seed{args.seed}_{int(time.time())}"
    save_dir = Path(args.save_dir); save_dir.mkdir(exist_ok=True, parents=True)
    out_csv = save_dir / "all_results.csv"
    train_out_csv = save_dir / "train_results.csv"

    # Time
    job_start_iso = datetime.now().isoformat(timespec="seconds")
    t0_total = time.perf_counter()
    prep_sec = np.nan
    run_sec = np.nan
    metrics_sec = np.nan

    # 1) data
    try:
        t0 = time.perf_counter()
        with Timer(log, "prepare_data"):
            data = load_dataset(args)
        prep_sec = time.perf_counter() - t0
        print("[data] prepared")

        # 2) method
        t1 = time.perf_counter()
        with Timer(log, f"run_method:{args.method}"):
            pred_pack = run_method(args, data)
        run_sec = time.perf_counter() - t1
        print("[done] run_method finished")
    finally:
        hb.stop()

    # 3) metric
    t2 = time.perf_counter()
    report = compute_metrics(args, data, pred_pack)
    train_report = compute_metrics_train(args, data, pred_pack)
    metrics_sec = time.perf_counter() - t2

    # 4) result.csv
    total_sec = time.perf_counter() - t0_total
    time_info = {
        "start_time": job_start_iso,
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "time_prepare_data_sec": round(float(prep_sec), 4),
        "time_run_method_sec":   round(float(run_sec), 4),
        "time_metrics_sec":      round(float(metrics_sec), 4),
        "time_total_sec":        round(float(total_sec), 4),
    }
    if isinstance(pred_pack, dict) and ("note" in pred_pack):
        time_info["method_note"] = str(pred_pack["note"])

    # parameters
    hparams = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exp_name": exp_name,
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "q": getattr(args, "q", None),
        "agg_repeat": getattr(args, "agg_repeat", None),

        # Common
        "x_sensitive": args.x_sensitive,
        "sens_keys": args.sens_keys,
        "sens_thresh": args.sens_thresh,

        # Parameter
        "epochs": args.epochs,
        "lr": args.lr,

        # DRAF
        "lambda_fair": getattr(args, "lambda_fair", None),
        "n_low": getattr(args, "n_low", None),
        "lambda_fair": getattr(args, "lambda_fair", None),
        "n_low": getattr(args, "n_low", None),
        "n_low_frac": getattr(args, "n_low_frac", None),   # NEW
        "shrink_smallest_frac": getattr(args, "shrink_smallest_frac", None),  # NEW
        "shrink_seed": getattr(args, "shrink_seed", None),
    }

    hparams.update(time_info)
    row = {**hparams, **report}

    # 5) save
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
    print(f"[SAVE+APPEND] {out_csv} (+1 row)")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    train_df = pd.DataFrame([{**hparams, **train_report}])
    train_df.to_csv(train_out_csv, mode="a", header=not os.path.exists(train_out_csv), index=False)


if __name__ == "__main__":
    main()
