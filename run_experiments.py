# run_experiments.py
import argparse, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from fairbench.utils.seed import set_seed
from fairbench.datasets import load_dataset   # 통합 로더
from fairbench.methods import run_method      # 통합 메서드 실행
from fairbench.metrics import compute_metrics # accuracy, supIPM, subgroup metrics

def parse_args():
    p = argparse.ArgumentParser(description="Subgroup Fairness Bench")

    # Dataset & data options
    p.add_argument("--dataset", type=str, required=True,
                   choices=["toy_manyq", "adult", "communities", "dutch", "celebA"])
    p.add_argument("--data_dir", type=str, default="../data/raw/")
    p.add_argument("--q", type=int, default=100, help="toy_manyq sensitive count")

    # X–S handling (공통)
    p.add_argument("--x_sensitive", type=str, default="drop",
                   choices=["drop", "keep", "concat"],
                   help="drop: 민감 원본컬럼을 X에서 제거, keep: 유지, concat: S를 X에 붙여 f(x,s)로 학습")
    p.add_argument("--sens_keys", type=str, default=None,
                   help=("민감 컬럼 리스트 또는 프리셋 키워드. "
                         "Adult는 None이면 내부 기본 세트 사용, "
                         "Communities/Dutch는 None이면 각 로더의 'auto' 프리셋 사용. "
                         "예: 'race_basic' 또는 'sex,age,education_num'"))
    p.add_argument("--sens_thresh", type=float, default=0.5,
                   help="민감 수치 컬럼 이진화 분위수 임계(0~1)")

    # Dataset-specific optional paths
    p.add_argument("--dutch_path", type=str, default="data/raw/dutch.csv",
                   help="Dutch CSV 경로(기본 data/raw/dutch.csv)")
    # p.add_argument("--communities_names", type=str, default=None,
    #                help="communities.names 경로(기본 data_dir/raw/communities.names)")
    # p.add_argument("--communities_data", type=str, default=None,
    #                help="communities.data 경로(기본 data_dir/raw/communities.data)")
    p.add_argument("--tfds_data_dir", type=str, default=None,
                help="TFDS data_dir (없으면 기본 캐시)")
    p.add_argument("--celebA_manual_dir", type=str, default=None,
                help="CelebA 수동 다운로드 디렉토리(파일들 존재해야 함)")

    # Method selection
    p.add_argument("--method", type=str, required=True,
                   choices=["dr", "gerryfair", "multicalib", "sequential", "reduction"])

    # Reduction (fairlearn ExponentiatedGradient) baseline
    p.add_argument("--red_constraint", type=str, default="DP",
                   choices=["DP", "EO"], help="DP=DemographicParity, EO=EqualizedOdds")
    p.add_argument("--red_eps", type=float, default=0.02,
                   help="fairness slack (작을수록 제약 강함)")
    p.add_argument("--red_max_iter", type=int, default=50)
    p.add_argument("--red_base", type=str, default="logreg",
                   choices=["logreg", "linear_svm", "rf", "mlp_clf", "mlp_reg"],)

    # Common training hparams (필요시 각 메서드에서 선택 사용)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)

    # DR (our method)
    p.add_argument("--lambda_fair", type=float, default=0.0,
                   help="DR fairness weight (정확도-공정성 trade-off)")
    p.add_argument("--n_low", type=int, default=100,
                help="partial subgroup fairness: 최소 서브그룹 크기 임계값")
    # Data options (여기에 추가)
    p.add_argument("--shrink_smallest_frac", type=float, default=1.0,
                help="전체 n 대비 가장 작은 교차 서브그룹을 이 비율*n까지 다운샘플")
    p.add_argument("--shrink_seed", type=int, default=None,
                help="다운샘플 시드(미지정시 --seed 사용)")
    p.add_argument("--n_low_frac", type=float, default=None,
                help="partial subgroup: 최소지지 비율 (0~1). 지정 시 n_low보다 우선")


    # GerryFair
    p.add_argument("--gamma", type=float, default=0.01,
                   help="GerryFair gamma (페어니스 강도/허용 위반 수준)")
    p.add_argument("--gf_base", type=str, default="logistic",
                   choices=["logistic", "linear", "mlp_clf", "mlp_reg"])
    p.add_argument("--gf_max_iters", type=int, default=10)
    p.add_argument("--gf_C", type=float, default=50.0)
    p.add_argument("--gf_fairness", type=str, default="SP",
                   choices=["FP", "FN", "FPR", "FNR", "SP"])
    p.add_argument('--decision_threshold', type=float, default=0.5)

    # Multicalibration
    p.add_argument("--mc_alpha", type=float, default=0.1,
                   help="Multicalibration alpha (캘리브레이션 허용오차)")
    p.add_argument("--mc_lambda", type=float, default=0.1,
                   help="Multicalibration lambda (학습 스텝/규제 강도)")
    p.add_argument("--mc_max_iter", type=int, default=30)
    p.add_argument("--mc_randomized", action="store_true", default=True)
    p.add_argument("--mc_use_oracle", action="store_true", default=False)

    # Sequential Fairness
    p.add_argument("--seq_alpha", type=float, default=0.1,
                   help="Sequential Fairness의 alpha/step (기존 0.1을 인자로 노출)")
    p.add_argument("--seq_max_iter", type=int, default=50)

    # Results/logging
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="results")
    # --- logging / UX ---
    p.add_argument("--log-interval", type=int, default=50,
                        help="train step마다 로그 찍는 간격")
    p.add_argument("--progress-bar", action="store_true",
                        help="tqdm 진행바 사용")
    p.add_argument("--logfile", type=str, default="",
                        help="로그를 파일에도 기록 (경로 지정)")
    p.add_argument("--heartbeat-secs", type=int, default=0,
                        help="N초마다 heartbeat 로그 (0이면 끔)")
    p.add_argument("--results-csv", type=str, default="results/all_runs.csv",
                   help="실험 결과를 누적 저장할 CSV 경로")

    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # --- logger / heartbeat setup ---
    from fairbench.utils.logging_utils import setup_logger, Timer, Heartbeat
    log = setup_logger("fair", args.logfile or None)
    hb = Heartbeat(log, args.heartbeat_secs)
    hb.start()
    log.info(f"args: {{k: v for k, v in vars(args).items() if k not in ['password', 'token']}}")

    exp_name = args.exp_name or f"{args.dataset}_{args.method}_seed{args.seed}_{int(time.time())}"
    save_dir = Path(args.save_dir); save_dir.mkdir(exist_ok=True, parents=True)
    out_csv = save_dir / "all_results.csv"

    # ---- 타이밍 측정 시작 ----
    job_start_iso = datetime.now().isoformat(timespec="seconds")
    t0_total = time.perf_counter()
    prep_sec = np.nan
    run_sec = np.nan
    metrics_sec = np.nan

    # 1) 데이터 로딩 + 2) 방법 실행
    try:
        t0 = time.perf_counter()
        with Timer(log, "prepare_data"):
            data = load_dataset(args)
        prep_sec = time.perf_counter() - t0
        log.info("[data] prepared")

        t1 = time.perf_counter()
        with Timer(log, f"run_method:{args.method}"):
            pred_pack = run_method(args, data)
        run_sec = time.perf_counter() - t1
        log.info("[done] run_method finished")
    finally:
        hb.stop()

    # 3) 측정
    t2 = time.perf_counter()
    report = compute_metrics(args, data, pred_pack)
    metrics_sec = time.perf_counter() - t2

    # ---- 타이밍/메타 정보 구성 ----
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

    # data 로더가 meta를 담아줬다면 같이 기록(선택)
    meta = (data.get("meta", {}) if isinstance(data, dict) else {}) or {}

    def _as_str_list(x):
        if isinstance(x, (list, tuple)):
            return ",".join(map(str, x))
        return x

    # 실험/페어니스 관련 하이퍼파라미터 모으기
    hparams = {
        # 공통/식별
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exp_name": exp_name,
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,

        # 데이터/민감속성 설정
        "x_sensitive": args.x_sensitive,
        "sens_keys": args.sens_keys,
        "sens_thresh": args.sens_thresh,

        # 학습 공통
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,

        # DR (ours)
        "lambda_fair": getattr(args, "lambda_fair", None),
        "n_low": getattr(args, "n_low", None),
        "lambda_fair": getattr(args, "lambda_fair", None),
        "n_low": getattr(args, "n_low", None),
        "n_low_frac": getattr(args, "n_low_frac", None),   # NEW
        "shrink_smallest_frac": getattr(args, "shrink_smallest_frac", None),  # NEW
        "shrink_seed": getattr(args, "shrink_seed", None),

        # GerryFair
        "gamma": getattr(args, "gamma", None),
        "gf_max_iters": getattr(args, "gf_max_iters", None),
        "gf_C": getattr(args, "gf_C", None),
        "gf_fairness": getattr(args, "gf_fairness", None),

        # Multicalibration
        "mc_alpha": getattr(args, "mc_alpha", None),
        "mc_lambda": getattr(args, "mc_lambda", None),
        "mc_max_iter": getattr(args, "mc_max_iter", None),
        "mc_randomized": getattr(args, "mc_randomized", None),
        "mc_use_oracle": getattr(args, "mc_use_oracle", None),

        # Sequential
        "seq_alpha": getattr(args, "seq_alpha", None),
        "seq_max_iter": getattr(args, "seq_max_iter", None),

        # Reduction (fairlearn ExponentiatedGradient)
        "red_constraint": getattr(args, "red_constraint", None),
        "red_eps": getattr(args, "red_eps", None),
        "red_max_iter": getattr(args, "red_max_iter", None),
        "red_base": getattr(args, "red_base", None),

        # 로더 메타(있으면 기록)
        "x_sensitive_mode": meta.get("x_sensitive_mode"),
        "used_S_cols": _as_str_list(meta.get("used_S_cols")),
        "dropped_cols": _as_str_list(meta.get("dropped_cols")),
    }
    # 타이밍 정보 합치기
    hparams.update(time_info)

    # 한 행(row)으로 합치기: report가 동일 키를 갖고 있으면 report 값을 우선
    row = {**hparams, **report}

    # 4) 저장
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
    print(f"[SAVE+APPEND] {out_csv} (+1 row)")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
