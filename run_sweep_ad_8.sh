#!/usr/bin/env bash
set -euo pipefail

# 병렬 갯수 (환경변수로 조절)
JOBS="${JOBS:-4}"

# 공통 런타임 튜닝
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# 파이썬 실행기
PY="${PY:-python}"

# 결과 저장 루트
SAVE_DIR="${SAVE_DIR:-results}"

# x_sensitive 모드 (concat | drop)
X_SENSITIVE="${X_SENSITIVE:-concat}"

# 반복할 시드들
SEEDS="${SEEDS:-1 11 21 31 41}"
read -r -a SEED_ARR <<< "$SEEDS"

############################################
# GPU 라운드로빈 배정 (CUDA_VISIBLE_DEVICES)
############################################
if [[ -z "${GPU_IDS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NGPU_DET="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$NGPU_DET" =~ ^[0-9]+$ ]] && (( NGPU_DET > 0 )); then
      GPU_IDS="$(seq -s, 0 $((NGPU_DET-1)))"
    else
      GPU_IDS=""
    fi
  else
    GPU_IDS=""
  fi
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS:-}"
NGPU=${#GPUS[@]}
IDX=0
gpu_prefix() {
  if (( NGPU > 0 )); then
    local id="${GPUS[$((IDX % NGPU))]}"
    IDX=$((IDX+1))
    echo "CUDA_VISIBLE_DEVICES=$id"
  fi
}

# ===== 데이터셋별 인자: sens_keys는 첫 4줄과 동일, + CelebA 추가 =====
ds_args() {
  local ds="$1"
  case "$ds" in
    adult)
      echo "--dataset adult --sens_keys sex,race,age,marital-status --sens_thresh 0.5 --data_dir ../data/raw/"
      ;;
    communities)
      echo "--dataset communities --sens_keys paper18 --sens_thresh 0.5 \
            --data_dir ../data/raw/"
      ;;
    dutch)
      echo "--dataset dutch --sens_keys sex,age --sens_thresh 0.5 \
            --data_dir ../data/raw/"
      ;;
    # toy_manyq)  
    #   echo "--dataset toy_manyq --n 2000 --d 10 --q 4"
    #   ;;
    # celebA)
    #   # CelebA는 이미지 로더 내부에서 sens_keys=auto 처리한다고 가정
    #   echo "--dataset celebA --sens_keys auto --sens_thresh 0.5"
      # ;;
    *)
      echo "Unknown dataset: $ds" >&2; exit 1;;
  esac
}

# ===== 스윕 그리드 =====
# DR
DR_LAMS=(0.00 0.01 0.02 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00 20.0 50.0 100.0 200.0 500.0 1000.0 2000.0 5000.0 10000.0)
# DR_LAMS=(0.00)
DR_NLOWS_FRAC=(0.05 0.1 0.2)
DR_NLOWS_CSV="${DR_NLOWS_CSV:-0}"
IFS=' ' read -r -a DR_NLOWS <<< "$DR_NLOWS_CSV"

# Reduction
RED_GAMMAS=(0.005 0.01 0.02 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00)
RED_CONSTRAINT="${RED_CONSTRAINT:-DP}"
RED_MAX_ITER="${RED_MAX_ITER:-30}"

# GerryFair
# GF_GAMMAS=(0.00001 0.00005 0.0001 0.0002 0.0005 0.001 0.005 0.01 0.02 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00)
# GF_GAMMAS=(1.0)
GF_CS=(0.0 0.1 0.5 1.0 5.0 20.0 50.0 200.0 500.0 1000.0 5000.0)
# GF_CS=(0.0)
GF_MAX_ITERS="${GF_MAX_ITERS:-200}"
GF_FAIRNESS="${GF_FAIRNESS:-SP}"

# Multicalib
MC_ALPHAS=(0.01 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00 50.00 100.00)
MC_LAMBDAS=(0.01 0.05) # (0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00)
MC_MAX_ITER="${MC_MAX_ITER:-30}"

# Sequential
SEQ_ALPHAS=(0.001 0.005 0.01 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00 50.00 100.00)
SEQ_SCHED="${SEQ_SCHED:-}"     # const|linear|cosine|exp (비우면 전달 안함)
SEQ_WARMUP="${SEQ_WARMUP:-}"   # 에폭 단위 (비우면 전달 안함)

# Reg
# Sequential
REG_LAMS=(10.0 5.0 2.0 1.0 0.7 0.5 0.3 0.2 0.1 0.05 0.01 0.005 0.002 0.001 0.0)
# REG_LAMS=(0.0)
# REG_LAMS=(0.005 0.002 0.001 0.0)


# Apriori repeat
AGG_REPEATS=(4096)




# ===== 커맨드 큐 =====
CMDS=()
enqueue() { CMDS+=("$*"); }

# ===== 메소드별 스윕 구성 =====
# build_dr_cmds() {
#   local ds extra seed lam nlow
#   for ds in adult dutch communities; do
#     extra=$(ds_args "$ds")
#     for lam in "${DR_LAMS[@]}"; do
#       for nlow in "${DR_NLOWS[@]}"; do
#         for seed in "${SEED_ARR[@]}"; do
#           enqueue "$(gpu_prefix) $PY run_experiments_3q.py $extra --method dr \
#             --lambda_fair $lam --n_low $nlow \
#             --x_sensitive $X_SENSITIVE --seed $seed \
#             --save_dir \"$SAVE_DIR/dr_$ds\""
#         done
#       done
#     done
#   done
# }



build_dr_subgroup_subset_3q_cmds() {
  local ds extra seed lam nlow nlow_frac
  for ds in communities adult dutch; do
    extra=$(ds_args "$ds")
    for lam in "${DR_LAMS[@]}"; do
      for nlow_frac in "${DR_NLOWS_FRAC[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
          for agg_repeat in "${AGG_REPEATS[@]}"; do
            enqueue "$(gpu_prefix) $PY run_experiments_3q.py $extra --method dr_subgroup_subset_random \
              --lambda_fair $lam --n_low_frac $nlow_frac --agg_repeat $agg_repeat \
              --x_sensitive $X_SENSITIVE --seed $seed --fair_warmup_epochs 100 \
              --fair_conf_gamma 1.5 --fair_margin 0.01 --fair_adv_steps 1 --dr_temp_gz 0.7 --agg_max_len 100 \
              --save_dir \"../1009_mlp_$SAVE_DIR/$nlow_frac/dr_$ds\""
          done
        done
      done
    done
  done
}

build_reduction_cmds() {
  local ds extra seed r
  for ds in adult dutch communities; do
    extra=$(ds_args "$ds")
    for r in "${RED_GAMMAS[@]}"; do
      for seed in "${SEED_ARR[@]}"; do
        enqueue "$(gpu_prefix) $PY run_experiments_3q.py $extra --method reduction \
          --red_constraint $RED_CONSTRAINT --red_max_iter $RED_MAX_ITER --red_eps $r \
          --x_sensitive $X_SENSITIVE --seed $seed \
          --save_dir \"../1009_mlp_$SAVE_DIR/reduction_$ds\""
      done
    done
  done
}


build_gerryfair_cmds() {
  local ds extra seed g
  for ds in adult dutch communities; do
    extra=$(ds_args "$ds")
    for c in "${GF_CS[@]}"; do
      for seed in "${SEED_ARR[@]}"; do
        enqueue "$(gpu_prefix) $PY run_experiments_3q.py $extra --method gerryfair \
          --gf_C $c --gf_max_iters $GF_MAX_ITERS --gf_fairness $GF_FAIRNESS \
          --x_sensitive concat --seed $seed --fair_warmup_epochs 100 \
          --fair_conf_gamma 1.5 --fair_margin 0.01 --fair_adv_steps 1 --dr_temp_gz 0.7 --agg_max_len 100 \
          --save_dir \"../1009_mlp_$SAVE_DIR/gerryfair_$ds\""
      done
    done
  done
}

build_multicalib_cmds() {
  local ds extra seed a l
  for ds in adult dutch communities; do
    extra=$(ds_args "$ds")
    for a in "${MC_ALPHAS[@]}"; do
      for l in "${MC_LAMBDAS[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
          enqueue "$(gpu_prefix) $PY run_experiments_3q.py $extra --method multicalib \
            --mc_alpha $a --mc_lambda $l --mc_max_iter $MC_MAX_ITER \
            --x_sensitive $X_SENSITIVE --seed $seed \
            --save_dir \"../1009_mlp_$SAVE_DIR/mc_$ds\""
        done
      done
    done
  done
}

build_sequential_cmds() {
  local extra seed a sched_flags=""
  for ds in adult dutch communities; do
    extra=$(ds_args "$ds")
    #   extra=$(ds_args "celebA")   # ★ CelebA만
    #   # 선택적 스케줄 플래그
    #   [[ -n "$SEQ_SCHED"  ]] && sched_flags+=" --seq_sched $SEQ_SCHED"
    #   [[ -n "$SEQ_WARMUP" ]] && sched_flags+=" --seq_warmup $SEQ_WARMUP"
    for a in "${SEQ_ALPHAS[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
        enqueue "$(gpu_prefix) $PY run_experiments_3q.py $extra --method sequential \
            --seq_alpha $a $sched_flags \
            --x_sensitive $X_SENSITIVE --seed $seed \
            --tfds_data_dir data/tfds \
            --save_dir \"../1009_mlp_$SAVE_DIR/sequential_$ds\""
        done
    done
  done
}

build_unfair_cmds() {
  local ds extra seed
  # 선택 옵션(없으면 기본값)
  local base="${BASE:-mlp_clf}"   # logreg | linear_svm | rf | mlp_clf | mlp_reg
  for ds in adult dutch communities; do
    extra=$(ds_args "$ds")
    for seed in "${SEED_ARR[@]}"; do
      enqueue "$(gpu_prefix) $PY run_experiments_3q.py $extra --method unfair \
        --red_base $base \
        --x_sensitive $X_SENSITIVE --seed $seed \
        --save_dir \"../1009_mlp_$SAVE_DIR/unfair_$ds\""
    done
  done
}

build_reg_cmds() {
  local ds extra seed lam
  for ds in adult dutch communities; do
    extra=$(ds_args "$ds")
    for lam in "${REG_LAMS[@]}"; do
      for seed in "${SEED_ARR[@]}"; do
        enqueue "$(gpu_prefix) $PY run_experiments_3q.py $extra --method reg \
          --mf_lambda $lam --mf_base mlp --fair_warmup_epochs 100 \
          --fair_conf_gamma 1.5 --fair_margin 0.01 --fair_adv_steps 1 --dr_temp_gz 0.7 --agg_max_len 100 \
          --x_sensitive $X_SENSITIVE --seed $seed \
          --save_dir \"../1009_mlp_$SAVE_DIR/reg_$ds\""
      done
    done
  done
}




# ===== 진입점 =====
METHOD="${1:-}"
if [[ -z "$METHOD" ]]; then
  echo "Usage: $0 {dr|gerryfair|multicalib|sequential|reduction|reg|unfair}" >&2
  exit 1
fi

case "$METHOD" in
  unfair)      build_unfair_cmds ;;
  dr)          build_dr_subgroup_subset_3q_cmds ;;
  gerryfair)   build_gerryfair_cmds ;;
  multicalib)  build_multicalib_cmds ;;
  reduction)  build_reduction_cmds ;;
  sequential)  build_sequential_cmds ;;   # ★ CelebA only
  reg)  build_reg_cmds ;;
  *) echo "Unknown method: $METHOD" >&2; exit 1 ;;
esac

# ===== 실행 (GNU parallel 선호, 미설치 시 xargs/백그라운드) =====
echo "[INFO] Launching ${#CMDS[@]} jobs (JOBS=$JOBS, METHOD=$METHOD, GPU_IDS='${GPU_IDS:-none}', SEEDS='${SEEDS}')"
if command -v parallel >/dev/null 2>&1; then
  printf '%s\n' "${CMDS[@]}" | parallel -j "$JOBS"
elif command -v xargs >/dev/null 2>&1; then
  printf '%s\0' "${CMDS[@]}" | xargs -0 -I {} -P "$JOBS" bash -lc "{}"
fi

echo "[DONE] $METHOD sweeps finished."