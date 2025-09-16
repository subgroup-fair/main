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
SEEDS="${SEEDS:-2025 2026 2027 2028 2029}"
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
    # adult)
    #   echo "--dataset adult --sens_keys sex,race,age,marital-status --sens_thresh 0.5 --data_dir ../data/raw/"
    #   ;;
    # communities)
    #   echo "--dataset communities --sens_keys paper18 --sens_thresh 0.5 \
    #         --data_dir ../data/raw/"
    #   ;;
    # dutch)
    #   echo "--dataset dutch --sens_keys sex,age --sens_thresh 0.5 \
    #         --data_dir ../data/raw/"
    #   ;;
    toy_manyq)  
      echo "--dataset toy_manyq"
      ;;
    # celebA)
    #   # CelebA는 이미지 로더 내부에서 sens_keys=auto 처리한다고 가정
    #   echo "--dataset celebA --sens_keys auto --sens_thresh 0.5"
      # ;;
    *)
      echo "Unknown dataset: $ds" >&2; exit 1;;
  esac
}

# ===== 스윕 그리드 =====

QS=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50)   # toy_manyq용

# DR
DR_LAMS=(0.00 0.01 0.02 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00)
DR_NLOWS_CSV="${DR_NLOWS_CSV:-0,50,100,200,400}"
IFS=',' read -r -a DR_NLOWS <<< "$DR_NLOWS_CSV"

# Reduction
RED_GAMMAS=(0.005 0.01 0.02 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00)
RED_CONSTRAINT="${RED_CONSTRAINT:-DP}"
RED_MAX_ITER="${RED_MAX_ITER:-30}"

# GerryFair
GF_GAMMAS=(0.00001 0.00005 0.0001 0.0002 0.0005 0.001 0.005 0.01 0.02 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00)
# GF_GAMMAS=(5.00 10.00 50.00)
GF_MAX_ITERS="${GF_MAX_ITERS:-30}"
GF_FAIRNESS="${GF_FAIRNESS:-SP}"

# Multicalib
MC_ALPHAS=(0.01 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00 50.00 100.00)
MC_LAMBDAS=(0.01 0.05) # (0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00)
MC_MAX_ITER="${MC_MAX_ITER:-30}"

# Sequential
SEQ_ALPHAS=(0.001 0.005 0.01 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00 50.00 100.00)
SEQ_SCHED="${SEQ_SCHED:-}"     # const|linear|cosine|exp (비우면 전달 안함)
SEQ_WARMUP="${SEQ_WARMUP:-}"   # 에폭 단위 (비우면 전달 안함)

# ===== 커맨드 큐 =====
CMDS=()
enqueue() { CMDS+=("$*"); }

# ===== 메소드별 스윕 구성 =====
# build_dr_cmds() {
#   local ds extra seed lam nlow
#   for ds in toy_manyq; do
#     extra=$(ds_args "$ds")
#     for lam in "${DR_LAMS[@]}"; do
#       for nlow in "${DR_NLOWS[@]}"; do
#         for seed in "${SEED_ARR[@]}"; do
#           enqueue "$(gpu_prefix) $PY run_experiments2.py $extra --method dr \
#             --lambda_fair $lam --n_low $nlow \
#             --x_sensitive $X_SENSITIVE --seed $seed \
#             --save_dir \"$SAVE_DIR/dr_$ds\""
#         done
#       done
#     done
#   done
# }

build_dr_subgroup_subset_3q_cmds() {
  local ds extra seed lam nlow
  for q in "${QS[@]}"; do
    for ds in toy_manyq; do
      extra=$(ds_args "$ds")
      for lam in "${DR_LAMS[@]}"; do
        for nlow in "${DR_NLOWS[@]}"; do
          for seed in "${SEED_ARR[@]}"; do
            enqueue "$(gpu_prefix) $PY run_experiments2.py $extra --method dr_subgroup_subset_3q \
              --lambda_fair $lam --n_low $nlow --q $q \
              --x_sensitive $X_SENSITIVE --seed $seed \
              --save_dir \"../0910_$SAVE_DIR/dr_$ds\""
          done
        done
      done
    done
  done
}

build_reduction_cmds() {
  local ds extra seed r
  for q in "${QS[@]}"; do
    for ds in toy_manyq; do
      extra=$(ds_args "$ds")
      for r in "${RED_GAMMAS[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
          enqueue "$(gpu_prefix) $PY run_experiments2.py $extra --method reduction \
            --red_constraint $RED_CONSTRAINT --red_max_iter $RED_MAX_ITER --red_eps $r \
            --x_sensitive $X_SENSITIVE --seed $seed --q $q \
            --save_dir \"../0910_$SAVE_DIR/reduction_$ds\""
        done
      done
    done
  done
}

build_gerryfair_cmds() {
  local ds extra seed g
  for q in "${QS[@]}"; do
    for ds in toy_manyq; do
      extra=$(ds_args "$ds")
      for g in "${GF_GAMMAS[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
          
          enqueue "$(gpu_prefix) $PY run_experiments2.py $extra --method gerryfair \
            --gamma $g --gf_max_iters $GF_MAX_ITERS --gf_fairness $GF_FAIRNESS \
            --x_sensitive $X_SENSITIVE --seed $seed --q $q \
            --save_dir \"../0910_$SAVE_DIR/gerryfair_$ds\""
        done
      done
    done
  done
}

build_multicalib_cmds() {
  local ds extra seed a l
  for q in "${QS[@]}"; do
    for ds in toy_manyq; do
      extra=$(ds_args "$ds")
      for a in "${MC_ALPHAS[@]}"; do
        for l in "${MC_LAMBDAS[@]}"; do
          for seed in "${SEED_ARR[@]}"; do
          
            enqueue "$(gpu_prefix) $PY run_experiments2.py $extra --method multicalib \
              --mc_alpha $a --mc_lambda $l --mc_max_iter $MC_MAX_ITER \
              --x_sensitive $X_SENSITIVE --seed $seed --q $q \
              --save_dir \"../0910_$SAVE_DIR/mc_$ds\""
          done
        done
      done
    done
  done
}

build_sequential_cmds() {
  local extra seed a sched_flags=""
  for q in "${QS[@]}"; do
    for ds in toy_manyq; do
      extra=$(ds_args "$ds")
      #   extra=$(ds_args "celebA")   # ★ CelebA만
      #   # 선택적 스케줄 플래그
      #   [[ -n "$SEQ_SCHED"  ]] && sched_flags+=" --seq_sched $SEQ_SCHED"
      #   [[ -n "$SEQ_WARMUP" ]] && sched_flags+=" --seq_warmup $SEQ_WARMUP"
      for a in "${SEQ_ALPHAS[@]}"; do
        for seed in "${SEED_ARR[@]}"; do
        
          enqueue "$(gpu_prefix) $PY run_experiments2.py $extra --method sequential \
              --seq_alpha $a $sched_flags \
              --x_sensitive $X_SENSITIVE --seed $seed --q $q \
              --tfds_data_dir data/tfds \
              --save_dir \"$SAVE_DIR/seq_$ds\""
        done
      done
    done
  done
}

# ===== 진입점 =====
METHOD="${1:-}"
if [[ -z "$METHOD" ]]; then
  echo "Usage: $0 {dr|gerryfair|multicalib|sequential|reduction}" >&2
  exit 1
fi

# ===== 진입점 =====
METHOD="${1:-}"
if [[ -z "$METHOD" ]]; then
  echo "Usage: $0 {dr|gerryfair|multicalib|sequential|reduction}" >&2
  exit 1
fi

case "$METHOD" in
  dr)          build_dr_subgroup_subset_3q_cmds ;;
  gerryfair)   build_gerryfair_cmds ;;
  multicalib)  build_multicalib_cmds ;;
  reduction)  build_reduction_cmds ;;
  sequential)  build_sequential_cmds ;;   # ★ CelebA only
  *) echo "Unknown method: $METHOD" >&2; exit 1 ;;
esac

# ===== 실행 (GNU parallel 선호, 미설치 시 xargs/백그라운드) =====
echo "[INFO] Launching ${#CMDS[@]} jobs (JOBS=$JOBS, METHOD=$METHOD, GPU_IDS='${GPU_IDS:-none}', SEEDS='${SEEDS}')"
if command -v parallel >/dev/null 2>&1; then
  printf '%s\n' "${CMDS[@]}" | parallel -j "$JOBS"
elif command -v xargs >/dev/null 2>&1; then
  printf '%s\0' "${CMDS[@]}" | xargs -0 -I {} -P "$JOBS" bash -lc "{}"
else
  for c in "${CMDS[@]}"; do bash -lc "$c" & done
  wait
fi

echo "[DONE] $METHOD sweeps finished."
