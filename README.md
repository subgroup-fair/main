# Quick README

---

### Data
- Adult `data/raw/adult.csv`
    - --dataset adult --sens_keys sex,race,age,marital-status --sens_thresh 0.5

- Comminities & Crime `data/raw/communities.data`, `data/raw/communities.names`
    - --dataset communities --sens_keys paper18 --sens_thresh 0.5 \ --communities_names data/raw/communities.names --communities_data data/raw/communities.data
- Dutch `data/raw/dutch.csv`
    - --dataset dutch --sens_keys sex,age --sens_thresh 0.5 --dutch_path data/raw/dutch.csv

### Run 
`run_sweep.sh`

```bash
# DR
GPU_IDS=0,1 JOBS=6 ./run_sweep.sh dr
# GerryFair
GPU_IDS=0,1 JOBS=6 X_SENSITIVE=drop ./run_sweep.sh gerryfair
# Multicalibration
GPU_IDS=2,3 JOBS=6 ./run_sweep.sh multicalib
# Reduction
GPU_IDS=2,3 JOBS=6 ./run_sweep.sh reduction
# Sequential
GPU_IDS=2,3 JOBS=6 ./run_sweep.sh sequential
```

### Arguments

- DR
    - lambda_fair: Fairness trade-off hyperparameter
    - n_low_frac: Gamma in the paper.

- Reduction
    - `red_eps': Fairness trade-off hyperparameter
      
- GerryFair
    - `gamma': Fairness trade-off hyperparameter
      
- Multicalibration
    - `mc_alpha': Fairness trade-off hyperparameter
      
- Sequential
    - `seq_alpha': Fairness trade-off hyperparameter

### Save the result
- `results/[ALG]_[DATA]/all_results.csv` 



--------

    marg_mc_worst, marg_mc_mean: 단일 속성 기준의 멀티캘리브레이션 위반 최악/평균.
