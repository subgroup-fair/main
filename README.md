# Quick README
---
### Experiment Checklist
실험 우선순위를 작성한 리스트입니다.. 다음 순서로 진행되고 있습니다.. 최우선목표는 어떤 결과든 뽑는 것...
#### 실험 결과에 관하여
- [v] 22q vs 3q
- [ ] E1) 데이터: synthetic 조절변수:q
- [ ] E2,3) 데이터: Real(A,C,D) 조절변수: n_low
- [ ] E4) DR vs SupIPM 조절변수: lambda
- [ ] E7,8)
- [ ] E5,6)

#### 코드 변경에 관하여
- [ ] 현재 코드에 하드코딩된 부분 조절 가능하도록 변경, 조절하면서 실험
      - gamma(n_low/n 비율)
      - q(sensitive attribute 개수) q가 커질수록 가장 작은 subgroup의 size도 감소하도록 빌드
      - real data에서 subgroup 하나의 샘플 개수 극단적으로 줄여보기(1,3,5,7, ...)
- [ ] subgroup => subgroup subset
- [ ] ```sup_mmd_gaussian(scores, groups, V, sigma), sup_wasserstein_1d(scores, groups, V)```
- [ ] synthetic data로 실험
- [v] d(P_s,P_.) => d(P_s,P_s^c)
      - metrics/supipm.py 에서 ```max_s over {0,1}^q```
- [v] 현재 코드에 대한 이해, 디버깅, 신뢰성검증
- [v] 각 서브그룹의 샘플 수 저장하도록 만들기
      - fairbench/__init__.py main 함수에서 조절
- [v] V 구성 전략 비교
      - 현재(서브그룹만), 원소n개짜리 서브셋만 만들기, 민감변수에 대해 0,1,all 경우로 선택 3^q
- [v] fairness measure는 8가지 (현재 논문의 MMD, WD, DR 학습 시 썼던 \mathcal{W}에 대한 supMMD, supWD, worstSUBGROUP, avgSUBGROUP, test data에서 그룹빈도로 가중치줘서 계산된 sum, sup subgroup fairness까지 추가)
- [v] 22q실험 -> 오래걸리면 실용적방법 실행 -> 둘이 결과 비슷하다 리포트

#### Q1
- [ ] DR 학습 안정화
      - fairbench/methods/dr.py
      - validation 활용 얼리스타핑 / discriminator 구조 변경

- [ ] 결과 그림 예쁘게 만들기



---

### 데이터
- Adult `data/raw/adult.csv`
    - --dataset adult --sens_keys sex,race,age,marital-status --sens_thresh 0.5
    - 4가지 S를 쓰고, age 같은 연속형 변수는 0.5 quantile (중앙값)으로 binarize

- Comminities & Crime `data/raw/communities.data`, `data/raw/communities.names`
    - --dataset communities --sens_keys paper18 --sens_thresh 0.5 \ --communities_names data/raw/communities.names --communities_data data/raw/communities.data
    - 18가지 S를 씀 (GerryFair에서 그랬다고 써있음). 근데 이 중에 서로 correlation 있는 변수들인 것 같은데.. GerryFair에서 그렇게 했다고 하니 우선 유지.
- Dutch `data/raw/dutch.csv`
    - --dataset dutch --sens_keys sex,age --sens_thresh 0.5 --dutch_path data/raw/dutch.csv
    - 일부러 2개만 써보자. 이 때에는 기존하고 비슷한데, 여러 개 (Adult나 Communities) S 일 때 우리 거가 좋다는 얘기할 수도 있으니.

### 실행 (예시)
`run_sweep.sh`를 실행

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

- 여기서 X_SENSITIVE=drop은 f에 S를 입력 변수로 안 쓴다는 의미인데, GerryFair는 자체적으로 S를 쓴다고 알고 있어서 (확실치 않음) drop으로 넣어야 구조적 에러 없이 S를 입력 변수로 쓸 수 있을 듯.

### Arguments

- DR
    - `lambda_fair'로 fairness 조절.
    - n_low_frac으로 논문의 gamma (최소 subgroup size) 조절
    - 
- Reduction
    - `red_eps'로 fairness 조절.
- GerryFair
    - `gamma'로 fairness 조절.
- Multicalibration
    - `mc_alpha'로 fairness 조절.
- Sequential
    - `seq_alpha'로 fairness 조절.

### 결과 저장
- `results/[ALG]_[DATA]/all_results.csv` 에 저장됨



---------

## all_results.csv 해석하기

- Meta

    timestamp: 이 결과 행이 기록된 시각(UTC/로컬 포맷).

    exp_name: 사용자가 지정한 실험 이름(재현성·로그 구분용).

    dataset: 사용한 데이터셋 식별자(예: adult, toy_manyq…).

    method: 실행한 알고리즘(dr, sequential, multicalib, gerryfair, reduction 등).

    seed: 난수 시드(데이터 split/초기화 등에 영향).

- 민감속성(S) 설정/전처리

    x_sensitive: 학습 입력 X에 S를 어떻게 다루는지 (drop/keep/concat).

    sens_keys: 서브그룹을 정의할 때 쓴 민감속성(컬럼 이름/인덱스 목록).

    sens_thresh: 연속형 S 이진화·분할 기준(예: quantile:0.2, >=:30 등; 범주형은 값 셋).

    x_sensitive_mode: S를 보존/결합할 때의 인코딩 방식(예: one-hot/원값/기본값).

- 공통 학습 하이퍼파라미터

    epochs: 에폭 수.

    batch_size: 배치 크기.

    lr: 학습률.

    lambda_fair: 공정성 항 가중치(방법에 따라 사용/미사용).

    gamma: 공정성-정확도 트레이드오프 관련 추가 튜닝 노브(방법별 의미 상이; 미사용 시 NaN).

- DR/서브그룹 구성 관련

    n_low: (절대 개수) 하위 비율/조건으로 생성된 후보 서브그룹 중 유지할 수(구버전 파라미터).

    n_low_frac: (0~1) 위의 비율 버전; 전체 n 대비 하위 부분을 몇 % 사용할지.

    shrink_smallest_frac: (0~1) “가장 작은 서브그룹”을 의도적으로 축소하는 비율. 예: 0.05면 해당 그룹 샘플을 5%만 남김(데이터 전체에서 먼저 축소 후 train/val/test split).

shrink_seed: shrink 샘플링에 쓸 시드.

- GerryFair (gf_*)

    gf_max_iters: Auditor–Learner 반복(부스팅) 라운드 수.

    gf_C: Learner(보통 로지스틱 회귀)의 규제 강도(Scikit-learn의 C).

    gf_fairness: GerryFair에서 쓰는 공정성 제약 종류(예: SP, EO).

- Multicalibration (mc_*)

    mc_alpha: 캘리브레이션 업데이트 스텝 크기/학습률.

    mc_lambda: 정규화 강도(방법 구현에 따라 의미 상이).

    mc_max_iter: 멀티캘리브레이션 반복 횟수 상한.

    mc_randomized: 무작위화(랜덤 라운딩 등) 사용 여부(Boolean).

    mc_use_oracle: 오라클/이상적인 서브그룹 접근 가정 사용 여부(Boolean).

- Sequential/Auditing (seq_*)

    seq_alpha: 순차 감사/수정 스킴에서의 스텝 크기.

    seq_max_iter: 순차 반복 상한.

- Reductions (red_*)

    red_constraint: 리덕션 제약 종류(예: dp, eo).

    red_eps: 허용 위반 허용치(슬랙) — 작을수록 강한 공정성.

    red_max_iter: 비용-민감 분류기 호출/반복 상한.

    red_base: 리덕션에서 쓰는 베이스 모델(예: logreg, mlp).

- 컬럼 관리(재현성)

    used_S_cols: 전처리 후 실제로 사용된 S 컬럼 목록.

    dropped_cols: 로딩/전처리 시 제거한 컬럼들(ID, 누설 가능 컬럼, 혹은 x_sensitive='drop'로 제거된 S 등).

- 실행 시간/구간

    start_time, end_time: 실행 시작/종료 시각.

    time_prepare_data_sec: 데이터 준비(로딩/전처리) 시간(초).

    time_run_method_sec: 학습/알고리즘 실행 시간(초).

    time_metrics_sec: 메트릭 계산 시간(초).

    time_total_sec: 전체 소요 시간(초).

- 성능/공정성 지표

    accuracy: 테스트 정확도(일반적으로 검증셋에서 임계값 튜닝 후 테스트에 평가).

    supipm_rbf: RBF 커널 기반 sup-IPM(서브그룹 함수족에 대한 최악의 평균 차이; 낮을수록 좋음).

    supipm_w1: Lipschitz(=W1 계열) 함수족 기반 sup-IPM(예상 분포 차이; 낮을수록 좋음).

    spd_worst, spd_mean: 통계적 패리티 차이( |P(ŷ=1|g)−P(ŷ=1|g′)| )의 최악값/평균값.

    fpr_worst, fpr_mean: 거짓양성률(FPR) 격차 최악/평균.

    mc_worst, mc_mean: 멀티캘리브레이션 위반( |E[Y−p(X) | g, bin]| ) 최악/평균.

    marg_spd_worst, marg_spd_mean: 단일 속성별(주변, marginal) 서브그룹만 고려했을 때의 SPD 최악/평균(교차/결합 서브그룹은 제외).

    marg_fpr_worst, marg_fpr_mean: 단일 속성 기준의 FPR 격차 최악/평균.

    marg_mc_worst, marg_mc_mean: 단일 속성 기준의 멀티캘리브레이션 위반 최악/평균.
