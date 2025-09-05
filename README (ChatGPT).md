
--------

# 아래는 GPT가 코드 보고 써준 README


**환경변수**
- `JOBS` : 동시에 실행할 잡 수 (기본 `4`)
- `GPU_IDS` : `0,1,2,...` 형식. 비우면 GPU 라운드로빈 미사용
- `SAVE_DIR` : 결과 루트(기본 `results`)
- `X_SENSITIVE` : `concat | drop`  
  - **drop**: 원본 민감 raw 컬럼을 특징에서 제거 (**권장**)  
  - **concat**: 전처리된 `X`에 **이진 `S`**(민감 변수)를 붙여 학습
- `SEEDS` : 반복 시드 목록 (기본 `2025 2026 2027 2028 2029`)

**데이터셋별 기본 인자(`ds_args`)**
- **adult**  
  `--dataset adult --sens_keys sex,race,age,marital-status --sens_thresh 0.5`
- **communities**  
  `--dataset communities --sens_keys paper18 --sens_thresh 0.5 \
  --communities_names data/raw/communities.names --communities_data data/raw/communities.data`
- **dutch**  
  `--dataset dutch --sens_keys sex,age --sens_thresh 0.5 --dutch_path data/raw/dutch.csv`


> `sens_keys`: 사용할 **이진 민감 변수들**. Adult에서는 `sex_Male, race_White, age_ge_40, married`가 생성됩니다.  
> `sens_thresh`: 연속형을 이진화할 때의 분위수 컷(기본 0.5).

---

## 2) 베이스라인 – 공정성 파라미터 요약

| 방법 | 핵심 공정성 파라미터 | 의미(한 줄 요약) |
|---|---|---|
| **DR** | `--lambda_fair` | 최악 subgroup에 주는 가중치(↑: 더 엄격). |
|  | `--n_low`, `--n_low_frac` | DRO에 포함할 **최소 지원**(절대/비율) 컷오프. |
| **Reduction (EG)** | `--red_constraint` | 제약 유형(`DP`, `EOP` 등). -> `DP` 써야함 !! |
|  | `--red_eps` | 허용 격차(ε). **작을수록 더 엄격**. |
|  | `--red_max_iter` | EG 반복 횟수. |
| **GerryFair** | `--gf_fairness` | `SP`/`EO` 등 감사 기준. -> `SP` 써야함 !! |
|  | `--gamma` | 허용 격차(γ). **작을수록 더 엄격**. |
|  | `--gf_max_iters` | 감사–업데이트 반복 횟수. |
| **Multicalibration** | `--mc_alpha` | 보정 오차 임계(α). **작을수록 더 엄격**. |
|  | `--mc_lambda` | 업데이트 스텝/규제 강도. |
|  | `--mc_max_iter` | 반복 횟수. |
| **Sequential** | `--seq_alpha` | 공정성 페널티 세기. |
|  | `--seq_sched` | `const|linear|cosine|exp` 스케줄. |
|  | `--seq_warmup` | 페널티 도입 워ーム업 에폭. |

> 모든 방법은 **교차 서브그룹(교차 조합)** 기준으로 동작(마지널 X).

---

## 3) DR(Distributionally Robust) 상세

DR은 **교차 subgroup** 단위의 **최악 집단(worst-group) 손실**을 줄이도록 평균 손실과 트레이드오프를 조절합니다.

### 핵심 파라미터
- `--lambda_fair (λ)`  
  - **역할**: 평균(ERM) 손실 ↔ **최악 subgroup 손실** 가중치.  
  - **크게** → 최악 subgroup 개선(공정/강건 ↑), 평균 정확도는 다소 ↓ 가능.  
  - **작게** → 평균 성능 위주(ERM에 가까움).

- `--n_low`, `--n_low_frac`  **(최소 지원 컷오프)**
  - DR 내부의 `build_C_tensor`가 **민감 컬럼**별로 양/음 클래스(0/1) 모두 **최소 지원 이상인 컬럼만** 선택해 공정성 항을 구성합니다.  
    - `n_low` : **절대 개수** 임계.  
    - `n_low_frac` : **비율 임계**. **우선순위가 더 높으며**, `ceil(n_low_frac * N)`을 임계로 사용합니다. 여기서 `N`은 **(탭ular) 학습용 S의 샘플 수**, **(이미지) 현재 배치 크기**입니다.  
  - 예) `n_low_frac=0.05`이고 학습 세트가 20,000개면 임계는 `ceil(1,000)`입니다. 어떤 민감 컬럼이든 **0/1 양쪽이 모두** 이 임계 이상일 때만 DRO에 포함됩니다.  
  - 팁: 이미지에서 `n_low_frac`는 **배치 기준**으로 계산되므로, **작은 배치**에선 임계가 작아질 수 있습니다. 안정적으로 쓰려면 배치를 키우거나 `n_low`를 사용하세요.

- `--x_sensitive = drop|concat`  
  - **drop** 권장. `concat`은 `f(x,s)` 모델 비교나 감사/후처리 결합 실험 시 사용.

### 권장 스윕
- `λ`: `0.00 0.01 0.02 0.05 0.10 0.20 0.30 0.50 0.70 1.00 1.20 1.50 2.00 5.00 10.00`  
- `n_low`: `0,50,100,200,400` (또는 `n_low_frac`: `0.01, 0.02, 0.05`)



**해석 팁**
- λ↑ → 최악 subgroup 지표(에러/EO gap 등) 개선되는지 확인  
- `n_low / n_low_frac`↑ → 극소수 패턴(너무 작아 불안정)을 **DRO 대상에서 제외**하여 안정성↑ (대신 해당 소수 패턴 공정성 반영도는 ↓)

---

## 4) 데이터 축소 옵션 – `shrink_smallest_frac`

각 로더(`adult.py`, `communities.py`, `dutch.py`)는 **split 이전** 단계에서 선택적으로 전체 데이터를 축소할 수 있습니다.

- 인자: `--shrink_smallest_frac <float>` (기본 1.0 → 비활성)  
- **0 < frac < 1.0** 이면, `(Xp, y, S)`를 **전역적으로 일관된 랜덤 서브샘플**로 축소합니다.  
  - 구현은 **서브그룹 정보를 보존**하도록 설계되어, **각 subgroup의 비율은 기대적으로 유지**되며 **절대 샘플 수만 감소**합니다.  
  - 반환되는 `shrink_info`에 축소 적용 여부/시드 등이 담겨 있습니다.
- **효과**
  - **훈련/검증/테스트 split 전에 적용**되므로, 학습 시간 단축/스윕 가속에 유용합니다.
  - 축소 후 학습 세트 크기 `N`이 줄기 때문에 **`n_low`/`n_low_frac` 임계에 간접적 영향**이 있습니다. (예: `n_low_frac`는 `ceil(frac * N)`이므로 N 축소 시 임계도 자동으로 낮아짐)

> 실험을 정확히 재현하려면 `--shrink_smallest_frac=1.0`(비활성)을 권장하고, 탐색 단계에서만 0.5~0.8 등으로 속도 튜닝하세요.

---

## 5) 민감 변수(S)와 Subgroup 집계

### 이진화/정규화 요지
- **Adult**: `sex_Male`, `race_White`, `age_ge_40`, `married` (모두 **이진**).  
  CSV에서 `gender`가 와도 로더가 **`sex`로 자동 정규화**하여 `sex_Male` 생성.
- **Communities**: preset `paper18`(18개 이진). 연속형은 `--sens_thresh`(기본 0.5 분위)로 이진화.
- **Dutch**: 기본 `sex, age` (필요 시 확장).
- **CelebA**: 이미지 태그/메타로 이진 키 자동 구성.

### Subgroup 통계 CSV (모든 데이터셋 공통: **전체 합산**)
- `train/val/test`를 **합쳐 한 번만** 집계(**split='all'**).
- **k ≤ 16**이면 **2^k 전 조합**을 모두 저장(존재하지 않는 조합은 `count=0`).
- **k > 16**이면 파일 폭주 방지를 위해 **관측된 조합만** 저장.

