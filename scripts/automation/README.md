# Automated Experiment Execution System

**완전 자동화된 실험 실행 시스템**으로 다음 기능을 제공합니다:

## 🚀 주요 기능

### ✅ 구현된 핵심 기능들

1. **강력한 실험 실행기 (Robust Experiment Runner)**
   - 입력 검증 및 자동 오류 처리
   - 병렬 실행 및 리소스 관리
   - 중단 시 우아한 복구

2. **실시간 모니터링 대시보드**
   - 웹 기반 실시간 진행 상황 모니터링
   - 성능 메트릭 및 리소스 사용량 추적
   - 오류 로그 및 완료 시간 예측

3. **재현성 관리 시스템**
   - 랜덤 시드 관리 및 환경 캡처
   - Git 버전 제어 통합
   - 매개변수 로깅 및 환경 스냅샷

4. **자동 결과 검증**
   - 예상 결과 대비 자동 검증
   - 통계적 유의성 검사
   - 품질 점수 계산

5. **리소스 모니터링**
   - CPU, 메모리, 디스크 실시간 모니터링
   - GPU 사용량 추적 (가능한 경우)
   - 임계값 초과 시 자동 알림

## 📁 파일 구조

```
scripts/automation/
├── experiment_orchestrator.py      # 🎯 마스터 오케스트레이션 시스템
├── experiment_executor.py          # ⚡ 병렬 실험 실행기
├── monitoring_dashboard.py         # 📊 실시간 모니터링 대시보드
├── resource_monitor.py             # 💻 시스템 리소스 모니터링
├── experiment_validator.py         # ✅ 입력/결과 검증 시스템
├── reproducibility_manager.py      # 🔄 재현성 관리
└── README.md                       # 📖 이 파일
```

## 🏃‍♂️ 빠른 시작

### 1. 간단한 테스트 실행

```python
from scripts.automation.experiment_orchestrator import run_quick_orchestrated_test

# 빠른 테스트 실행 (2-3개 실험)
results = run_quick_orchestrated_test()
print(f"성공률: {results['execution_results']['job_statistics']['success_rate']:.1%}")
```

### 2. 명령줄에서 실행

```bash
# 빠른 테스트
python -m scripts.automation.experiment_orchestrator --quick-test

# 전체 실험 스위트 실행
python -m scripts.automation.experiment_orchestrator --max-parallel 4 --dashboard-port 5555
```

### 3. 대시보드만 실행

```bash
# 독립 실행 대시보드 (테스트용)
python -m scripts.automation.monitoring_dashboard --standalone --port 5555
```

## 💡 사용 예제

### 기본 사용법

```python
from scripts.automation.experiment_orchestrator import ExperimentOrchestrator, OrchestrationConfig
from experimental_config import ExperimentType, ExperimentParams

# 설정 생성
config = OrchestrationConfig(
    max_parallel_experiments=4,
    enable_dashboard=True,
    dashboard_port=5555,
    output_base_dir="my_experiments"
)

# 오케스트레이터 생성
orchestrator = ExperimentOrchestrator(config)

# 실험 스위트 생성
experiment_types = [
    ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
    ExperimentType.COMPUTATIONAL_EFFICIENCY
]

parameter_grid = {
    'lambda_values': [[0.0, 0.5, 1.0], [0.0, 1.0, 2.0]],
    'random_seeds': [[42, 123, 456]]
}

jobs = orchestrator.create_experiment_suite(
    experiment_types=experiment_types,
    parameter_grid=parameter_grid
)

# 실험 실행
results = orchestrator.run_experiment_suite(jobs)
```

### 고급 사용법: 개별 컴포넌트 사용

```python
from scripts.automation.experiment_executor import ExperimentExecutor, ExperimentJob
from scripts.automation.monitoring_dashboard import create_experiment_dashboard
from scripts.automation.resource_monitor import ResourceMonitor

# 리소스 모니터링 시작
resource_monitor = ResourceMonitor()
resource_monitor.start_monitoring()

# 실행기 생성
executor = ExperimentExecutor(
    output_dir="custom_experiments",
    max_workers=6,
    enable_monitoring=True
)

# 대시보드 생성
dashboard = create_experiment_dashboard(
    executor=executor,
    resource_monitor=resource_monitor,
    port=8080
)

# 실험 작업 생성
job = ExperimentJob(
    job_id="custom_experiment_1",
    experiment_type=ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
    params=ExperimentParams(lambda_values=[0.0, 1.0]),
    timeout_minutes=60
)

# 실험 추가 및 실행
executor.add_experiment(job)
results = executor.execute_all()

# 대시보드 실행 (별도 스레드에서)
dashboard.run(open_browser=True)
```

## 📊 대시보드 기능

웹 대시보드 (`http://localhost:5555`)에서 제공하는 기능:

- **실시간 실험 진행 상황**: 현재 실행 중인 실험들의 상태
- **성능 메트릭**: 정확도, 공정성, 훈련 시간 등의 실시간 차트
- **리소스 사용량**: CPU, 메모리, GPU 사용률 모니터링
- **완료 시간 예측**: 남은 실험들의 예상 완료 시간
- **에러 로그**: 실패한 실험들의 상세 오류 정보

## ⚙️ 설정 옵션

### OrchestrationConfig 주요 설정

```python
config = OrchestrationConfig(
    # 실행 설정
    max_parallel_experiments=4,        # 동시 실행 실험 수
    enable_resource_monitoring=True,   # 리소스 모니터링 활성화
    enable_validation=True,            # 입력/결과 검증 활성화
    enable_dashboard=True,             # 대시보드 활성화
    
    # 리소스 제한
    memory_limit_gb=16.0,              # 메모리 한계 (GB)
    cpu_limit_percent=80.0,            # CPU 사용률 한계 (%)
    disk_limit_gb=20.0,                # 디스크 사용 한계 (GB)
    
    # 모니터링 설정
    monitoring_interval=1.0,           # 모니터링 간격 (초)
    dashboard_port=5555,               # 대시보드 포트
    
    # 출력 설정
    output_base_dir="experiments",     # 결과 저장 디렉토리
    log_level="INFO",                  # 로그 레벨
    
    # 재현성 설정
    enforce_reproducibility=True,      # 재현성 강제
    save_reproducibility_artifacts=True,  # 재현성 아티팩트 저장
    
    # 안전 설정
    max_experiment_duration_hours=24.0,    # 최대 실행 시간
    auto_stop_on_high_failure_rate=True,   # 높은 실패율 시 자동 중단
    failure_rate_threshold=0.8             # 실패율 임계값
)
```

## 🔧 트러블슈팅

### 공통 문제 해결

1. **대시보드 접속 불가**
   ```bash
   # Flask 설치 확인
   pip install flask plotly
   
   # 포트 충돌 확인
   python -m scripts.automation.monitoring_dashboard --port 8080
   ```

2. **메모리 부족**
   ```python
   # 설정에서 병렬 실험 수 줄이기
   config.max_parallel_experiments = 2
   config.memory_limit_gb = 8.0
   ```

3. **GPU 모니터링 오류**
   ```bash
   # GPUtil 설치
   pip install GPUtil
   ```

4. **재현성 문제**
   ```python
   # 수동으로 재현성 설정
   from scripts.automation.reproducibility_manager import setup_reproducible_experiment
   
   manager, repro_info = setup_reproducible_experiment(seed=42)
   print(f"Environment hash: {repro_info['environment_hash']}")
   ```

## 📈 결과 분석

실험 완료 후 다음 파일들이 생성됩니다:

```
output_directory/
├── orchestration_results_YYYYMMDD_HHMMSS.json  # 전체 결과 요약
├── resource_usage_detailed.json                # 리소스 사용 상세 데이터
├── experiments/                                 # 개별 실험 결과들
│   ├── job_experiment1/
│   │   ├── results.json
│   │   ├── job_config.json
│   │   └── reproducibility/
└── repro_archive_YYYYMMDD_HHMMSS/              # 재현성 아카이브
    ├── environment/
    ├── experiment_results.json
    ├── reproduce.py
    └── README.md
```

### 결과 분석 예제

```python
import json

# 메인 결과 로드
with open('orchestration_results_20250827_123456.json', 'r') as f:
    results = json.load(f)

# 성공률 확인
success_rate = results['execution_results']['job_statistics']['success_rate']
print(f"전체 성공률: {success_rate:.1%}")

# 성능 요약
perf_summary = results['execution_results']['performance_summary']
if 'accuracy' in perf_summary:
    avg_accuracy = perf_summary['accuracy']['mean']
    print(f"평균 정확도: {avg_accuracy:.3f}")

# 리소스 효율성
resource_summary = results['analysis_results']['resource_efficiency']
efficiency_score = resource_summary['efficiency_score']
print(f"리소스 효율성: {efficiency_score:.1%}")
```

## 🎯 고급 기능

### 1. 커스텀 검증 규칙

```python
from scripts.automation.experiment_validator import ExperimentValidator

validator = ExperimentValidator()

# 커스텀 검증 함수 추가
def custom_validation(job_result):
    # 사용자 정의 검증 로직
    return {'valid': True, 'warnings': []}

# 검증 실행
validation_result = validator.validate_experiment_results(result)
```

### 2. 재현성 검증

```python
from scripts.automation.reproducibility_manager import ReproducibilityManager

manager = ReproducibilityManager()

# 환경 검증
verification = manager.verify_reproducibility(
    baseline_env_file="environment_snapshot.json"
)

if verification['reproducible']:
    print("✅ 재현 가능한 환경입니다")
else:
    print("⚠️ 환경 차이점 발견:")
    for mismatch in verification['mismatches']:
        print(f"  - {mismatch}")
```

### 3. 리소스 알림 설정

```python
from scripts.automation.resource_monitor import ResourceMonitor

# 커스텀 임계값으로 모니터 생성
monitor = ResourceMonitor(
    alert_thresholds={
        'cpu_percent': 85.0,
        'memory_percent': 90.0,
        'disk_usage_percent': 95.0
    }
)

monitor.start_monitoring()

# 알림 확인
alerts = monitor.get_resource_alerts()
for alert in alerts:
    print(f"🚨 {alert['message']}")
```

## 🎪 예제 시나리오들

### 시나리오 1: 연구 논문용 전체 실험

```python
# 논문용 완전한 실험 스위트
def run_paper_experiments():
    config = OrchestrationConfig(
        max_parallel_experiments=8,
        output_base_dir="paper_experiments_2025",
        max_experiment_duration_hours=48.0,
        enable_dashboard=True
    )
    
    orchestrator = ExperimentOrchestrator(config)
    
    # 모든 실험 타입 포함
    all_experiment_types = list(ExperimentType)
    
    # 광범위한 매개변수 그리드
    extensive_grid = {
        'lambda_values': [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0]
        ],
        'random_seeds': [[42, 123, 456, 789, 999]],
        'n_low_values': [[10, 50, 100, 200]]
    }
    
    jobs = orchestrator.create_experiment_suite(
        experiment_types=all_experiment_types,
        parameter_grid=extensive_grid
    )
    
    return orchestrator.run_experiment_suite(jobs)
```

### 시나리오 2: 디버깅용 단일 실험

```python
# 디버깅을 위한 단순화된 실험
def debug_single_experiment():
    from scripts.automation.experiment_executor import ExperimentExecutor, ExperimentJob
    
    executor = ExperimentExecutor(
        output_dir="debug_experiment",
        max_workers=1,
        log_level="DEBUG"
    )
    
    job = ExperimentJob(
        job_id="debug_test",
        experiment_type=ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
        params=ExperimentParams(
            lambda_values=[1.0],
            random_seeds=[42],
            max_iterations=100  # 빠른 디버깅용
        ),
        timeout_minutes=10
    )
    
    executor.add_experiment(job)
    return executor.execute_all()
```

### 시나리오 3: 성능 벤치마킹

```python
# 다양한 하드웨어에서 성능 측정
def benchmark_performance():
    from scripts.automation.resource_monitor import monitor_experiment
    
    print("5분간 시스템 성능 모니터링...")
    performance_summary = monitor_experiment(
        duration_minutes=5,
        output_file="performance_baseline.json"
    )
    
    print(f"평균 CPU: {performance_summary['cpu_usage']['mean']:.1f}%")
    print(f"최대 메모리: {performance_summary['memory_usage']['max_gb']:.1f} GB")
    
    return performance_summary
```

## 📚 추가 자료

- **실험 설정**: `experimental_config.py`에서 실험 타입과 매개변수 확인
- **베이스라인 메소드**: `scripts/baselines/` 디렉토리의 구현된 알고리즘들
- **데이터 로더**: `scripts/utils/data_loader.py`에서 데이터 로딩 옵션 확인
- **메트릭 계산**: `scripts/utils/metrics.py`에서 평가 메트릭 정의 확인

## 🤝 기여하기

새로운 기능 추가나 버그 수정을 위한 개발 가이드라인:

1. 각 컴포넌트는 독립적으로 테스트 가능해야 함
2. 로깅을 통한 상세한 상태 추적
3. 예외 처리 및 우아한 실패 복구
4. 타입 힌트 및 문서화 문자열 포함

---

**🎉 자동화된 실험 실행 시스템이 성공적으로 구축되었습니다!**

모든 주요 기능이 구현되어 있으며, 웹 대시보드를 통해 실시간으로 실험 진행 상황을 모니터링할 수 있습니다.