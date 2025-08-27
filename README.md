# Subgroup Fairness Research Automation & Debugging System

본 프로젝트는 **연구 실험 자동화**와 **지능형 디버깅**을 위한 통합 Python 시스템입니다.

## 📁 주요 폴더 구조

- `scripts/automation/`  
  - 실험 오케스트레이션, 병렬 실행, 리소스 모니터링, 대시보드, 결과 검증, 재현성 관리  
- `scripts/debugging/`  
  - 자동 오류 탐지, 실행 추적, 데이터 품질 검사, 자가치유, 웹 기반 탐색기  
- `scripts/baselines/`  
  - 베이스라인 알고리즘 구현  
- `scripts/utils/`  
  - 데이터 로딩, 메트릭 계산 등 유틸리티  
- `../data/`  
  - 원본 및 전처리 데이터  
- `../results/`  
  - 실험 결과, 로그, 플롯  
- `../tests/`  
  - 단위/통합 테스트 코드

## 🚀 빠른 시작

### 실험 자동화

```python
from scripts.automation.experiment_orchestrator import run_quick_orchestrated_test
results = run_quick_orchestrated_test()
```
- 명령줄 실행:  
  `python -m scripts.automation.experiment_orchestrator --quick-test`

### 디버깅 시스템

```python
from scripts.debugging import debug_experiment
with debug_experiment("my_experiment") as debug:
    debug.log_metric("accuracy", 0.85)
```

## 💡 주요 기능

- **실험 자동화**: 병렬 실행, 실시간 대시보드, 재현성 관리, 결과 검증
- **지능형 디버깅**: 자동 오류 탐지, 데이터 품질 검사, 자가치유, 실행 추적, 웹 탐색기
- **리소스 모니터링**: CPU/메모리/GPU 실시간 추적 및 알림
- **결과 분석**: 성공률, 성능 메트릭, 리소스 효율성 자동 요약

## 📊 대시보드

- 웹 대시보드: `http://localhost:5555`  
  실험 진행 상황, 성능 메트릭, 리소스 사용량, 에러 로그 실시간 확인

## 🔧 트러블슈팅

- 대시보드/모듈 오류: 의존 패키지 설치(`pip install flask plotly GPUtil`)
- 메모리 부족: 병렬 실험 수/메모리 제한 조정
- 재현성 문제: `reproducibility_manager`로 환경 캡처

## 📚 참고

- 실험 타입/파라미터: [`main/experimental_config.py`](main/experimental_config.py)
- 베이스라인: [`scripts/baselines/`](scripts/baselines/)
- 데이터 로더/메트릭: [`scripts/utils/data_loader.py`](scripts/utils/data_loader.py), [`scripts/utils/metrics.py`](scripts/utils/metrics.py)
- 상세 예제/고급 사용법:  
  [`main/scripts/automation/README.md`](scripts/automation/README.md),  
  [`main/scripts/debugging/README.md`](scripts/debugging/README.md)

---

**문의/기여**: 각 컴포넌트는 독립적으로 테스트 가능하며, 타입 힌트와 문서화가
