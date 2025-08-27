# 🧠 Intelligent Debugging System for Research Experiments

**완전 자동화된 지능형 디버깅 시스템**으로 연구 실험의 모든 오류를 탐지하고 자동으로 해결합니다.

## 🚀 핵심 기능

### ✅ 구현된 주요 시스템들

1. **🔍 자동 오류 탐지 및 진단**
   - 통계적 이상 탐지 (Statistical Anomaly Detection)
   - 성능 병목 지점 식별 (Performance Bottleneck Identification)
   - 데이터 품질 이슈 자동 감지
   - 코드 로직 오류 포착 및 분석

2. **🛠️ 지능형 디버깅 도구**
   - 📊 **인터랙티브 데이터 탐색기**: 웹 기반 실시간 데이터 분석
   - 🔬 **단계별 실행 추적기**: 변수 상태 및 함수 호출 추적  
   - 🔧 **변수 상태 검사기**: 실시간 변수 모니터링
   - 📋 **자동 오류 리포트 생성**: 상세한 진단 보고서

3. **🩹 자가치유 능력**
   - 🔄 **자동 재시도 메커니즘**: 일시적 실패에 대한 지능적 재시도
   - 🧹 **자동 데이터 정리**: 결측값, 이상값, 중복값 자동 처리
   - ⚙️ **매개변수 조정 제안**: 성능 향상을 위한 최적 매개변수 추천
   - 🔀 **대체 방법 추천**: 더 나은 알고리즘 및 접근법 제안

4. **🧠 지능형 로깅**
   - 📝 **컨텍스트 인식 오류 메시지**: 상황에 맞는 상세한 오류 정보
   - 💡 **디버깅 힌트 및 제안**: 문제 해결을 위한 구체적 가이드
   - 📈 **성능 프로파일링**: CPU, 메모리, GPU 사용량 추적
   - 🔍 **스택 트레이스 분석**: 오류 발생 경로 자동 분석

## 📁 시스템 구조

```
scripts/debugging/
├── intelligent_debugger.py        # 🧠 핵심 오류 탐지 및 진단
├── interactive_explorer.py        # 📊 웹 기반 데이터 탐색기
├── execution_tracer.py            # 🔬 단계별 실행 추적
├── self_healing_system.py         # 🩹 자가치유 시스템
├── comprehensive_debug_system.py  # 🎯 통합 디버깅 시스템
├── examples/                      # 📝 사용 예제들
├── README.md                      # 📖 이 파일
└── __init__.py                    # 📦 패키지 초기화
```

## 🏃‍♂️ 빠른 시작

### 1. 가장 간단한 사용법

```python
from scripts.debugging import debug_experiment

# 실험을 디버깅 세션으로 감싸기
with debug_experiment("my_research_experiment") as debug:
    # 데이터 추가
    debug.add_data("training_data", your_dataframe)
    debug.add_data("test_data", your_test_data)
    
    # 컨텍스트 설정
    debug.set_context(learning_rate=0.01, batch_size=32)
    
    # 메트릭 로깅 (이상 탐지 자동 수행)
    debug.log_metric("accuracy", 0.85)
    debug.log_metric("loss", 0.23)
    
    # 여기서 실험 코드 실행
    # 시스템이 자동으로 모든 것을 모니터링합니다!
```

### 2. 자동 치유가 포함된 함수

```python
from scripts.debugging import auto_heal, SelfHealingSystem

# 자동 치유 시스템 생성
healer = SelfHealingSystem()

@auto_heal(healing_system=healer, max_attempts=3)
def potentially_problematic_function(data):
    # 메모리 오류, 네트워크 오류 등이 발생할 수 있는 함수
    # 시스템이 자동으로 문제를 감지하고 해결 시도
    return process_data(data)

result = potentially_problematic_function(my_data)
```

### 3. 실행 추적 및 성능 분석

```python
from scripts.debugging import trace_execution, ExecutionTracer

# 함수 실행 추적
@trace_execution(max_steps=1000)
def complex_algorithm(parameters):
    # 복잡한 알고리즘 실행
    # 모든 단계가 자동으로 추적됩니다
    return results

# 또는 컨텍스트 매니저 사용
with TraceExecution("algorithm_analysis") as trace:
    result = complex_algorithm(params)
    
# 실행 요약 확인
summary = trace.get_summary()
```

## 💻 웹 기반 인터랙티브 탐색기

```python
from scripts.debugging import create_comprehensive_debug_system

# 완전한 디버깅 시스템 생성
debug_system = create_comprehensive_debug_system()

# 웹 탐색기 실행 (http://localhost:5560)
debug_system.launch_explorer()

# 데이터 추가
with debug_system.create_debug_session("analysis") as session:
    session.add_data("dataset", your_data)
    session.log_metric("performance", 0.92)
```

웹 인터페이스에서 확인할 수 있는 기능들:
- 📊 **실시간 데이터 시각화**: 히스토그램, 산점도, 시계열 차트
- 🔍 **변수 상태 검사**: 모든 변수의 실시간 값과 타입 확인
- 🐛 **오류 및 이상 현황**: 실시간 문제 탐지 현황
- 📈 **성능 메트릭**: CPU, 메모리, GPU 사용량 모니터링
- 💡 **제안 사항**: 자동 생성되는 개선 제안들

## 🛠️ 고급 사용법

### 자세한 오류 분석

```python
from scripts.debugging import IntelligentDebugger

debugger = IntelligentDebugger(
    enable_anomaly_detection=True,
    enable_self_healing=True,
    anomaly_sensitivity=0.05
)

# 메트릭에서 이상 탐지
metrics = {'accuracy': 0.45, 'loss': 2.3}  # 비정상적인 값들
anomalies = debugger.detect_statistical_anomalies(metrics)

for anomaly in anomalies:
    print(f"⚠️ 이상 탐지: {anomaly.description}")
    print(f"신뢰도: {anomaly.confidence:.2%}")
```

### 데이터 품질 자동 검사

```python
import pandas as pd
import numpy as np

# 문제가 있는 데이터 생성
problematic_data = pd.DataFrame({
    'feature1': [1, 2, np.nan, 4, 1000],    # 결측값과 이상값
    'feature2': [1, 1, 1, 1, 1],            # 분산 없음
    'feature3': ['a', 'b', 'a', 'b', 'a']   # 중복값
})

# 자동 품질 검사
issues = debugger.detect_data_quality_issues(problematic_data, "training_data")

print("🔍 데이터 품질 이슈:")
for issue in issues:
    print(f"  - {issue}")
```

### 성능 병목 지점 분석

```python
import time

def slow_function():
    time.sleep(2)  # 의도적으로 느린 함수
    return "완료"

# 성능 분석
start_time = time.time()
result = slow_function()
execution_time = time.time() - start_time

bottlenecks = debugger.detect_performance_bottlenecks(execution_time, "slow_function")

for bottleneck in bottlenecks:
    print(f"🐌 성능 병목: {bottleneck}")
```

### 자동 매개변수 조정

```python
from scripts.debugging import SelfHealingSystem

healer = SelfHealingSystem()

# 매개변수 컨텍스트 설정
context = {
    'learning_rate': 0.1,
    'batch_size': 128,
    'regularization': 0.001
}

# 문제 상황 시뮬레이션
problem = "Training loss is exploding and not converging"

# 자동 치유 시도
healing_action = healer.attempt_healing(problem, context)

print(f"🩹 치유 시도: {healing_action.action_description}")
print(f"성공 여부: {'✅ 성공' if healing_action.success else '❌ 실패'}")
print(f"신뢰도: {healing_action.confidence:.1%}")
```

## 📊 실제 연구 시나리오 예제

### 시나리오 1: 머신러닝 모델 훈련 디버깅

```python
from scripts.debugging import debug_experiment
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 완전 자동화된 머신러닝 디버깅
with debug_experiment("ml_training_debug", auto_launch_explorer=True) as debug:
    # 데이터 로드 및 추가
    train_data = pd.read_csv("training_data.csv")
    debug.add_data("training_data", train_data)
    
    # 자동 데이터 품질 검사 수행됨
    # 이상값, 결측값, 중복값 등 자동 탐지
    
    # 모델 설정
    debug.set_context(
        model_type="RandomForest",
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # 훈련 진행 with 자동 추적
    @debug.trace_function
    @debug.auto_heal_function
    def train_model(X, y):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        return model
    
    X = train_data.drop('target', axis=1)
    y = train_data['target']
    
    model = train_model(X, y)
    
    # 성능 메트릭 로깅 (이상 탐지 자동 수행)
    for epoch in range(10):
        accuracy = 0.8 + 0.02 * epoch + np.random.normal(0, 0.01)
        precision = 0.75 + 0.015 * epoch + np.random.normal(0, 0.02)
        recall = 0.82 + 0.01 * epoch + np.random.normal(0, 0.015)
        
        debug.log_metric("accuracy", accuracy)
        debug.log_metric("precision", precision)
        debug.log_metric("recall", recall)
    
# 세션 종료 후 자동으로 종합 리포트 생성
# 웹 브라우저에서 실시간 결과 확인 가능
```

### 시나리오 2: 데이터 전처리 파이프라인 디버깅

```python
from scripts.debugging import create_comprehensive_debug_system

# 포괄적 디버깅 시스템 생성
debug_system = create_comprehensive_debug_system()

# 데이터 전처리 파이프라인
def preprocess_data(raw_data):
    # 각 단계마다 자동 품질 검사
    
    # 1단계: 기본 정리
    cleaned_data = raw_data.dropna()  # 결측값 제거
    debug_system.debugger.detect_data_quality_issues(cleaned_data, "after_dropna")
    
    # 2단계: 이상값 처리  
    Q1 = cleaned_data.quantile(0.25)
    Q3 = cleaned_data.quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = cleaned_data[~((cleaned_data < (Q1 - 1.5 * IQR)) | 
                                  (cleaned_data > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # 3단계: 정규화
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(filtered_data),
        columns=filtered_data.columns
    )
    
    return normalized_data

# 자동 치유가 포함된 전처리 실행
@debug_system.healing_system.auto_heal()
def robust_preprocessing(data):
    return preprocess_data(data)

# 실행 및 모니터링
with debug_system.create_debug_session("preprocessing_pipeline") as session:
    raw_data = pd.read_csv("messy_data.csv")
    session.add_data("raw_data", raw_data)
    
    processed_data = robust_preprocessing(raw_data)
    session.add_data("processed_data", processed_data)
    
    # 처리 전후 비교 분석 자동 수행
```

### 시나리오 3: 딥러닝 모델 훈련 중 자동 치유

```python
from scripts.debugging import auto_heal, SelfHealingSystem
import torch
import torch.nn as nn

# 고급 자가치유 시스템 설정
healer = SelfHealingSystem(
    enable_data_cleaning=True,
    enable_parameter_tuning=True,
    enable_method_switching=True,
    learning_enabled=True  # 과거 경험에서 학습
)

@auto_heal(healing_system=healer, max_attempts=5)
def train_deep_model(model, train_loader, optimizer, epochs):
    """자동 치유가 포함된 딥러닝 훈련 함수"""
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 자동 이상 탐지
            if loss.item() > 10.0:  # 손실이 너무 큰 경우
                raise ValueError(f"Loss exploding: {loss.item()}")
            
            if torch.isnan(loss):  # NaN 발생
                raise ValueError("NaN loss detected")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # 메모리 정리 (자동 치유 시스템에서 관리)
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
    
    return model

# 사용 예제
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 훈련 실행 - 자동으로 문제 해결
try:
    trained_model = train_deep_model(model, train_loader, optimizer, epochs=100)
    print("✅ 훈련 성공적으로 완료!")
except Exception as e:
    print(f"❌ 모든 치유 시도 실패: {e}")
```

## 🎛️ 시스템 설정 옵션

### 디버깅 시스템 커스터마이징

```python
from scripts.debugging import ComprehensiveDebugSystem

# 커스텀 설정으로 디버깅 시스템 생성
debug_system = ComprehensiveDebugSystem(
    output_dir="my_debug_logs",           # 로그 출력 디렉토리
    enable_auto_healing=True,             # 자동 치유 활성화
    enable_execution_tracing=True,        # 실행 추적 활성화
    enable_web_explorer=True,             # 웹 탐색기 활성화
    explorer_port=8080,                   # 웹 탐색기 포트
    auto_start=True                       # 자동 시작
)
```

### 자가치유 시스템 세밀 조정

```python
from scripts.debugging import SelfHealingSystem

# 고급 자가치유 설정
healer = SelfHealingSystem(
    max_healing_attempts=5,               # 최대 치유 시도 횟수
    enable_data_cleaning=True,            # 데이터 정리 활성화
    enable_parameter_tuning=True,         # 매개변수 조정 활성화
    enable_method_switching=True,         # 방법 전환 활성화
    learning_enabled=True,                # 학습 기능 활성화
    output_dir="healing_logs"             # 치유 로그 디렉토리
)
```

### 이상 탐지 민감도 조정

```python
from scripts.debugging import IntelligentDebugger

# 이상 탐지 민감도 설정
debugger = IntelligentDebugger(
    enable_anomaly_detection=True,
    anomaly_sensitivity=0.01,             # 더 민감하게 (기본값: 0.05)
    enable_self_healing=True,
    enable_profiling=True
)
```

## 📈 성능 모니터링 및 분석

### 시스템 상태 모니터링

```python
# 실시간 시스템 상태 확인
status = debug_system.get_system_status()

print("🖥️ 시스템 현황:")
print(f"  모니터링 활성화: {status['system_info']['monitoring_active']}")
print(f"  활성 세션: {len(status['system_info']['active_sessions'])}")
print(f"  총 디버그 이벤트: {status['debugger_stats']['total_events']}")

if 'healing_stats' in status:
    healing = status['healing_stats']
    print(f"  치유 시도 횟수: {healing['total_attempts']}")
    print(f"  치유 성공률: {healing['success_rate']:.1%}")
```

### 종합 보고서 생성

```python
# 전체 시스템에 대한 종합 보고서 생성
report = debug_system.generate_comprehensive_report()

print("📊 종합 분석 보고서:")
print(f"  분석 기간: {report['report_info']['generated_at']}")
print(f"  총 디버그 이벤트: {report['debug_events_analysis']['total_events']}")
print(f"  가장 흔한 문제 유형: {report['debug_events_analysis']['most_common_type']}")

print("\n💡 시스템 추천사항:")
for recommendation in report['recommendations']:
    print(f"  - {recommendation}")
```

## 🔧 문제 해결 가이드

### 일반적인 문제들

1. **웹 탐색기 접속 불가**
   ```bash
   # 필요한 패키지 설치
   pip install flask plotly pandas numpy scikit-learn
   
   # 다른 포트로 실행
   debug_system.explorer_port = 8080
   debug_system.launch_explorer()
   ```

2. **메모리 부족 오류**
   ```python
   # 자가치유 시스템이 자동으로 처리하지만, 수동 설정도 가능
   healer = SelfHealingSystem()
   healer.attempt_healing("MemoryError: out of memory", {
       'batch_size': 128  # 자동으로 줄여줌
   })
   ```

3. **이상 탐지 민감도 조정**
   ```python
   # 너무 많은 이상이 탐지되는 경우
   debugger = IntelligentDebugger(anomaly_sensitivity=0.1)  # 덜 민감하게
   
   # 이상을 놓치는 경우
   debugger = IntelligentDebugger(anomaly_sensitivity=0.01)  # 더 민감하게
   ```

### 성능 최적화

```python
# 대용량 데이터를 다룰 때
debug_system = ComprehensiveDebugSystem(
    enable_execution_tracing=False,  # 추적 비활성화로 성능 향상
)

# 또는 추적 범위 제한
tracer = ExecutionTracer(
    max_steps=1000,                   # 최대 추적 단계 제한
    filter_functions=['my_function']   # 특정 함수만 추적
)
```

## 🎯 고급 팁과 요령

### 1. 실험별 맞춤 설정

```python
# 실험 유형에 따른 맞춤 디버깅
def create_ml_debugger():
    """머신러닝 실험용 디버거"""
    return IntelligentDebugger(
        anomaly_sensitivity=0.05,
        enable_self_healing=True
    )

def create_dl_debugger():
    """딥러닝 실험용 디버거"""
    return ComprehensiveDebugSystem(
        enable_execution_tracing=False,  # 성능상 이유로 비활성화
        enable_auto_healing=True
    )
```

### 2. 자동 치유 전략 커스터마이징

```python
# 특정 문제에 대한 맞춤 치유 전략
healer = SelfHealingSystem()

# 사용자 정의 치유 함수 추가 (개념적 예시)
def custom_memory_healing(context):
    """메모리 부족 시 사용자 정의 치유"""
    if 'batch_size' in context:
        context['batch_size'] = max(1, context['batch_size'] // 4)
    return True

# healer.add_custom_strategy('memory_error', custom_memory_healing)
```

### 3. 다중 실험 모니터링

```python
# 여러 실험을 동시에 모니터링
debug_system = create_comprehensive_debug_system()

experiments = ['exp_1', 'exp_2', 'exp_3']

for exp_name in experiments:
    with debug_system.create_debug_session(exp_name) as session:
        # 각 실험별 독립적 모니터링
        session.add_data(f"{exp_name}_data", data)
        session.log_metric("accuracy", accuracy)

# 모든 실험 결과 비교 분석
comparative_report = debug_system.generate_comprehensive_report()
```

## 📚 추가 자료

- **실행 추적 상세 가이드**: `execution_tracer.py` 모듈 문서 참조
- **자가치유 전략 가이드**: `self_healing_system.py` 모듈 문서 참조
- **웹 탐색기 사용법**: `interactive_explorer.py` 모듈 문서 참조
- **API 레퍼런스**: 각 모듈의 독스트링 참조

---

## 🎉 결론

**지능형 디버깅 시스템**이 성공적으로 구축되었습니다! 이 시스템은:

- ✅ **완전 자동화된 오류 탐지** - 통계적 이상부터 성능 병목까지
- ✅ **실시간 웹 기반 탐색기** - 데이터와 변수 상태 실시간 모니터링
- ✅ **자가치유 능력** - 메모리 오류, 네트워크 오류 등 자동 복구
- ✅ **단계별 실행 추적** - 함수 호출부터 변수 변화까지 완전 추적
- ✅ **지능형 제안 시스템** - 매개변수 조정, 방법 전환 등 자동 제안

이제 연구 실험 중 발생하는 모든 문제를 **자동으로 탐지하고 해결**할 수 있습니다!

**🚀 시작해보세요:**

```python
from scripts.debugging import debug_experiment

with debug_experiment("your_research") as debug:
    # 여기서 실험 코드를 실행하면
    # 모든 것이 자동으로 모니터링되고 문제가 해결됩니다!
    pass
```