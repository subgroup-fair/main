#!/usr/bin/env python3
"""
Examples for the Automated Experiment Execution System

This file contains practical examples showing how to use different components
of the experiment automation system for various research scenarios.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.automation import *
from experimental_config import ExperimentType, ExperimentParams

def example_1_quick_test():
    """
    예제 1: 빠른 테스트 실행
    가장 간단한 시작 방법
    """
    print("=== 예제 1: 빠른 테스트 실행 ===")
    
    # 단 한 줄로 테스트 실행
    results = run_quick_orchestrated_test("example1_output")
    
    # 결과 확인
    stats = results['execution_results']['job_statistics']
    print(f"✅ 완료된 실험: {stats['completed_jobs']}")
    print(f"❌ 실패한 실험: {stats['failed_jobs']}")
    print(f"📊 성공률: {stats['success_rate']:.1%}")
    
    return results

def example_2_basic_orchestration():
    """
    예제 2: 기본 오케스트레이션 사용
    설정을 커스터마이징하여 실험 실행
    """
    print("\n=== 예제 2: 기본 오케스트레이션 ===")
    
    # 커스텀 설정 생성
    config = OrchestrationConfig(
        max_parallel_experiments=2,
        output_base_dir="example2_output",
        enable_dashboard=True,
        dashboard_port=5556,  # 다른 포트 사용
        log_level="INFO"
    )
    
    # 오케스트레이터 생성
    orchestrator = ExperimentOrchestrator(config)
    
    # 간단한 실험 스위트 생성
    experiment_types = [ExperimentType.ACCURACY_FAIRNESS_TRADEOFF]
    
    parameter_grid = {
        'lambda_values': [[0.0, 1.0]],  # 2개 람다 값만
        'random_seeds': [[42, 123]]     # 2개 시드만
    }
    
    jobs = orchestrator.create_experiment_suite(
        experiment_types=experiment_types,
        parameter_grid=parameter_grid
    )
    
    print(f"🚀 {len(jobs)}개 실험 작업 생성됨")
    print("💻 대시보드: http://localhost:5556")
    
    # 실험 실행
    results = orchestrator.run_experiment_suite(jobs)
    
    # 결과 요약
    duration = results['orchestration_info']['total_duration']
    print(f"⏱️ 총 실행 시간: {duration:.1f}초")
    
    return results

def example_3_individual_components():
    """
    예제 3: 개별 컴포넌트 사용
    각 컴포넌트를 개별적으로 사용하는 방법
    """
    print("\n=== 예제 3: 개별 컴포넌트 사용 ===")
    
    # 1. 리소스 모니터링만 사용
    print("📊 리소스 모니터링 시작...")
    resource_monitor = ResourceMonitor(monitoring_interval=2.0)
    resource_monitor.start_monitoring()
    
    # 2. 실험 실행기만 사용
    print("⚡ 실험 실행기 설정...")
    executor = ExperimentExecutor(
        output_dir="example3_output",
        max_workers=1,
        enable_monitoring=False,  # 별도로 모니터링 중
        log_level="DEBUG"
    )
    
    # 3. 단일 실험 작업 생성
    job = ExperimentJob(
        job_id="individual_test",
        experiment_type=ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
        params=ExperimentParams(
            lambda_values=[0.5],
            random_seeds=[42],
            max_iterations=50  # 빠른 실행을 위해
        ),
        timeout_minutes=5
    )
    
    print("🔧 실험 실행 중...")
    executor.add_experiment(job)
    results = executor.execute_all()
    
    # 4. 리소스 사용량 확인
    resource_summary = resource_monitor.get_summary()
    print(f"💾 평균 CPU 사용률: {resource_summary['cpu_usage']['mean']:.1f}%")
    print(f"🧠 최대 메모리 사용량: {resource_summary['memory_usage']['max_gb']:.1f} GB")
    
    resource_monitor.stop_monitoring()
    
    return results

def example_4_reproducibility():
    """
    예제 4: 재현성 관리 사용
    실험의 재현 가능성을 보장하는 방법
    """
    print("\n=== 예제 4: 재현성 관리 ===")
    
    # 1. 재현 가능한 환경 설정
    manager, repro_info = setup_reproducible_experiment(seed=42)
    
    print(f"🌱 시드 설정: {repro_info['seed']}")
    print(f"🔒 환경 해시: {repro_info['environment_hash']}")
    print(f"🐍 Python 버전: {repro_info['environment']['platform']['python_version']}")
    
    if repro_info['git_state']:
        print(f"📝 Git 커밋: {repro_info['git_state']['commit_hash'][:8]}...")
        print(f"🌿 Git 브랜치: {repro_info['git_state']['branch']}")
    
    # 2. 재현성 아카이브 생성을 위한 더미 결과
    dummy_results = {
        "experiment_type": "test",
        "accuracy": 0.85,
        "fairness": 0.12,
        "training_time": 45.2
    }
    
    archive_path = create_experiment_archive(
        results=dummy_results,
        output_dir="example4_output"
    )
    
    print(f"📦 재현성 아카이브 생성: {archive_path}")
    
    return repro_info

def example_5_validation():
    """
    예제 5: 실험 검증 사용
    입력과 출력을 자동으로 검증하는 방법
    """
    print("\n=== 예제 5: 실험 검증 ===")
    
    # 1. 검증기 생성
    validator = ExperimentValidator()
    
    # 2. 테스트용 실험 작업 생성
    test_job = ExperimentJob(
        job_id="validation_test",
        experiment_type=ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
        params=ExperimentParams(
            lambda_values=[0.0, 0.5, 1.0],
            random_seeds=[42, 123, 456],
            max_iterations=1000
        ),
        timeout_minutes=30
    )
    
    # 3. 입력 검증
    print("✅ 실험 작업 검증 중...")
    validation_result = validator.validate_experiment_job(test_job)
    
    if validation_result['valid']:
        print("✅ 실험 작업 검증 통과")
        print(f"📋 수행된 검사: {validation_result['checks_performed']}")
        if validation_result['warnings']:
            print(f"⚠️ 경고사항: {validation_result['warnings']}")
    else:
        print("❌ 실험 작업 검증 실패")
        print(f"🚫 오류: {validation_result['errors']}")
    
    # 4. 더미 결과로 출력 검증 시뮬레이션
    from dataclasses import asdict
    dummy_result = JobResult(
        job_id="validation_test",
        status="completed",
        start_time=repro_info.get('start_time', '2025-01-01T00:00:00') if 'repro_info' in locals() else '2025-01-01T00:00:00',
        end_time='2025-01-01T00:05:00',
        results={
            'datasets': {
                'test_dataset': {
                    'methods': {
                        'test_method': {
                            'lambda_sweep': [
                                {
                                    'lambda': 0.0,
                                    'accuracy_mean': 0.85,
                                    'sup_ipm_mean': 0.15,
                                    'training_time_mean': 45.2
                                }
                            ]
                        }
                    }
                }
            }
        }
    )
    
    result_validation = validator.validate_experiment_results(dummy_result)
    
    print(f"📊 결과 검증: {'✅ 통과' if result_validation['valid'] else '❌ 실패'}")
    
    if 'quality_scores' in result_validation:
        for metric, score in result_validation['quality_scores'].items():
            print(f"🎯 {metric} 품질 점수: {score:.2f}")
    
    return validation_result

def example_6_dashboard_only():
    """
    예제 6: 대시보드만 실행
    실험 없이 대시보드만 테스트하는 방법
    """
    print("\n=== 예제 6: 독립 실행 대시보드 ===")
    
    print("🌐 독립 실행 대시보드를 시작합니다...")
    print("📱 브라우저에서 http://localhost:5557 접속")
    print("🛑 Ctrl+C로 중단하세요")
    
    try:
        # 독립 실행 대시보드 (포트 5557 사용)
        run_standalone_dashboard(port=5557)
    except KeyboardInterrupt:
        print("\n🛑 대시보드가 중단되었습니다.")

def example_7_comprehensive_research():
    """
    예제 7: 종합 연구용 실험
    실제 연구에서 사용할 수 있는 완전한 실험 설정
    """
    print("\n=== 예제 7: 종합 연구용 실험 (데모용) ===")
    
    # 연구용 설정 - 실제로는 더 많은 실험을 실행
    config = OrchestrationConfig(
        max_parallel_experiments=4,
        output_base_dir="comprehensive_research",
        enable_dashboard=True,
        dashboard_port=5558,
        max_experiment_duration_hours=2.0,  # 데모용으로 짧게
        enable_resource_monitoring=True,
        enable_validation=True,
        save_reproducibility_artifacts=True
    )
    
    orchestrator = ExperimentOrchestrator(config)
    
    # 여러 실험 타입 포함
    experiment_types = [
        ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
        ExperimentType.COMPUTATIONAL_EFFICIENCY,
        ExperimentType.PARTIAL_VS_COMPLETE
    ]
    
    # 체계적인 매개변수 탐색
    comprehensive_grid = {
        'lambda_values': [
            [0.0, 0.5, 1.0],      # 기본 설정
            [0.0, 1.0, 2.0],      # 강한 정규화
        ],
        'random_seeds': [
            [42, 123, 456]        # 통계적 신뢰성을 위한 다중 시드
        ]
    }
    
    jobs = orchestrator.create_experiment_suite(
        experiment_types=experiment_types,
        parameter_grid=comprehensive_grid
    )
    
    print(f"🔬 {len(jobs)}개의 연구용 실험 생성")
    print("💻 실시간 대시보드: http://localhost:5558")
    print("📊 모든 메트릭과 리소스가 모니터링됩니다")
    print("📦 재현성 아카이브가 자동으로 생성됩니다")
    
    # 실험 실행 (데모에서는 처음 3개만)
    demo_jobs = jobs[:3]
    print(f"📝 데모를 위해 {len(demo_jobs)}개 실험만 실행")
    
    results = orchestrator.run_experiment_suite(demo_jobs)
    
    # 결과 분석
    orchestration_info = results['orchestration_info']
    execution_results = results['execution_results']
    
    print(f"⏱️ 총 실행 시간: {orchestration_info['total_duration']:.1f}초")
    print(f"📈 성공률: {execution_results['job_statistics']['success_rate']:.1%}")
    
    if 'reproducibility_archive' in results:
        print(f"📦 재현성 아카이브: {results['reproducibility_archive']}")
    
    return results

def run_all_examples():
    """모든 예제를 순차적으로 실행"""
    
    print("🚀 자동화된 실험 실행 시스템 예제 모음\n")
    
    try:
        # 예제 1: 빠른 테스트
        example_1_quick_test()
        input("\n⏸️ 다음 예제로 넘어가려면 Enter를 누르세요...")
        
        # 예제 2: 기본 오케스트레이션  
        example_2_basic_orchestration()
        input("\n⏸️ 다음 예제로 넘어가려면 Enter를 누르세요...")
        
        # 예제 3: 개별 컴포넌트
        example_3_individual_components()
        input("\n⏸️ 다음 예제로 넘어가려면 Enter를 누르세요...")
        
        # 예제 4: 재현성
        example_4_reproducibility()
        input("\n⏸️ 다음 예제로 넘어가려면 Enter를 누르세요...")
        
        # 예제 5: 검증
        example_5_validation()
        input("\n⏸️ 다음 예제로 넘어가려면 Enter를 누르세요...")
        
        print("\n✅ 모든 예제가 완료되었습니다!")
        print("📝 예제 6 (대시보드)와 예제 7 (종합 연구)는 개별적으로 실행해보세요.")
        
    except KeyboardInterrupt:
        print("\n🛑 사용자가 예제 실행을 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 예제 실행 중 오류 발생: {e}")

def main():
    """메인 함수 - 명령줄 인터페이스"""
    import argparse
    
    parser = argparse.ArgumentParser(description="자동화된 실험 실행 시스템 예제")
    parser.add_argument("--example", type=int, choices=range(1, 8), 
                       help="실행할 예제 번호 (1-7)")
    parser.add_argument("--all", action="store_true",
                       help="모든 예제를 순차적으로 실행")
    
    args = parser.parse_args()
    
    if args.all:
        run_all_examples()
    elif args.example:
        example_functions = {
            1: example_1_quick_test,
            2: example_2_basic_orchestration,
            3: example_3_individual_components, 
            4: example_4_reproducibility,
            5: example_5_validation,
            6: example_6_dashboard_only,
            7: example_7_comprehensive_research
        }
        
        print(f"🎯 예제 {args.example} 실행")
        example_functions[args.example]()
        print(f"✅ 예제 {args.example} 완료")
    else:
        print("사용법:")
        print("  python examples.py --example 1    # 특정 예제 실행")
        print("  python examples.py --all          # 모든 예제 실행") 
        print("\n예제 목록:")
        print("  1: 빠른 테스트 실행")
        print("  2: 기본 오케스트레이션")
        print("  3: 개별 컴포넌트 사용")
        print("  4: 재현성 관리")
        print("  5: 실험 검증")
        print("  6: 대시보드만 실행")
        print("  7: 종합 연구용 실험")

if __name__ == "__main__":
    main()