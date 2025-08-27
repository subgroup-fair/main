"""
Automated Experiment Execution System

A comprehensive system for running, monitoring, and managing machine learning experiments
with built-in validation, reproducibility, and real-time monitoring capabilities.

Main Components:
- ExperimentOrchestrator: Master control system
- ExperimentExecutor: Parallel experiment execution
- MonitoringDashboard: Real-time web dashboard  
- ResourceMonitor: System resource monitoring
- ExperimentValidator: Input/output validation
- ReproducibilityManager: Reproducibility assurance

Quick Start:
    from scripts.automation import run_quick_orchestrated_test
    
    # Run quick test
    results = run_quick_orchestrated_test()
    
    # Check results
    success_rate = results['execution_results']['job_statistics']['success_rate']
    print(f"Success rate: {success_rate:.1%}")

For detailed usage, see README.md
"""

from .experiment_orchestrator import (
    ExperimentOrchestrator,
    OrchestrationConfig,
    run_quick_orchestrated_test,
    create_standard_experiment_suite
)

from .experiment_executor import (
    ExperimentExecutor,
    ExperimentJob,
    JobResult
)

from .monitoring_dashboard import (
    ExperimentDashboard,
    create_experiment_dashboard,
    run_standalone_dashboard
)

from .resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    monitor_experiment
)

from .experiment_validator import (
    ExperimentValidator,
    validate_single_experiment,
    run_validation_suite
)

from .reproducibility_manager import (
    ReproducibilityManager,
    setup_reproducible_experiment,
    create_experiment_archive
)

__version__ = "1.0.0"
__author__ = "Experiment Automation System"

# Convenience functions for common use cases
__all__ = [
    # Main orchestration
    "ExperimentOrchestrator",
    "OrchestrationConfig", 
    "run_quick_orchestrated_test",
    "create_standard_experiment_suite",
    
    # Experiment execution
    "ExperimentExecutor",
    "ExperimentJob",
    "JobResult",
    
    # Monitoring and dashboard
    "ExperimentDashboard", 
    "create_experiment_dashboard",
    "run_standalone_dashboard",
    "ResourceMonitor",
    "ResourceSnapshot",
    "monitor_experiment",
    
    # Validation and reproducibility
    "ExperimentValidator",
    "validate_single_experiment", 
    "run_validation_suite",
    "ReproducibilityManager",
    "setup_reproducible_experiment",
    "create_experiment_archive"
]