"""
Master Experiment Orchestration System
Integrates all components: execution, monitoring, validation, and reproducibility
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import argparse
from dataclasses import dataclass, asdict
import subprocess
import signal

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import all automation components
from scripts.automation.experiment_executor import ExperimentExecutor, ExperimentJob, JobResult
from scripts.automation.resource_monitor import ResourceMonitor
from scripts.automation.experiment_validator import ExperimentValidator
from scripts.automation.reproducibility_manager import ReproducibilityManager
from scripts.automation.monitoring_dashboard import ExperimentDashboard
from experimental_config import ExperimentType, ExperimentParams, EXPERIMENT_CONFIGS

@dataclass
class OrchestrationConfig:
    """Configuration for experiment orchestration"""
    
    # Execution settings
    max_parallel_experiments: int = 4
    enable_resource_monitoring: bool = True
    enable_validation: bool = True
    enable_dashboard: bool = True
    
    # Resource limits
    memory_limit_gb: float = 16.0
    cpu_limit_percent: float = 80.0
    disk_limit_gb: float = 20.0
    
    # Monitoring settings
    monitoring_interval: float = 1.0
    dashboard_port: int = 5555
    
    # Output settings
    output_base_dir: str = "orchestrated_experiments"
    log_level: str = "INFO"
    
    # Reproducibility settings
    enforce_reproducibility: bool = True
    save_reproducibility_artifacts: bool = True
    
    # Validation settings
    validate_before_execution: bool = True
    validate_after_execution: bool = True
    
    # Safety settings
    max_experiment_duration_hours: float = 24.0
    auto_stop_on_high_failure_rate: bool = True
    failure_rate_threshold: float = 0.8

class ExperimentOrchestrator:
    """
    Master orchestration system that manages all aspects of experiment execution
    """
    
    def __init__(self, config: OrchestrationConfig = None):
        
        if config is None:
            config = OrchestrationConfig()
        
        self.config = config
        
        # Setup output directory
        self.output_dir = Path(config.output_base_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.executor = None
        self.resource_monitor = None
        self.validator = None
        self.repro_manager = None
        self.dashboard = None
        
        # State tracking
        self.orchestration_active = False
        self.orchestration_start_time = None
        self.total_experiments_run = 0
        self.experiment_suite_results = []
        
        # Safety monitoring
        self.safety_monitor_active = False
        self.safety_monitor_thread = None
        
        # Initialize all components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("ExperimentOrchestrator initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for orchestration"""
        
        logger = logging.getLogger("experiment_orchestrator")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        log_file = self.output_dir / f"orchestration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize all orchestration components"""
        
        try:
            # Initialize executor
            self.executor = ExperimentExecutor(
                output_dir=str(self.output_dir / "experiments"),
                max_workers=self.config.max_parallel_experiments,
                enable_monitoring=self.config.enable_resource_monitoring,
                enable_validation=self.config.enable_validation,
                log_level=self.config.log_level
            )
            
            # Initialize resource monitor
            if self.config.enable_resource_monitoring:
                self.resource_monitor = ResourceMonitor(
                    monitoring_interval=self.config.monitoring_interval,
                    alert_thresholds={
                        'cpu_percent': self.config.cpu_limit_percent,
                        'memory_percent': 90.0,
                        'disk_usage_percent': 95.0
                    }
                )
                self.resource_monitor.start_monitoring()
            
            # Initialize validator
            if self.config.enable_validation:
                self.validator = ExperimentValidator()
            
            # Initialize reproducibility manager
            if self.config.enforce_reproducibility:
                self.repro_manager = ReproducibilityManager(
                    base_output_dir=str(self.output_dir / "reproducibility")
                )
            
            # Initialize dashboard
            if self.config.enable_dashboard:
                self.dashboard = ExperimentDashboard(
                    executor=self.executor,
                    resource_monitor=self.resource_monitor,
                    port=self.config.dashboard_port
                )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_orchestration()
    
    def create_experiment_suite(self, 
                              experiment_types: List[ExperimentType],
                              base_params: ExperimentParams = None,
                              parameter_grid: Dict[str, List[Any]] = None) -> List[ExperimentJob]:
        """
        Create comprehensive experiment suite with parameter variations
        
        Args:
            experiment_types: Types of experiments to run
            base_params: Base parameters for all experiments
            parameter_grid: Dictionary of parameters and their values to sweep
            
        Returns:
            List of experiment jobs
        """
        
        if base_params is None:
            base_params = ExperimentParams()
        
        if parameter_grid is None:
            parameter_grid = {}
        
        experiment_jobs = []
        
        for exp_type in experiment_types:
            # Get experiment-specific configuration
            exp_config = EXPERIMENT_CONFIGS[exp_type]
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(base_params, parameter_grid)
            
            for i, params in enumerate(param_combinations):
                # Create job
                job_id = f"{exp_type.value}_param_set_{i:03d}"
                
                job = ExperimentJob(
                    job_id=job_id,
                    experiment_type=exp_type,
                    params=params,
                    priority=1,
                    max_retries=2,
                    timeout_minutes=int(self.config.max_experiment_duration_hours * 60),
                    resource_limits={
                        'max_memory_gb': self.config.memory_limit_gb,
                        'max_cpu_percent': self.config.cpu_limit_percent,
                        'max_disk_gb': self.config.disk_limit_gb
                    },
                    metadata={
                        'created_by': 'orchestrator',
                        'creation_time': datetime.now().isoformat(),
                        'parameter_set_index': i
                    }
                )
                
                experiment_jobs.append(job)
        
        self.logger.info(f"Created experiment suite with {len(experiment_jobs)} jobs")
        
        return experiment_jobs
    
    def _generate_parameter_combinations(self, 
                                       base_params: ExperimentParams,
                                       parameter_grid: Dict[str, List[Any]]) -> List[ExperimentParams]:
        """Generate all parameter combinations from parameter grid"""
        
        if not parameter_grid:
            return [base_params]
        
        import itertools
        
        param_combinations = []
        
        # Get parameter names and values
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        # Generate all combinations
        for combination in itertools.product(*param_values):
            # Create new parameters object
            new_params = ExperimentParams(**asdict(base_params))
            
            # Update with combination values
            for param_name, param_value in zip(param_names, combination):
                setattr(new_params, param_name, param_value)
            
            param_combinations.append(new_params)
        
        return param_combinations
    
    def run_experiment_suite(self, 
                           jobs: List[ExperimentJob],
                           run_dashboard: bool = None,
                           validate_before: bool = None,
                           validate_after: bool = None) -> Dict[str, Any]:
        """
        Run complete experiment suite with full orchestration
        
        Args:
            jobs: List of experiment jobs to run
            run_dashboard: Whether to run dashboard (defaults to config)
            validate_before: Whether to validate before execution (defaults to config)
            validate_after: Whether to validate after execution (defaults to config)
            
        Returns:
            Comprehensive results summary
        """
        
        # Use config defaults if not specified
        if run_dashboard is None:
            run_dashboard = self.config.enable_dashboard
        if validate_before is None:
            validate_before = self.config.validate_before_execution
        if validate_after is None:
            validate_after = self.config.validate_after_execution
        
        self.orchestration_start_time = datetime.now()
        self.orchestration_active = True
        
        self.logger.info(f"Starting orchestration of {len(jobs)} experiments")
        
        try:
            # Pre-execution validation
            if validate_before:
                self.logger.info("Performing pre-execution validation...")
                validation_results = self._validate_experiment_suite(jobs)
                
                if not validation_results['all_valid']:
                    self.logger.error("Pre-execution validation failed")
                    return {
                        'status': 'failed',
                        'error': 'Pre-execution validation failed',
                        'validation_results': validation_results
                    }
                
                self.logger.info("Pre-execution validation passed")
            
            # Setup reproducibility for the entire suite
            if self.config.enforce_reproducibility:
                self.logger.info("Setting up reproducible environment...")
                repro_info = self.repro_manager.setup_reproducible_environment()
                self.logger.info(f"Reproducible environment setup: {repro_info['environment_hash']}")
            
            # Start dashboard if requested
            dashboard_thread = None
            if run_dashboard and self.dashboard:
                dashboard_thread = threading.Thread(
                    target=lambda: self.dashboard.run(open_browser=True), 
                    daemon=True
                )
                dashboard_thread.start()
                self.logger.info(f"Dashboard started at http://127.0.0.1:{self.config.dashboard_port}")
                
                # Give dashboard time to start
                time.sleep(2)
            
            # Start safety monitoring
            self._start_safety_monitoring()
            
            # Add jobs to executor
            for job in jobs:
                self.executor.add_experiment(job)
            
            # Execute all experiments
            self.logger.info("Beginning experiment execution...")
            execution_results = self.executor.execute_all(
                save_intermediate=True,
                validate_results=validate_after
            )
            
            # Post-execution analysis
            self.logger.info("Performing post-execution analysis...")
            analysis_results = self._analyze_suite_results(execution_results)
            
            # Save comprehensive results
            final_results = {
                'orchestration_info': {
                    'start_time': self.orchestration_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_duration': (datetime.now() - self.orchestration_start_time).total_seconds(),
                    'config': asdict(self.config)
                },
                'execution_results': execution_results,
                'analysis_results': analysis_results,
                'resource_usage_summary': self._get_resource_usage_summary(),
                'reproducibility_info': repro_info if self.config.enforce_reproducibility else None
            }
            
            # Save results
            self._save_orchestration_results(final_results)
            
            # Create reproducibility archive if enabled
            if self.config.save_reproducibility_artifacts:
                self.logger.info("Creating reproducibility archive...")
                archive_path = self.repro_manager.create_reproducibility_archive(
                    final_results, self.output_dir
                )
                final_results['reproducibility_archive'] = str(archive_path)
            
            self.logger.info("Experiment suite completed successfully")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'partial_results': getattr(self, 'partial_results', None)
            }
        
        finally:
            self.orchestration_active = False
            self._stop_safety_monitoring()
    
    def _validate_experiment_suite(self, jobs: List[ExperimentJob]) -> Dict[str, Any]:
        """Validate entire experiment suite before execution"""
        
        if not self.validator:
            return {'all_valid': True, 'message': 'Validation disabled'}
        
        validation_results = []
        
        for job in jobs:
            result = self.validator.validate_experiment_job(job)
            validation_results.append(result)
        
        # Generate validation report
        report = self.validator.generate_validation_report(validation_results)
        
        return {
            'all_valid': report['summary']['success_rate'] == 1.0,
            'total_jobs': len(jobs),
            'valid_jobs': report['summary']['valid_count'],
            'invalid_jobs': report['summary']['invalid_count'],
            'total_errors': report['summary']['total_errors'],
            'total_warnings': report['summary']['total_warnings'],
            'detailed_report': report
        }
    
    def _analyze_suite_results(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results across entire experiment suite"""
        
        analysis = {
            'performance_analysis': self._analyze_performance_trends(execution_results),
            'resource_efficiency': self._analyze_resource_efficiency(execution_results),
            'method_comparison': self._compare_methods_across_experiments(execution_results),
            'statistical_summary': self._compute_statistical_summary(execution_results),
            'recommendations': self._generate_recommendations(execution_results)
        }
        
        return analysis
    
    def _analyze_performance_trends(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends across experiments"""
        
        trends = {
            'accuracy_trends': [],
            'fairness_trends': [],
            'efficiency_trends': [],
            'convergence_patterns': []
        }
        
        # Extract trends from results
        # This is a placeholder - implement based on your specific metrics structure
        
        return trends
    
    def _analyze_resource_efficiency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource usage efficiency"""
        
        if not self.resource_monitor:
            return {'message': 'Resource monitoring not available'}
        
        summary = self.resource_monitor.get_summary()
        
        efficiency_analysis = {
            'avg_cpu_utilization': summary.get('cpu_usage', {}).get('mean', 0),
            'peak_memory_usage': summary.get('memory_usage', {}).get('max_gb', 0),
            'resource_alerts_count': len(summary.get('alerts', {}).get('recent_alerts', [])),
            'efficiency_score': self._compute_efficiency_score(summary)
        }
        
        return efficiency_analysis
    
    def _compare_methods_across_experiments(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different methods across all experiments"""
        
        method_comparison = {
            'method_rankings': {},
            'statistical_significance': {},
            'pareto_analysis': {}
        }
        
        # Implement method comparison logic based on your results structure
        
        return method_comparison
    
    def _compute_statistical_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistical summary of all results"""
        
        return {
            'total_experiments': len(results.get('detailed_results', [])),
            'success_rate': results.get('job_statistics', {}).get('success_rate', 0),
            'avg_execution_time': 0,  # Compute from results
            'confidence_intervals': {},  # Compute CI for key metrics
            'effect_sizes': {}  # Compute effect sizes for comparisons
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results"""
        
        recommendations = []
        
        # Analyze success rate
        success_rate = results.get('job_statistics', {}).get('success_rate', 0)
        if success_rate < 0.8:
            recommendations.append("Low success rate detected. Consider reviewing experiment parameters and resource limits.")
        
        # Analyze resource usage
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            summary = self.resource_monitor.get_summary()
            
            if summary.get('cpu_usage', {}).get('mean', 0) < 30:
                recommendations.append("Low CPU utilization. Consider increasing parallel workers.")
            
            if summary.get('memory_usage', {}).get('max_gb', 0) > self.config.memory_limit_gb * 0.9:
                recommendations.append("High memory usage detected. Consider reducing batch size or increasing memory limits.")
        
        return recommendations
    
    def _compute_efficiency_score(self, resource_summary: Dict[str, Any]) -> float:
        """Compute overall resource efficiency score (0-1)"""
        
        try:
            cpu_usage = resource_summary.get('cpu_usage', {}).get('mean', 0)
            memory_usage = resource_summary.get('memory_usage', {}).get('mean_percent', 0)
            alerts_count = len(resource_summary.get('alerts', {}).get('recent_alerts', []))
            
            # Ideal CPU usage is around 70-80%
            cpu_efficiency = 1.0 - abs(cpu_usage - 75) / 75
            cpu_efficiency = max(0, min(1, cpu_efficiency))
            
            # Memory usage should be reasonable
            memory_efficiency = 1.0 - (memory_usage / 100)
            memory_efficiency = max(0.1, memory_efficiency)
            
            # Penalize for resource alerts
            alert_penalty = min(0.5, alerts_count * 0.1)
            
            efficiency_score = (cpu_efficiency * 0.4 + memory_efficiency * 0.4 + (1 - alert_penalty) * 0.2)
            
            return round(efficiency_score, 3)
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _get_resource_usage_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive resource usage summary"""
        
        if not self.resource_monitor:
            return None
        
        return self.resource_monitor.get_summary()
    
    def _save_orchestration_results(self, results: Dict[str, Any]):
        """Save comprehensive orchestration results"""
        
        # Save main results
        results_file = self.output_dir / f"orchestration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Orchestration results saved to {results_file}")
        
        # Save resource monitoring data if available
        if self.resource_monitor:
            resource_file = self.output_dir / "resource_usage_detailed.json"
            self.resource_monitor.export_data(resource_file, 'json')
    
    def _start_safety_monitoring(self):
        """Start safety monitoring thread"""
        
        self.safety_monitor_active = True
        self.safety_monitor_thread = threading.Thread(target=self._safety_monitoring_loop, daemon=True)
        self.safety_monitor_thread.start()
        
        self.logger.info("Safety monitoring started")
    
    def _stop_safety_monitoring(self):
        """Stop safety monitoring"""
        
        self.safety_monitor_active = False
        
        if self.safety_monitor_thread:
            self.safety_monitor_thread.join(timeout=5)
        
        self.logger.info("Safety monitoring stopped")
    
    def _safety_monitoring_loop(self):
        """Safety monitoring loop to catch dangerous conditions"""
        
        while self.safety_monitor_active and self.orchestration_active:
            try:
                # Check resource usage
                if self.resource_monitor:
                    current_usage = self.resource_monitor.get_current_usage()
                    
                    if current_usage:
                        # Check memory usage
                        if current_usage.memory_gb > self.config.memory_limit_gb:
                            self.logger.critical(f"Memory usage exceeded limit: {current_usage.memory_gb} GB")
                            self._trigger_emergency_stop("Memory limit exceeded")
                            break
                        
                        # Check disk space
                        if current_usage.disk_free_gb < 1.0:  # Less than 1 GB free
                            self.logger.critical("Disk space critically low")
                            self._trigger_emergency_stop("Disk space critically low")
                            break
                
                # Check execution duration
                if self.orchestration_start_time:
                    duration = datetime.now() - self.orchestration_start_time
                    max_duration = timedelta(hours=self.config.max_experiment_duration_hours)
                    
                    if duration > max_duration:
                        self.logger.critical("Maximum orchestration duration exceeded")
                        self._trigger_emergency_stop("Maximum duration exceeded")
                        break
                
                # Check failure rate
                if self.config.auto_stop_on_high_failure_rate and self.executor:
                    status = self.executor.get_status()
                    total_completed = status['jobs_completed'] + status['jobs_failed']
                    
                    if total_completed > 5:  # Only check after some experiments
                        failure_rate = status['jobs_failed'] / total_completed
                        
                        if failure_rate > self.config.failure_rate_threshold:
                            self.logger.critical(f"High failure rate detected: {failure_rate:.2%}")
                            self._trigger_emergency_stop("High failure rate")
                            break
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in safety monitoring: {e}")
                time.sleep(30)
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop of all operations"""
        
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Stop executor
        if self.executor:
            self.executor.pause_execution()
        
        # Save current state
        emergency_state = {
            'emergency_stop_time': datetime.now().isoformat(),
            'reason': reason,
            'orchestration_start_time': self.orchestration_start_time.isoformat() if self.orchestration_start_time else None,
            'executor_status': self.executor.get_status() if self.executor else None,
            'resource_usage': self.resource_monitor.get_current_usage() if self.resource_monitor else None
        }
        
        emergency_file = self.output_dir / f"emergency_stop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(emergency_file, 'w') as f:
            json.dump(emergency_state, f, indent=2, default=str)
        
        self.orchestration_active = False
    
    def stop_orchestration(self):
        """Stop orchestration gracefully"""
        
        self.logger.info("Stopping orchestration...")
        
        self.orchestration_active = False
        
        # Stop executor
        if self.executor:
            self.executor.pause_execution()
        
        # Stop monitoring
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        # Stop safety monitoring
        self._stop_safety_monitoring()
        
        self.logger.info("Orchestration stopped")
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        
        status = {
            'orchestration_active': self.orchestration_active,
            'orchestration_start_time': self.orchestration_start_time.isoformat() if self.orchestration_start_time else None,
            'executor_status': self.executor.get_status() if self.executor else None,
            'resource_monitor_active': self.resource_monitor.is_monitoring() if self.resource_monitor else False,
            'safety_monitor_active': self.safety_monitor_active,
            'config': asdict(self.config)
        }
        
        return status

# Convenience functions for common use cases
def create_standard_experiment_suite() -> List[ExperimentJob]:
    """Create standard experiment suite for subgroup fairness research"""
    
    orchestrator = ExperimentOrchestrator()
    
    # Define standard experiments
    experiment_types = [
        ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
        ExperimentType.COMPUTATIONAL_EFFICIENCY,
        ExperimentType.PARTIAL_VS_COMPLETE
    ]
    
    # Define parameter variations
    parameter_grid = {
        'lambda_values': [[0.0, 0.5, 1.0], [0.0, 1.0, 2.0]],
        'random_seeds': [[42, 123, 456], [789, 999, 111]]
    }
    
    return orchestrator.create_experiment_suite(experiment_types, parameter_grid=parameter_grid)

def run_quick_orchestrated_test(output_dir: str = "quick_test_orchestration") -> Dict[str, Any]:
    """Run quick orchestrated test with minimal experiments"""
    
    config = OrchestrationConfig(
        max_parallel_experiments=2,
        output_base_dir=output_dir,
        enable_dashboard=True,
        max_experiment_duration_hours=1.0
    )
    
    orchestrator = ExperimentOrchestrator(config)
    
    # Create simple test suite
    jobs = orchestrator.create_experiment_suite(
        experiment_types=[ExperimentType.ACCURACY_FAIRNESS_TRADEOFF],
        parameter_grid={'lambda_values': [[0.0, 1.0]]}
    )
    
    # Run first 2 jobs only for quick test
    return orchestrator.run_experiment_suite(jobs[:2])

def main():
    """Main entry point for orchestration"""
    
    parser = argparse.ArgumentParser(description="Experiment Orchestration System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="orchestrated_experiments")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--dashboard-port", type=int, default=5555)
    parser.add_argument("--quick-test", action="store_true", help="Run quick test")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("Running quick orchestrated test...")
        results = run_quick_orchestrated_test(args.output_dir)
        print(f"Quick test completed! Check results in {args.output_dir}")
        return
    
    # Create configuration
    config = OrchestrationConfig(
        max_parallel_experiments=args.max_parallel,
        output_base_dir=args.output_dir,
        dashboard_port=args.dashboard_port,
        enable_dashboard=not args.no_dashboard
    )
    
    # Create orchestrator
    orchestrator = ExperimentOrchestrator(config)
    
    # Create standard experiment suite
    print("Creating standard experiment suite...")
    jobs = create_standard_experiment_suite()
    
    print(f"Running {len(jobs)} experiments with orchestration...")
    
    # Run orchestration
    results = orchestrator.run_experiment_suite(jobs)
    
    print("Orchestration completed!")
    print(f"Success rate: {results.get('execution_results', {}).get('job_statistics', {}).get('success_rate', 0):.1%}")
    print(f"Total duration: {results.get('orchestration_info', {}).get('total_duration', 0):.1f} seconds")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()