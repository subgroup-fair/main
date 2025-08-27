"""
Robust Automated Experiment Execution System
Handles parallel execution, resource monitoring, and graceful interruption handling
"""

import os
import sys
import json
import time
import psutil
import signal
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import logging
import traceback
import pickle
import hashlib
from dataclasses import dataclass, asdict
import subprocess
import shutil
import yaml

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from experimental_config import ExperimentType, ExperimentParams, EXPERIMENT_CONFIGS
from scripts.experiments.main_experiment_pipeline import ExperimentRunner
from scripts.automation.resource_monitor import ResourceMonitor
from scripts.automation.experiment_validator import ExperimentValidator
from scripts.automation.reproducibility_manager import ReproducibilityManager

@dataclass
class ExperimentJob:
    """Individual experiment job configuration"""
    job_id: str
    experiment_type: ExperimentType
    params: ExperimentParams
    priority: int = 0
    max_retries: int = 3
    timeout_minutes: int = 120
    resource_limits: Dict[str, Any] = None
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                'max_memory_gb': 8,
                'max_cpu_percent': 80,
                'max_disk_gb': 10
            }
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class JobResult:
    """Experiment job result"""
    job_id: str
    status: str  # 'completed', 'failed', 'timeout', 'interrupted'
    start_time: datetime
    end_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None

class ExperimentExecutor:
    """
    Robust experiment executor with parallel processing, monitoring, and fault tolerance
    """
    
    def __init__(self, 
                 output_dir: str = "automated_experiments",
                 max_workers: int = None,
                 enable_monitoring: bool = True,
                 enable_validation: bool = True,
                 log_level: str = "INFO"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Worker configuration
        self.max_workers = max_workers or min(4, mp.cpu_count())
        
        # Component initialization
        self.enable_monitoring = enable_monitoring
        self.enable_validation = enable_validation
        
        # State tracking
        self.jobs_queue: List[ExperimentJob] = []
        self.running_jobs: Dict[str, Any] = {}
        self.completed_jobs: List[JobResult] = []
        self.failed_jobs: List[JobResult] = []
        self.interrupted = False
        
        # Monitoring components
        if self.enable_monitoring:
            self.resource_monitor = ResourceMonitor()
            self.resource_monitor.start_monitoring()
        
        if self.enable_validation:
            self.validator = ExperimentValidator()
        
        self.repro_manager = ReproducibilityManager()
        
        # Logging setup
        self.logger = self._setup_logging(log_level)
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"ExperimentExecutor initialized with {self.max_workers} workers")
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("experiment_executor")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        log_file = self.output_dir / f"executor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.interrupted = True
        
        # Stop resource monitoring
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop_monitoring()
        
        # Save current state
        self._save_executor_state()
        
        self.logger.info("Graceful shutdown completed")
    
    def add_experiment(self, job: ExperimentJob) -> str:
        """Add experiment job to execution queue"""
        
        # Input validation
        if self.enable_validation:
            validation_result = self.validator.validate_experiment_job(job)
            if not validation_result['valid']:
                raise ValueError(f"Invalid experiment job: {validation_result['errors']}")
        
        # Generate unique job ID if not provided
        if not job.job_id:
            job.job_id = self._generate_job_id(job)
        
        # Check for duplicates
        existing_ids = [j.job_id for j in self.jobs_queue]
        if job.job_id in existing_ids:
            raise ValueError(f"Job ID {job.job_id} already exists")
        
        self.jobs_queue.append(job)
        self.logger.info(f"Added experiment job: {job.job_id}")
        
        return job.job_id
    
    def add_experiment_batch(self, jobs: List[ExperimentJob]) -> List[str]:
        """Add multiple experiment jobs"""
        job_ids = []
        for job in jobs:
            try:
                job_id = self.add_experiment(job)
                job_ids.append(job_id)
            except Exception as e:
                self.logger.error(f"Failed to add job {job.job_id}: {e}")
        
        return job_ids
    
    def execute_all(self, 
                   save_intermediate: bool = True,
                   validate_results: bool = True) -> Dict[str, Any]:
        """Execute all queued experiments with parallel processing"""
        
        if not self.jobs_queue:
            self.logger.warning("No jobs in queue")
            return {'status': 'no_jobs', 'results': []}
        
        start_time = datetime.now()
        self.logger.info(f"Starting execution of {len(self.jobs_queue)} jobs with {self.max_workers} workers")
        
        # Sort jobs by priority and dependencies
        sorted_jobs = self._resolve_job_dependencies()
        
        # Create execution batches based on dependencies
        execution_batches = self._create_execution_batches(sorted_jobs)
        
        total_results = []
        
        try:
            # Execute batches sequentially, jobs within batch in parallel
            for batch_idx, batch in enumerate(execution_batches):
                if self.interrupted:
                    break
                
                self.logger.info(f"Executing batch {batch_idx + 1}/{len(execution_batches)} with {len(batch)} jobs")
                
                batch_results = self._execute_job_batch(
                    batch, 
                    save_intermediate=save_intermediate,
                    validate_results=validate_results
                )
                
                total_results.extend(batch_results)
                
                # Save intermediate state after each batch
                if save_intermediate:
                    self._save_intermediate_results(batch_results, batch_idx)
        
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        finally:
            # Cleanup and final reporting
            end_time = datetime.now()
            duration = end_time - start_time
            
            execution_summary = self._generate_execution_summary(
                total_results, start_time, end_time, duration
            )
            
            # Save final results
            self._save_final_results(execution_summary)
            
            # Stop monitoring
            if self.enable_monitoring:
                self.resource_monitor.stop_monitoring()
            
            self.logger.info(f"Execution completed in {duration}")
            
        return execution_summary
    
    def _execute_job_batch(self, 
                          jobs: List[ExperimentJob],
                          save_intermediate: bool = True,
                          validate_results: bool = True) -> List[JobResult]:
        """Execute a batch of jobs in parallel"""
        
        batch_results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs in the batch
            future_to_job = {}
            
            for job in jobs:
                if self.interrupted:
                    break
                
                future = executor.submit(self._execute_single_job, job)
                future_to_job[future] = job
                self.running_jobs[job.job_id] = {
                    'job': job,
                    'future': future,
                    'start_time': datetime.now()
                }
            
            # Process completed jobs
            for future in as_completed(future_to_job):
                if self.interrupted:
                    break
                
                job = future_to_job[future]
                
                try:
                    # Get result with timeout
                    result = future.result(timeout=job.timeout_minutes * 60)
                    
                    # Validate results if enabled
                    if validate_results and self.enable_validation:
                        validation_result = self.validator.validate_experiment_results(result)
                        result.validation_results = validation_result
                    
                    batch_results.append(result)
                    
                    if result.status == 'completed':
                        self.completed_jobs.append(result)
                        self.logger.info(f"Job {job.job_id} completed successfully")
                    else:
                        self.failed_jobs.append(result)
                        self.logger.error(f"Job {job.job_id} failed: {result.error_message}")
                
                except Exception as e:
                    # Handle job execution errors
                    error_result = JobResult(
                        job_id=job.job_id,
                        status='failed',
                        start_time=self.running_jobs[job.job_id]['start_time'],
                        end_time=datetime.now(),
                        error_message=str(e)
                    )
                    
                    batch_results.append(error_result)
                    self.failed_jobs.append(error_result)
                    self.logger.error(f"Job {job.job_id} execution error: {e}")
                
                finally:
                    # Cleanup
                    if job.job_id in self.running_jobs:
                        del self.running_jobs[job.job_id]
        
        return batch_results
    
    def _execute_single_job(self, job: ExperimentJob) -> JobResult:
        """Execute a single experiment job in isolated process"""
        
        start_time = datetime.now()
        
        try:
            # Setup reproducibility
            repro_info = self.repro_manager.setup_reproducible_environment(
                job.params.random_seeds[0] if job.params.random_seeds else 42
            )
            
            # Resource monitoring for this job
            job_monitor = ResourceMonitor() if self.enable_monitoring else None
            if job_monitor:
                job_monitor.start_monitoring()
            
            # Create isolated experiment runner
            job_output_dir = self.output_dir / f"job_{job.job_id}"
            job_output_dir.mkdir(exist_ok=True, parents=True)
            
            runner = ExperimentRunner(
                output_dir=str(job_output_dir),
                log_level="INFO"
            )
            
            # Execute experiment
            experiment_results = runner.run_experiment(
                job.experiment_type,
                job.params
            )
            
            # Stop monitoring
            resource_usage = None
            if job_monitor:
                job_monitor.stop_monitoring()
                resource_usage = job_monitor.get_summary()
            
            # Create successful result
            result = JobResult(
                job_id=job.job_id,
                status='completed',
                start_time=start_time,
                end_time=datetime.now(),
                results=experiment_results,
                resource_usage=resource_usage
            )
            
            # Save job-specific results
            self._save_job_results(job, result, repro_info)
            
            return result
        
        except Exception as e:
            # Handle job failure
            return JobResult(
                job_id=job.job_id,
                status='failed',
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e),
                resource_usage=job_monitor.get_summary() if 'job_monitor' in locals() and job_monitor else None
            )
    
    def _resolve_job_dependencies(self) -> List[ExperimentJob]:
        """Resolve job dependencies and sort by priority"""
        
        # Simple topological sort for dependencies
        # For now, just sort by priority
        return sorted(self.jobs_queue, key=lambda x: x.priority, reverse=True)
    
    def _create_execution_batches(self, jobs: List[ExperimentJob]) -> List[List[ExperimentJob]]:
        """Create execution batches based on dependencies and resources"""
        
        # Simple batching: create batches of max_workers size
        batches = []
        current_batch = []
        
        for job in jobs:
            current_batch.append(job)
            
            if len(current_batch) >= self.max_workers:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining jobs
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _generate_job_id(self, job: ExperimentJob) -> str:
        """Generate unique job ID based on job configuration"""
        
        job_config = {
            'experiment_type': job.experiment_type.value,
            'params': asdict(job.params),
            'timestamp': datetime.now().isoformat()
        }
        
        config_str = json.dumps(job_config, sort_keys=True)
        job_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{job.experiment_type.value}_{job_hash}"
    
    def _save_job_results(self, job: ExperimentJob, result: JobResult, repro_info: Dict[str, Any]):
        """Save individual job results with reproducibility information"""
        
        job_dir = self.output_dir / f"job_{job.job_id}"
        
        # Save job configuration
        job_config_file = job_dir / "job_config.json"
        with open(job_config_file, 'w') as f:
            json.dump({
                'job': asdict(job),
                'reproducibility': repro_info
            }, f, indent=2, default=str)
        
        # Save results
        results_file = job_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save reproducibility artifacts
        self.repro_manager.save_reproducibility_artifacts(job_dir / "reproducibility")
    
    def _save_intermediate_results(self, batch_results: List[JobResult], batch_idx: int):
        """Save intermediate results after each batch"""
        
        intermediate_file = self.output_dir / f"intermediate_batch_{batch_idx}.json"
        
        with open(intermediate_file, 'w') as f:
            json.dump([asdict(result) for result in batch_results], f, indent=2, default=str)
        
        self.logger.info(f"Saved intermediate results for batch {batch_idx}")
    
    def _save_final_results(self, execution_summary: Dict[str, Any]):
        """Save final execution summary"""
        
        summary_file = self.output_dir / "execution_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(execution_summary, f, indent=2, default=str)
        
        self.logger.info(f"Saved final execution summary to {summary_file}")
    
    def _save_executor_state(self):
        """Save current executor state for recovery"""
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'jobs_queue': [asdict(job) for job in self.jobs_queue],
            'completed_jobs': [asdict(result) for result in self.completed_jobs],
            'failed_jobs': [asdict(result) for result in self.failed_jobs],
            'running_jobs': list(self.running_jobs.keys())
        }
        
        state_file = self.output_dir / "executor_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info("Saved executor state")
    
    def _generate_execution_summary(self, 
                                  results: List[JobResult],
                                  start_time: datetime,
                                  end_time: datetime,
                                  duration: timedelta) -> Dict[str, Any]:
        """Generate comprehensive execution summary"""
        
        completed = [r for r in results if r.status == 'completed']
        failed = [r for r in results if r.status == 'failed']
        
        summary = {
            'execution_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'max_workers': self.max_workers,
                'interrupted': self.interrupted
            },
            'job_statistics': {
                'total_jobs': len(results),
                'completed_jobs': len(completed),
                'failed_jobs': len(failed),
                'success_rate': len(completed) / len(results) if results else 0
            },
            'performance_summary': self._generate_performance_summary(results),
            'resource_usage_summary': self._generate_resource_summary(results),
            'detailed_results': [asdict(result) for result in results]
        }
        
        return summary
    
    def _generate_performance_summary(self, results: List[JobResult]) -> Dict[str, Any]:
        """Generate performance summary across all jobs"""
        
        if not results:
            return {}
        
        completed_results = [r for r in results if r.status == 'completed' and r.results]
        
        if not completed_results:
            return {'message': 'No completed results to analyze'}
        
        # Extract performance metrics
        accuracies = []
        fairness_metrics = []
        training_times = []
        
        for result in completed_results:
            if 'results' in result.results and 'datasets' in result.results['results']:
                for dataset_name, dataset_results in result.results['results']['datasets'].items():
                    if 'methods' in dataset_results:
                        for method_name, method_results in dataset_results['methods'].items():
                            if 'lambda_sweep' in method_results:
                                for sweep_point in method_results['lambda_sweep']:
                                    if 'accuracy_mean' in sweep_point:
                                        accuracies.append(sweep_point['accuracy_mean'])
                                    if 'sup_ipm_mean' in sweep_point:
                                        fairness_metrics.append(sweep_point['sup_ipm_mean'])
                                    if 'training_time_mean' in sweep_point:
                                        training_times.append(sweep_point['training_time_mean'])
        
        performance_summary = {}
        
        if accuracies:
            performance_summary['accuracy'] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'count': len(accuracies)
            }
        
        if fairness_metrics:
            performance_summary['fairness'] = {
                'mean': np.mean(fairness_metrics),
                'std': np.std(fairness_metrics),
                'min': np.min(fairness_metrics),
                'max': np.max(fairness_metrics),
                'count': len(fairness_metrics)
            }
        
        if training_times:
            performance_summary['training_time'] = {
                'mean': np.mean(training_times),
                'std': np.std(training_times),
                'min': np.min(training_times),
                'max': np.max(training_times),
                'count': len(training_times)
            }
        
        return performance_summary
    
    def _generate_resource_summary(self, results: List[JobResult]) -> Dict[str, Any]:
        """Generate resource usage summary"""
        
        resource_results = [r for r in results if r.resource_usage]
        
        if not resource_results:
            return {'message': 'No resource usage data available'}
        
        # Aggregate resource metrics
        cpu_usages = []
        memory_usages = []
        disk_usages = []
        
        for result in resource_results:
            usage = result.resource_usage
            if 'cpu_percent' in usage:
                cpu_usages.append(usage['cpu_percent'])
            if 'memory_gb' in usage:
                memory_usages.append(usage['memory_gb'])
            if 'disk_gb' in usage:
                disk_usages.append(usage['disk_gb'])
        
        summary = {}
        
        if cpu_usages:
            summary['cpu_usage'] = {
                'mean': np.mean(cpu_usages),
                'max': np.max(cpu_usages),
                'count': len(cpu_usages)
            }
        
        if memory_usages:
            summary['memory_usage'] = {
                'mean': np.mean(memory_usages),
                'max': np.max(memory_usages),
                'count': len(memory_usages)
            }
        
        if disk_usages:
            summary['disk_usage'] = {
                'mean': np.mean(disk_usages),
                'max': np.max(disk_usages),
                'count': len(disk_usages)
            }
        
        return summary
    
    def get_status(self) -> Dict[str, Any]:
        """Get current executor status"""
        
        return {
            'jobs_in_queue': len(self.jobs_queue),
            'jobs_running': len(self.running_jobs),
            'jobs_completed': len(self.completed_jobs),
            'jobs_failed': len(self.failed_jobs),
            'interrupted': self.interrupted,
            'monitoring_enabled': self.enable_monitoring,
            'validation_enabled': self.enable_validation
        }
    
    def pause_execution(self):
        """Pause experiment execution"""
        self.interrupted = True
        self.logger.info("Execution paused by user request")
    
    def resume_execution(self):
        """Resume experiment execution"""
        self.interrupted = False
        self.logger.info("Execution resumed")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job"""
        
        # Remove from queue if not started
        for i, job in enumerate(self.jobs_queue):
            if job.job_id == job_id:
                del self.jobs_queue[i]
                self.logger.info(f"Cancelled queued job {job_id}")
                return True
        
        # Cancel running job if possible
        if job_id in self.running_jobs:
            # Note: Cancelling running jobs in ProcessPoolExecutor is complex
            # This is a simplified implementation
            self.logger.warning(f"Job {job_id} is running and cannot be easily cancelled")
            return False
        
        return False

# Convenience functions for common use cases
def create_experiment_suite(experiment_types: List[ExperimentType],
                          param_variations: Dict[str, List[Any]],
                          base_params: ExperimentParams = None) -> List[ExperimentJob]:
    """Create a suite of experiments with parameter variations"""
    
    if base_params is None:
        base_params = ExperimentParams()
    
    jobs = []
    
    for exp_type in experiment_types:
        for param_name, param_values in param_variations.items():
            for param_value in param_values:
                # Create modified parameters
                modified_params = ExperimentParams(**asdict(base_params))
                setattr(modified_params, param_name, param_value)
                
                # Create job
                job = ExperimentJob(
                    job_id=f"{exp_type.value}_{param_name}_{param_value}",
                    experiment_type=exp_type,
                    params=modified_params,
                    metadata={
                        'param_variation': param_name,
                        'param_value': param_value
                    }
                )
                
                jobs.append(job)
    
    return jobs

def run_quick_experiments(output_dir: str = "quick_experiments") -> Dict[str, Any]:
    """Run a quick set of experiments for testing"""
    
    executor = ExperimentExecutor(output_dir=output_dir, max_workers=2)
    
    # Create simple experiment jobs
    jobs = [
        ExperimentJob(
            job_id="quick_test_1",
            experiment_type=ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
            params=ExperimentParams(
                lambda_values=[0.0, 1.0],
                random_seeds=[42, 123]
            ),
            timeout_minutes=30
        )
    ]
    
    # Add jobs and execute
    for job in jobs:
        executor.add_experiment(job)
    
    return executor.execute_all()

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Experiment Executor")
    parser.add_argument("--output-dir", type=str, default="automated_experiments")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--quick-test", action="store_true")
    
    args = parser.parse_args()
    
    if args.quick_test:
        results = run_quick_experiments(args.output_dir)
        print("Quick test completed!")
        print(f"Results summary: {results['job_statistics']}")
    else:
        print("Use run_quick_experiments() or create custom experiment suite")