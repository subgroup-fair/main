"""
Unified Validation Controller
Comprehensive experiment validation orchestration and automation
"""

import json
import logging
import os
import threading
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# Validation modules
from .statistical_validator import StatisticalValidator
from .code_quality_validator import CodeQualityValidator  
from .scientific_rigor_validator import ScientificRigorValidator
from .validation_reporter import ValidationReporter

# Advanced components (with graceful fallback)
try:
    from .interactive_dashboard import InteractiveDashboardGenerator
    from .advanced_visualization import AdvancedVisualizationEngine
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

@dataclass
class ValidationConfig:
    """Configuration for validation workflow"""
    experiment_name: str
    experiment_path: str
    output_dir: str = "validation_outputs"
    
    # Validation components to run
    run_statistical: bool = True
    run_code_quality: bool = True
    run_scientific_rigor: bool = True
    generate_reports: bool = True
    launch_dashboard: bool = False
    export_visualizations: bool = False
    
    # Statistical validation config
    statistical_config: Dict[str, Any] = None
    
    # Code quality config  
    code_quality_config: Dict[str, Any] = None
    
    # Scientific rigor config
    scientific_rigor_config: Dict[str, Any] = None
    
    # Reporting config
    report_formats: List[str] = None
    visualization_formats: List[str] = None
    
    # Automation config
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_minutes: int = 30
    retry_attempts: int = 2
    
    # Notification config
    send_notifications: bool = False
    notification_webhook: Optional[str] = None
    notification_email: Optional[str] = None

@dataclass 
class ValidationResult:
    """Result of validation workflow"""
    experiment_name: str
    validation_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Component results
    statistical_results: Optional[Dict[str, Any]] = None
    code_quality_results: Optional[Dict[str, Any]] = None
    scientific_rigor_results: Optional[Dict[str, Any]] = None
    
    # Overall results
    overall_score: float = 0.0
    overall_grade: str = "F"
    validation_status: str = "pending"  # pending, running, completed, failed
    
    # Generated artifacts
    report_files: Dict[str, str] = None
    visualization_files: Dict[str, str] = None
    dashboard_url: Optional[str] = None
    
    # Errors and warnings
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.report_files is None:
            self.report_files = {}
        if self.visualization_files is None:
            self.visualization_files = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class ValidationWorkflow:
    """Automated validation workflow orchestrator"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger("validation_workflow")
        
        # Initialize output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize validation components
        self.statistical_validator = None
        self.code_quality_validator = None
        self.scientific_rigor_validator = None
        self.validation_reporter = None
        
        # Initialize workflow state
        self.current_result = None
        self.is_running = False
        self.progress_callbacks: List[Callable] = []
        
        # Thread pool for parallel execution
        self.executor = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for validation workflow"""
        
        log_file = self.output_dir / f"{self.config.experiment_name}_validation.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def add_progress_callback(self, callback: Callable[[str, float], None]):
        """Add progress callback function"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, message: str, progress: float):
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(message, progress)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
    
    async def run_validation_async(self) -> ValidationResult:
        """Run validation workflow asynchronously"""
        
        if self.is_running:
            raise RuntimeError("Validation workflow is already running")
        
        self.is_running = True
        validation_id = f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        # Initialize result
        self.current_result = ValidationResult(
            experiment_name=self.config.experiment_name,
            validation_id=validation_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            duration_seconds=0.0,
            validation_status="running"
        )
        
        self.logger.info(f"Starting validation workflow: {validation_id}")
        self._notify_progress("Validation workflow started", 0.0)
        
        try:
            # Initialize validators
            await self._initialize_validators()
            self._notify_progress("Validators initialized", 10.0)
            
            # Run validation components
            if self.config.parallel_execution:
                await self._run_validations_parallel()
            else:
                await self._run_validations_sequential()
            
            self._notify_progress("Validation components completed", 70.0)
            
            # Calculate overall results
            self._calculate_overall_results()
            self._notify_progress("Overall results calculated", 80.0)
            
            # Generate reports and visualizations
            if self.config.generate_reports:
                await self._generate_reports_and_visualizations()
                self._notify_progress("Reports and visualizations generated", 90.0)
            
            # Launch dashboard if requested
            if self.config.launch_dashboard and ADVANCED_AVAILABLE:
                await self._launch_dashboard()
            
            # Mark as completed
            self.current_result.validation_status = "completed"
            self.current_result.end_time = datetime.now()
            self.current_result.duration_seconds = (
                self.current_result.end_time - self.current_result.start_time
            ).total_seconds()
            
            self.logger.info(f"Validation workflow completed: {validation_id}")
            self._notify_progress("Validation workflow completed", 100.0)
            
            # Send notifications if configured
            if self.config.send_notifications:
                await self._send_notifications()
            
        except Exception as e:
            self.current_result.validation_status = "failed"
            self.current_result.errors.append(str(e))
            self.current_result.end_time = datetime.now()
            self.current_result.duration_seconds = (
                self.current_result.end_time - self.current_result.start_time
            ).total_seconds()
            
            self.logger.error(f"Validation workflow failed: {e}")
            self._notify_progress(f"Validation workflow failed: {e}", 100.0)
            
            raise
        
        finally:
            self.is_running = False
            
            # Save final result
            result_file = self.output_dir / f"{validation_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump(asdict(self.current_result), f, indent=2, default=str)
        
        return self.current_result
    
    def run_validation(self) -> ValidationResult:
        """Run validation workflow synchronously"""
        return asyncio.run(self.run_validation_async())
    
    async def _initialize_validators(self):
        """Initialize validation components"""
        
        self.logger.info("Initializing validation components")
        
        if self.config.run_statistical:
            self.statistical_validator = StatisticalValidator()
            if self.config.statistical_config:
                # Apply custom configuration
                pass
        
        if self.config.run_code_quality:
            self.code_quality_validator = CodeQualityValidator()
            if self.config.code_quality_config:
                # Apply custom configuration
                pass
        
        if self.config.run_scientific_rigor:
            self.scientific_rigor_validator = ScientificRigorValidator()
            if self.config.scientific_rigor_config:
                # Apply custom configuration
                pass
        
        if self.config.generate_reports:
            self.validation_reporter = ValidationReporter(str(self.output_dir))
    
    async def _run_validations_parallel(self):
        """Run validation components in parallel"""
        
        self.logger.info("Running validation components in parallel")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            # Submit validation tasks
            if self.config.run_statistical and self.statistical_validator:
                future = executor.submit(self._run_statistical_validation)
                futures["statistical"] = future
            
            if self.config.run_code_quality and self.code_quality_validator:
                future = executor.submit(self._run_code_quality_validation)
                futures["code_quality"] = future
            
            if self.config.run_scientific_rigor and self.scientific_rigor_validator:
                future = executor.submit(self._run_scientific_rigor_validation)
                futures["scientific_rigor"] = future
            
            # Collect results with timeout
            timeout_seconds = self.config.timeout_minutes * 60
            completed_count = 0
            total_count = len(futures)
            
            for future in as_completed(futures.values(), timeout=timeout_seconds):
                try:
                    # Find which validation this future corresponds to
                    validation_type = None
                    for vtype, vfuture in futures.items():
                        if vfuture == future:
                            validation_type = vtype
                            break
                    
                    result = future.result()
                    
                    # Store result
                    if validation_type == "statistical":
                        self.current_result.statistical_results = result
                    elif validation_type == "code_quality":
                        self.current_result.code_quality_results = result
                    elif validation_type == "scientific_rigor":
                        self.current_result.scientific_rigor_results = result
                    
                    completed_count += 1
                    progress = 20.0 + (completed_count / total_count) * 50.0
                    self._notify_progress(f"Completed {validation_type} validation", progress)
                    
                    self.logger.info(f"Completed {validation_type} validation")
                    
                except Exception as e:
                    self.current_result.errors.append(f"Validation error: {e}")
                    self.logger.error(f"Validation component error: {e}")
    
    async def _run_validations_sequential(self):
        """Run validation components sequentially"""
        
        self.logger.info("Running validation components sequentially")
        
        progress_step = 50.0 / sum([
            self.config.run_statistical,
            self.config.run_code_quality, 
            self.config.run_scientific_rigor
        ])
        current_progress = 20.0
        
        if self.config.run_statistical and self.statistical_validator:
            try:
                self.current_result.statistical_results = self._run_statistical_validation()
                current_progress += progress_step
                self._notify_progress("Statistical validation completed", current_progress)
            except Exception as e:
                self.current_result.errors.append(f"Statistical validation error: {e}")
                self.logger.error(f"Statistical validation error: {e}")
        
        if self.config.run_code_quality and self.code_quality_validator:
            try:
                self.current_result.code_quality_results = self._run_code_quality_validation()
                current_progress += progress_step
                self._notify_progress("Code quality validation completed", current_progress)
            except Exception as e:
                self.current_result.errors.append(f"Code quality validation error: {e}")
                self.logger.error(f"Code quality validation error: {e}")
        
        if self.config.run_scientific_rigor and self.scientific_rigor_validator:
            try:
                self.current_result.scientific_rigor_results = self._run_scientific_rigor_validation()
                current_progress += progress_step
                self._notify_progress("Scientific rigor validation completed", current_progress)
            except Exception as e:
                self.current_result.errors.append(f"Scientific rigor validation error: {e}")
                self.logger.error(f"Scientific rigor validation error: {e}")
    
    def _run_statistical_validation(self) -> Dict[str, Any]:
        """Run statistical validation"""
        self.logger.info("Running statistical validation")
        
        # Mock data for demonstration - replace with actual experiment data
        mock_data = {
            'results_data': [1.2, 1.5, 1.8, 2.1, 1.9, 1.7, 1.3, 1.6],
            'baseline_data': [1.0, 1.1, 1.2, 1.3, 1.1, 1.0, 1.2, 1.1],
            'experiment_params': {
                'alpha': 0.05,
                'power': 0.8,
                'effect_size_threshold': 0.3
            }
        }
        
        return self.statistical_validator.validate_experiment(mock_data)
    
    def _run_code_quality_validation(self) -> Dict[str, Any]:
        """Run code quality validation"""
        self.logger.info("Running code quality validation")
        
        return self.code_quality_validator.validate_experiment_code(
            self.config.experiment_path
        )
    
    def _run_scientific_rigor_validation(self) -> Dict[str, Any]:
        """Run scientific rigor validation"""
        self.logger.info("Running scientific rigor validation")
        
        # Mock experiment metadata
        mock_metadata = {
            'methodology': 'randomized_controlled_trial',
            'sample_size': 1000,
            'duration_days': 30,
            'control_group_size': 500,
            'treatment_group_size': 500,
            'primary_metric': 'conversion_rate',
            'secondary_metrics': ['click_through_rate', 'engagement_time'],
            'confounding_variables': ['age', 'gender', 'location'],
            'randomization_method': 'stratified'
        }
        
        return self.scientific_rigor_validator.validate_experiment_design(mock_metadata)
    
    def _calculate_overall_results(self):
        """Calculate overall validation scores and grades"""
        
        scores = []
        
        # Extract scores from each validation component
        if self.current_result.statistical_results:
            stat_score = self.current_result.statistical_results.get('overall_score', 0)
            scores.append(stat_score)
        
        if self.current_result.code_quality_results:
            code_score = self.current_result.code_quality_results.get('overall_score', 0)
            scores.append(code_score)
        
        if self.current_result.scientific_rigor_results:
            rigor_score = self.current_result.scientific_rigor_results.get('overall_score', 0)
            scores.append(rigor_score)
        
        # Calculate weighted average
        if scores:
            self.current_result.overall_score = sum(scores) / len(scores)
        else:
            self.current_result.overall_score = 0.0
        
        # Determine grade
        if self.current_result.overall_score >= 90:
            self.current_result.overall_grade = "A"
        elif self.current_result.overall_score >= 80:
            self.current_result.overall_grade = "B"
        elif self.current_result.overall_score >= 70:
            self.current_result.overall_grade = "C"
        elif self.current_result.overall_score >= 60:
            self.current_result.overall_grade = "D"
        else:
            self.current_result.overall_grade = "F"
        
        self.logger.info(f"Overall validation score: {self.current_result.overall_score:.1f} ({self.current_result.overall_grade})")
    
    async def _generate_reports_and_visualizations(self):
        """Generate validation reports and visualizations"""
        
        self.logger.info("Generating validation reports and visualizations")
        
        if not self.validation_reporter:
            return
        
        # Combine all validation results
        combined_results = {}
        if self.current_result.statistical_results:
            combined_results["statistical_validation"] = self.current_result.statistical_results
        if self.current_result.code_quality_results:
            combined_results["code_quality"] = self.current_result.code_quality_results
        if self.current_result.scientific_rigor_results:
            combined_results["scientific_rigor"] = self.current_result.scientific_rigor_results
        
        # Generate comprehensive report
        comprehensive_report = self.validation_reporter.generate_comprehensive_report(
            combined_results, self.config.experiment_name
        )
        
        # Save report in multiple formats
        report_formats = self.config.report_formats or ['json', 'html']
        
        for format_type in report_formats:
            if format_type == 'json':
                report_file = self.output_dir / f"{self.current_result.validation_id}_report.json"
                with open(report_file, 'w') as f:
                    json.dump(comprehensive_report, f, indent=2, default=str)
                self.current_result.report_files['json'] = str(report_file)
            
            elif format_type == 'html':
                html_report = self._generate_html_report(comprehensive_report)
                report_file = self.output_dir / f"{self.current_result.validation_id}_report.html"
                with open(report_file, 'w') as f:
                    f.write(html_report)
                self.current_result.report_files['html'] = str(report_file)
        
        # Export advanced visualizations if requested
        if self.config.export_visualizations and hasattr(self.validation_reporter, 'export_advanced_visualizations'):
            viz_formats = self.config.visualization_formats or ['html', 'png']
            exported_viz = self.validation_reporter.export_advanced_visualizations(
                combined_results, self.config.experiment_name, viz_formats
            )
            self.current_result.visualization_files.update(exported_viz)
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report from report data"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report - {self.config.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .grade-A {{ color: #28a745; }}
                .grade-B {{ color: #17a2b8; }}
                .grade-C {{ color: #ffc107; }}
                .grade-D {{ color: #fd7e14; }}
                .grade-F {{ color: #dc3545; }}
                .metric {{ margin: 10px 0; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Validation Report</h1>
                <h2>Experiment: {self.config.experiment_name}</h2>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Overall Score: <span class="score grade-{self.current_result.overall_grade}">{self.current_result.overall_score:.1f} ({self.current_result.overall_grade})</span></p>
            </div>
            
            <div class="section">
                <h3>Executive Summary</h3>
                <p>{report_data.get('executive_summary', {}).get('summary', 'No summary available')}</p>
            </div>
            
            <div class="section">
                <h3>Key Findings</h3>
                <ul>
                    {self._format_list_items(report_data.get('executive_summary', {}).get('key_findings', []))}
                </ul>
            </div>
            
            <div class="section">
                <h3>Recommendations</h3>
                <div>
                    {self._format_recommendations(report_data.get('action_plan', {}).get('immediate_actions', []))}
                </div>
            </div>
            
            <div class="section">
                <h3>Validation Details</h3>
                <pre>{json.dumps(report_data, indent=2, default=str)}</pre>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _format_list_items(self, items: List[str]) -> str:
        """Format list items as HTML"""
        return "\n".join([f"<li>{item}</li>" for item in items])
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations as HTML"""
        return "\n".join([f'<div class="recommendation">{rec}</div>' for rec in recommendations])
    
    async def _launch_dashboard(self):
        """Launch interactive dashboard"""
        
        if not ADVANCED_AVAILABLE or not hasattr(self.validation_reporter, 'launch_interactive_dashboard'):
            self.logger.warning("Interactive dashboard not available")
            return
        
        try:
            combined_results = {}
            if self.current_result.statistical_results:
                combined_results["statistical_validation"] = self.current_result.statistical_results
            if self.current_result.code_quality_results:
                combined_results["code_quality"] = self.current_result.code_quality_results
            if self.current_result.scientific_rigor_results:
                combined_results["scientific_rigor"] = self.current_result.scientific_rigor_results
            
            dashboard_url = self.validation_reporter.launch_interactive_dashboard(
                combined_results, self.config.experiment_name
            )
            
            if dashboard_url:
                self.current_result.dashboard_url = dashboard_url
                self.logger.info(f"Interactive dashboard launched at: {dashboard_url}")
            
        except Exception as e:
            self.logger.error(f"Error launching dashboard: {e}")
            self.current_result.warnings.append(f"Dashboard launch failed: {e}")
    
    async def _send_notifications(self):
        """Send completion notifications"""
        
        self.logger.info("Sending validation completion notifications")
        
        notification_data = {
            "experiment_name": self.config.experiment_name,
            "validation_id": self.current_result.validation_id,
            "overall_score": self.current_result.overall_score,
            "overall_grade": self.current_result.overall_grade,
            "status": self.current_result.validation_status,
            "duration_minutes": self.current_result.duration_seconds / 60,
            "report_files": self.current_result.report_files,
            "dashboard_url": self.current_result.dashboard_url
        }
        
        # Send webhook notification
        if self.config.notification_webhook:
            try:
                import requests
                response = requests.post(
                    self.config.notification_webhook,
                    json=notification_data,
                    timeout=10
                )
                response.raise_for_status()
                self.logger.info("Webhook notification sent successfully")
            except Exception as e:
                self.logger.error(f"Webhook notification failed: {e}")
        
        # Send email notification (placeholder)
        if self.config.notification_email:
            self.logger.info(f"Email notification would be sent to: {self.config.notification_email}")

class UnifiedValidationController:
    """Main controller for unified validation system"""
    
    def __init__(self, base_output_dir: str = "validation_outputs"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger("unified_validation_controller")
        self.active_workflows: Dict[str, ValidationWorkflow] = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the controller"""
        
        log_file = self.base_output_dir / "unified_validation_controller.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def create_validation_workflow(self, config: ValidationConfig) -> ValidationWorkflow:
        """Create a new validation workflow"""
        
        # Ensure unique experiment names
        base_name = config.experiment_name
        counter = 1
        while config.experiment_name in self.active_workflows:
            config.experiment_name = f"{base_name}_{counter}"
            counter += 1
        
        # Create workflow
        workflow = ValidationWorkflow(config)
        self.active_workflows[config.experiment_name] = workflow
        
        self.logger.info(f"Created validation workflow: {config.experiment_name}")
        return workflow
    
    def run_validation(self, config: ValidationConfig) -> ValidationResult:
        """Create and run validation workflow"""
        
        workflow = self.create_validation_workflow(config)
        try:
            result = workflow.run_validation()
            return result
        finally:
            # Remove from active workflows when completed
            if config.experiment_name in self.active_workflows:
                del self.active_workflows[config.experiment_name]
    
    async def run_validation_async(self, config: ValidationConfig) -> ValidationResult:
        """Create and run validation workflow asynchronously"""
        
        workflow = self.create_validation_workflow(config)
        try:
            result = await workflow.run_validation_async()
            return result
        finally:
            # Remove from active workflows when completed
            if config.experiment_name in self.active_workflows:
                del self.active_workflows[config.experiment_name]
    
    def get_active_workflows(self) -> List[str]:
        """Get list of active workflow names"""
        return list(self.active_workflows.keys())
    
    def get_workflow_status(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow"""
        
        if experiment_name not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[experiment_name]
        
        if workflow.current_result:
            return {
                "experiment_name": workflow.current_result.experiment_name,
                "validation_id": workflow.current_result.validation_id,
                "status": workflow.current_result.validation_status,
                "overall_score": workflow.current_result.overall_score,
                "overall_grade": workflow.current_result.overall_grade,
                "start_time": workflow.current_result.start_time.isoformat(),
                "is_running": workflow.is_running,
                "errors": workflow.current_result.errors,
                "warnings": workflow.current_result.warnings
            }
        
        return {
            "experiment_name": experiment_name,
            "status": "initializing",
            "is_running": workflow.is_running
        }
    
    def cancel_workflow(self, experiment_name: str) -> bool:
        """Cancel a running workflow"""
        
        if experiment_name not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[experiment_name]
        
        if workflow.is_running:
            # Mark as cancelled and stop execution
            workflow.is_running = False
            if workflow.current_result:
                workflow.current_result.validation_status = "cancelled"
            
            self.logger.info(f"Cancelled validation workflow: {experiment_name}")
            return True
        
        return False
    
    def get_historical_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical validation results"""
        
        results = []
        
        # Search for result files in output directory
        for result_file in self.base_output_dir.rglob("*_result.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    results.append(result_data)
            except Exception as e:
                self.logger.warning(f"Could not load result file {result_file}: {e}")
        
        # Sort by start time (most recent first)
        results.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return results[:limit]


# Convenience functions for easy usage
def run_comprehensive_validation(experiment_name: str, experiment_path: str, 
                                output_dir: str = None) -> ValidationResult:
    """Run comprehensive validation with default configuration"""
    
    config = ValidationConfig(
        experiment_name=experiment_name,
        experiment_path=experiment_path,
        output_dir=output_dir or f"validation_outputs/{experiment_name}",
        run_statistical=True,
        run_code_quality=True,
        run_scientific_rigor=True,
        generate_reports=True,
        export_visualizations=True
    )
    
    controller = UnifiedValidationController()
    return controller.run_validation(config)


def create_validation_config(experiment_name: str, experiment_path: str, **kwargs) -> ValidationConfig:
    """Create validation configuration with sensible defaults"""
    
    defaults = {
        'output_dir': f"validation_outputs/{experiment_name}",
        'run_statistical': True,
        'run_code_quality': True,
        'run_scientific_rigor': True,
        'generate_reports': True,
        'export_visualizations': False,
        'launch_dashboard': False,
        'parallel_execution': True,
        'max_workers': 4,
        'timeout_minutes': 30,
        'report_formats': ['json', 'html'],
        'visualization_formats': ['html', 'png']
    }
    
    # Override defaults with provided kwargs
    for key, value in kwargs.items():
        if hasattr(ValidationConfig, key):
            defaults[key] = value
    
    return ValidationConfig(
        experiment_name=experiment_name,
        experiment_path=experiment_path,
        **defaults
    )


# Example usage and testing
if __name__ == "__main__":
    # Example: Run comprehensive validation
    config = create_validation_config(
        experiment_name="test_experiment",
        experiment_path="./test_experiment",
        launch_dashboard=False,
        export_visualizations=True
    )
    
    controller = UnifiedValidationController()
    result = controller.run_validation(config)
    
    print(f"Validation completed with score: {result.overall_score:.1f} ({result.overall_grade})")
    print(f"Duration: {result.duration_seconds:.1f} seconds")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.report_files:
        print(f"Reports generated: {list(result.report_files.keys())}")