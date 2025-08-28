"""
Validation CLI Interface
Command-line interface for comprehensive experiment validation
"""

import argparse
import asyncio
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import validation components
from .unified_validation_controller import (
    UnifiedValidationController,
    ValidationConfig,
    ValidationResult,
    create_validation_config,
    run_comprehensive_validation
)

class ValidationCLI:
    """Command-line interface for validation system"""
    
    def __init__(self):
        self.controller = UnifiedValidationController()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        
        parser = argparse.ArgumentParser(
            description="Comprehensive Experiment Validation System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run validation with default settings
  python validation_cli.py validate --name "my_experiment" --path "./experiment"
  
  # Run validation from config file
  python validation_cli.py validate --config validation_config.yaml
  
  # Run with custom settings
  python validation_cli.py validate --name "test" --path "./test" --output "./results" --dashboard
  
  # Check status of running validation
  python validation_cli.py status --name "my_experiment"
  
  # List historical results
  python validation_cli.py history --limit 10
  
  # Generate config template
  python validation_cli.py init-config --output validation_config.yaml
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Validate command
        validate_parser = subparsers.add_parser(
            'validate', help='Run validation workflow'
        )
        self._add_validate_arguments(validate_parser)
        
        # Status command
        status_parser = subparsers.add_parser(
            'status', help='Check validation status'
        )
        status_parser.add_argument(
            '--name', '-n', required=True,
            help='Experiment name to check status'
        )
        
        # History command
        history_parser = subparsers.add_parser(
            'history', help='Show validation history'
        )
        history_parser.add_argument(
            '--limit', '-l', type=int, default=10,
            help='Number of recent results to show'
        )
        history_parser.add_argument(
            '--format', '-f', choices=['table', 'json'], default='table',
            help='Output format'
        )
        
        # Cancel command
        cancel_parser = subparsers.add_parser(
            'cancel', help='Cancel running validation'
        )
        cancel_parser.add_argument(
            '--name', '-n', required=True,
            help='Experiment name to cancel'
        )
        
        # List active command
        list_parser = subparsers.add_parser(
            'list', help='List active validations'
        )
        
        # Init config command
        init_config_parser = subparsers.add_parser(
            'init-config', help='Generate configuration template'
        )
        init_config_parser.add_argument(
            '--output', '-o', default='validation_config.yaml',
            help='Output file for configuration template'
        )
        init_config_parser.add_argument(
            '--format', '-f', choices=['yaml', 'json'], default='yaml',
            help='Configuration format'
        )
        
        return parser
    
    def _add_validate_arguments(self, parser: argparse.ArgumentParser):
        """Add validation command arguments"""
        
        # Input arguments
        input_group = parser.add_argument_group('Input')
        input_group.add_argument(
            '--config', '-c',
            help='Configuration file (YAML or JSON)'
        )
        input_group.add_argument(
            '--name', '-n',
            help='Experiment name'
        )
        input_group.add_argument(
            '--path', '-p',
            help='Path to experiment code/data'
        )
        
        # Output arguments
        output_group = parser.add_argument_group('Output')
        output_group.add_argument(
            '--output', '-o',
            help='Output directory for validation results'
        )
        output_group.add_argument(
            '--report-formats', nargs='+', 
            choices=['json', 'html', 'csv'], default=['json', 'html'],
            help='Report output formats'
        )
        output_group.add_argument(
            '--viz-formats', nargs='+',
            choices=['html', 'png', 'svg', 'pdf'], default=['html', 'png'],
            help='Visualization output formats'
        )
        
        # Validation components
        components_group = parser.add_argument_group('Validation Components')
        components_group.add_argument(
            '--skip-statistical', action='store_true',
            help='Skip statistical validation'
        )
        components_group.add_argument(
            '--skip-code-quality', action='store_true',
            help='Skip code quality validation'
        )
        components_group.add_argument(
            '--skip-scientific-rigor', action='store_true',
            help='Skip scientific rigor validation'
        )
        components_group.add_argument(
            '--skip-reports', action='store_true',
            help='Skip report generation'
        )
        
        # Advanced features
        advanced_group = parser.add_argument_group('Advanced Features')
        advanced_group.add_argument(
            '--dashboard', action='store_true',
            help='Launch interactive dashboard'
        )
        advanced_group.add_argument(
            '--export-viz', action='store_true',
            help='Export advanced visualizations'
        )
        
        # Execution settings
        execution_group = parser.add_argument_group('Execution Settings')
        execution_group.add_argument(
            '--sequential', action='store_true',
            help='Run validation components sequentially (default: parallel)'
        )
        execution_group.add_argument(
            '--max-workers', type=int, default=4,
            help='Maximum number of parallel workers'
        )
        execution_group.add_argument(
            '--timeout', type=int, default=30,
            help='Timeout in minutes for validation'
        )
        execution_group.add_argument(
            '--async', dest='run_async', action='store_true',
            help='Run validation asynchronously'
        )
        
        # Notification settings
        notification_group = parser.add_argument_group('Notifications')
        notification_group.add_argument(
            '--webhook', 
            help='Webhook URL for completion notifications'
        )
        notification_group.add_argument(
            '--email',
            help='Email address for completion notifications'
        )
        
        # Other options
        parser.add_argument(
            '--verbose', '-v', action='store_true',
            help='Verbose output'
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be done without executing'
        )
    
    def run(self, args: List[str] = None) -> int:
        """Run CLI with provided arguments"""
        
        parsed_args = self.parser.parse_args(args)
        
        if parsed_args.command is None:
            self.parser.print_help()
            return 1
        
        try:
            if parsed_args.command == 'validate':
                return self._handle_validate(parsed_args)
            elif parsed_args.command == 'status':
                return self._handle_status(parsed_args)
            elif parsed_args.command == 'history':
                return self._handle_history(parsed_args)
            elif parsed_args.command == 'cancel':
                return self._handle_cancel(parsed_args)
            elif parsed_args.command == 'list':
                return self._handle_list(parsed_args)
            elif parsed_args.command == 'init-config':
                return self._handle_init_config(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}")
                return 1
        
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            if parsed_args.verbose if hasattr(parsed_args, 'verbose') else False:
                import traceback
                traceback.print_exc()
            return 1
    
    def _handle_validate(self, args) -> int:
        """Handle validate command"""
        
        # Load configuration
        config = self._load_validation_config(args)
        
        if args.dry_run:
            print("Dry run mode - showing configuration:")
            self._print_config(config)
            return 0
        
        # Run validation
        print(f"Starting validation: {config.experiment_name}")
        print(f"Experiment path: {config.experiment_path}")
        print(f"Output directory: {config.output_dir}")
        print("")
        
        # Setup progress reporting
        def progress_callback(message: str, progress: float):
            print(f"[{progress:5.1f}%] {message}")
        
        try:
            if args.run_async if hasattr(args, 'run_async') else False:
                result = asyncio.run(self._run_validation_async(config, progress_callback))
            else:
                result = self._run_validation_sync(config, progress_callback)
            
            # Display results
            self._display_validation_result(result)
            
            return 0 if result.validation_status == 'completed' else 1
        
        except Exception as e:
            print(f"Validation failed: {e}")
            return 1
    
    def _load_validation_config(self, args) -> ValidationConfig:
        """Load validation configuration from args or config file"""
        
        if args.config:
            # Load from configuration file
            config_path = Path(args.config)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {args.config}")
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Override with command line arguments
            self._override_config_from_args(config_data, args)
            
            return ValidationConfig(**config_data)
        
        else:
            # Create from command line arguments
            if not args.name or not args.path:
                raise ValueError("--name and --path are required when not using --config")
            
            return create_validation_config(
                experiment_name=args.name,
                experiment_path=args.path,
                output_dir=args.output,
                run_statistical=not (args.skip_statistical if hasattr(args, 'skip_statistical') else False),
                run_code_quality=not (args.skip_code_quality if hasattr(args, 'skip_code_quality') else False),
                run_scientific_rigor=not (args.skip_scientific_rigor if hasattr(args, 'skip_scientific_rigor') else False),
                generate_reports=not (args.skip_reports if hasattr(args, 'skip_reports') else False),
                launch_dashboard=args.dashboard if hasattr(args, 'dashboard') else False,
                export_visualizations=args.export_viz if hasattr(args, 'export_viz') else False,
                parallel_execution=not (args.sequential if hasattr(args, 'sequential') else False),
                max_workers=args.max_workers if hasattr(args, 'max_workers') else 4,
                timeout_minutes=args.timeout if hasattr(args, 'timeout') else 30,
                report_formats=args.report_formats if hasattr(args, 'report_formats') else ['json', 'html'],
                visualization_formats=args.viz_formats if hasattr(args, 'viz_formats') else ['html', 'png'],
                send_notifications=bool(args.webhook or args.email) if hasattr(args, 'webhook') else False,
                notification_webhook=args.webhook if hasattr(args, 'webhook') else None,
                notification_email=args.email if hasattr(args, 'email') else None
            )
    
    def _override_config_from_args(self, config_data: Dict[str, Any], args):
        """Override configuration data with command line arguments"""
        
        # Override basic settings
        if args.name:
            config_data['experiment_name'] = args.name
        if args.path:
            config_data['experiment_path'] = args.path
        if args.output:
            config_data['output_dir'] = args.output
        
        # Override component settings
        if hasattr(args, 'skip_statistical') and args.skip_statistical:
            config_data['run_statistical'] = False
        if hasattr(args, 'skip_code_quality') and args.skip_code_quality:
            config_data['run_code_quality'] = False
        if hasattr(args, 'skip_scientific_rigor') and args.skip_scientific_rigor:
            config_data['run_scientific_rigor'] = False
        if hasattr(args, 'skip_reports') and args.skip_reports:
            config_data['generate_reports'] = False
        
        # Override advanced features
        if hasattr(args, 'dashboard') and args.dashboard:
            config_data['launch_dashboard'] = True
        if hasattr(args, 'export_viz') and args.export_viz:
            config_data['export_visualizations'] = True
        
        # Override execution settings
        if hasattr(args, 'sequential') and args.sequential:
            config_data['parallel_execution'] = False
        if hasattr(args, 'max_workers'):
            config_data['max_workers'] = args.max_workers
        if hasattr(args, 'timeout'):
            config_data['timeout_minutes'] = args.timeout
        
        # Override notification settings
        if hasattr(args, 'webhook') and args.webhook:
            config_data['notification_webhook'] = args.webhook
            config_data['send_notifications'] = True
        if hasattr(args, 'email') and args.email:
            config_data['notification_email'] = args.email
            config_data['send_notifications'] = True
    
    def _run_validation_sync(self, config: ValidationConfig, progress_callback) -> ValidationResult:
        """Run validation synchronously"""
        
        workflow = self.controller.create_validation_workflow(config)
        workflow.add_progress_callback(progress_callback)
        
        return workflow.run_validation()
    
    async def _run_validation_async(self, config: ValidationConfig, progress_callback) -> ValidationResult:
        """Run validation asynchronously"""
        
        workflow = self.controller.create_validation_workflow(config)
        workflow.add_progress_callback(progress_callback)
        
        return await workflow.run_validation_async()
    
    def _display_validation_result(self, result: ValidationResult):
        """Display validation results"""
        
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(f"Experiment: {result.experiment_name}")
        print(f"Validation ID: {result.validation_id}")
        print(f"Status: {result.validation_status.upper()}")
        print(f"Overall Score: {result.overall_score:.1f}/100 ({result.overall_grade})")
        print(f"Duration: {result.duration_seconds:.1f} seconds")
        print("")
        
        # Component scores
        if result.statistical_results:
            stat_score = result.statistical_results.get('overall_score', 0)
            print(f"Statistical Validation: {stat_score:.1f}/100")
        
        if result.code_quality_results:
            code_score = result.code_quality_results.get('overall_score', 0)
            print(f"Code Quality: {code_score:.1f}/100")
        
        if result.scientific_rigor_results:
            rigor_score = result.scientific_rigor_results.get('overall_score', 0)
            print(f"Scientific Rigor: {rigor_score:.1f}/100")
        
        print("")
        
        # Generated files
        if result.report_files:
            print("Reports Generated:")
            for format_type, file_path in result.report_files.items():
                print(f"  {format_type.upper()}: {file_path}")
        
        if result.visualization_files:
            print("Visualizations Exported:")
            for format_type, file_path in result.visualization_files.items():
                print(f"  {format_type.upper()}: {file_path}")
        
        if result.dashboard_url:
            print(f"Interactive Dashboard: {result.dashboard_url}")
        
        print("")
        
        # Errors and warnings
        if result.errors:
            print("ERRORS:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print("WARNINGS:")
            for warning in result.warnings:
                print(f"  - {warning}")
    
    def _print_config(self, config: ValidationConfig):
        """Print validation configuration"""
        
        print("Validation Configuration:")
        print("-" * 40)
        print(f"Experiment Name: {config.experiment_name}")
        print(f"Experiment Path: {config.experiment_path}")
        print(f"Output Directory: {config.output_dir}")
        print("")
        print("Components:")
        print(f"  Statistical Validation: {'Yes' if config.run_statistical else 'No'}")
        print(f"  Code Quality: {'Yes' if config.run_code_quality else 'No'}")
        print(f"  Scientific Rigor: {'Yes' if config.run_scientific_rigor else 'No'}")
        print(f"  Generate Reports: {'Yes' if config.generate_reports else 'No'}")
        print("")
        print("Advanced Features:")
        print(f"  Interactive Dashboard: {'Yes' if config.launch_dashboard else 'No'}")
        print(f"  Export Visualizations: {'Yes' if config.export_visualizations else 'No'}")
        print("")
        print("Execution:")
        print(f"  Parallel Execution: {'Yes' if config.parallel_execution else 'No'}")
        print(f"  Max Workers: {config.max_workers}")
        print(f"  Timeout: {config.timeout_minutes} minutes")
    
    def _handle_status(self, args) -> int:
        """Handle status command"""
        
        status = self.controller.get_workflow_status(args.name)
        
        if status is None:
            print(f"No validation found with name: {args.name}")
            return 1
        
        print(f"Validation Status: {args.name}")
        print("-" * 40)
        print(f"Status: {status['status'].upper()}")
        print(f"Running: {'Yes' if status['is_running'] else 'No'}")
        
        if 'overall_score' in status:
            print(f"Overall Score: {status['overall_score']:.1f} ({status['overall_grade']})")
        
        if 'start_time' in status:
            print(f"Start Time: {status['start_time']}")
        
        if status.get('errors'):
            print("Errors:")
            for error in status['errors']:
                print(f"  - {error}")
        
        if status.get('warnings'):
            print("Warnings:")
            for warning in status['warnings']:
                print(f"  - {warning}")
        
        return 0
    
    def _handle_history(self, args) -> int:
        """Handle history command"""
        
        results = self.controller.get_historical_results(args.limit)
        
        if not results:
            print("No historical validation results found")
            return 0
        
        if args.format == 'json':
            print(json.dumps(results, indent=2, default=str))
        else:
            self._display_history_table(results)
        
        return 0
    
    def _display_history_table(self, results: List[Dict[str, Any]]):
        """Display history as a table"""
        
        print(f"{'Experiment Name':<25} {'Score':<8} {'Grade':<6} {'Status':<12} {'Date':<20}")
        print("-" * 80)
        
        for result in results:
            name = result.get('experiment_name', 'Unknown')[:24]
            score = result.get('overall_score', 0)
            grade = result.get('overall_grade', 'N/A')
            status = result.get('validation_status', 'Unknown')[:11]
            
            # Format date
            start_time = result.get('start_time', '')
            if start_time:
                try:
                    dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = start_time[:19]
            else:
                date_str = 'Unknown'
            
            print(f"{name:<25} {score:<8.1f} {grade:<6} {status:<12} {date_str:<20}")
    
    def _handle_cancel(self, args) -> int:
        """Handle cancel command"""
        
        success = self.controller.cancel_workflow(args.name)
        
        if success:
            print(f"Cancelled validation: {args.name}")
            return 0
        else:
            print(f"Could not cancel validation: {args.name}")
            print("Validation may not be running or may not exist")
            return 1
    
    def _handle_list(self, args) -> int:
        """Handle list command"""
        
        active_workflows = self.controller.get_active_workflows()
        
        if not active_workflows:
            print("No active validations")
            return 0
        
        print("Active Validations:")
        print("-" * 40)
        
        for workflow_name in active_workflows:
            status = self.controller.get_workflow_status(workflow_name)
            if status:
                print(f"{workflow_name}: {status['status'].upper()}")
            else:
                print(f"{workflow_name}: UNKNOWN")
        
        return 0
    
    def _handle_init_config(self, args) -> int:
        """Handle init-config command"""
        
        config_template = {
            "experiment_name": "my_experiment",
            "experiment_path": "./experiment",
            "output_dir": "./validation_outputs/my_experiment",
            
            "run_statistical": True,
            "run_code_quality": True,
            "run_scientific_rigor": True,
            "generate_reports": True,
            "launch_dashboard": False,
            "export_visualizations": False,
            
            "statistical_config": {
                "alpha": 0.05,
                "power": 0.8,
                "effect_size_threshold": 0.3,
                "run_meta_analysis": True,
                "run_bayesian_analysis": True
            },
            
            "code_quality_config": {
                "run_ai_analysis": True,
                "security_scan_enabled": True,
                "performance_analysis_enabled": True
            },
            
            "scientific_rigor_config": {
                "check_bias": True,
                "validate_metadata": True,
                "causal_inference_validation": True
            },
            
            "report_formats": ["json", "html"],
            "visualization_formats": ["html", "png"],
            
            "parallel_execution": True,
            "max_workers": 4,
            "timeout_minutes": 30,
            "retry_attempts": 2,
            
            "send_notifications": False,
            "notification_webhook": None,
            "notification_email": None
        }
        
        output_path = Path(args.output)
        
        try:
            with open(output_path, 'w') as f:
                if args.format == 'json':
                    json.dump(config_template, f, indent=2)
                else:  # yaml
                    yaml.dump(config_template, f, default_flow_style=False, indent=2)
            
            print(f"Configuration template created: {output_path}")
            print(f"Edit the file and run: python validation_cli.py validate --config {output_path}")
            
            return 0
        
        except Exception as e:
            print(f"Error creating configuration template: {e}")
            return 1


def main():
    """Main entry point for CLI"""
    cli = ValidationCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())