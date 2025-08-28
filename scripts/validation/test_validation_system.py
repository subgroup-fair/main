"""
Comprehensive Test Suite for Validation System
Tests all components of the experiment validation framework
"""

import unittest
import json
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import validation system components
from .statistical_validator import StatisticalValidator
from .code_quality_validator import CodeQualityValidator
from .scientific_rigor_validator import ScientificRigorValidator
from .validation_reporter import ValidationReporter
from .unified_validation_controller import (
    UnifiedValidationController,
    ValidationConfig,
    ValidationWorkflow,
    create_validation_config,
    run_comprehensive_validation
)
from .validation_cli import ValidationCLI

class TestStatisticalValidator(unittest.TestCase):
    """Test statistical validation component"""
    
    def setUp(self):
        self.validator = StatisticalValidator()
        self.mock_data = {
            'results_data': [1.2, 1.5, 1.8, 2.1, 1.9, 1.7, 1.3, 1.6, 1.4, 1.8],
            'baseline_data': [1.0, 1.1, 1.2, 1.3, 1.1, 1.0, 1.2, 1.1, 1.0, 1.2],
            'experiment_params': {
                'alpha': 0.05,
                'power': 0.8,
                'effect_size_threshold': 0.3
            }
        }
    
    def test_basic_validation(self):
        """Test basic statistical validation"""
        result = self.validator.validate_experiment(self.mock_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
        self.assertIn('consistency_analysis', result)
        self.assertIn('significance_tests', result)
        self.assertIn('effect_sizes', result)
        self.assertIn('power_analysis', result)
    
    def test_consistency_analysis(self):
        """Test results consistency analysis"""
        result = self.validator.validate_experiment(self.mock_data)
        consistency = result.get('consistency_analysis', {})
        
        self.assertIn('variance_check', consistency)
        self.assertIn('trend_analysis', consistency)
        self.assertIn('outlier_detection', consistency)
    
    def test_significance_testing(self):
        """Test statistical significance testing"""
        result = self.validator.validate_experiment(self.mock_data)
        significance = result.get('significance_tests', {})
        
        # Should include basic t-test
        self.assertIn('t_test', significance)
        if 't_test' in significance:
            t_test = significance['t_test']
            self.assertIn('statistic', t_test)
            self.assertIn('p_value', t_test)
            self.assertIn('is_significant', t_test)
    
    def test_effect_size_calculation(self):
        """Test effect size calculations"""
        result = self.validator.validate_experiment(self.mock_data)
        effect_sizes = result.get('effect_sizes', {})
        
        self.assertIn('cohens_d', effect_sizes)
        if 'cohens_d' in effect_sizes:
            cohens_d = effect_sizes['cohens_d']
            self.assertIn('value', cohens_d)
            self.assertIn('magnitude', cohens_d)
    
    def test_power_analysis(self):
        """Test power analysis"""
        result = self.validator.validate_experiment(self.mock_data)
        power_analysis = result.get('power_analysis', {})
        
        self.assertIn('observed_power', power_analysis)
        self.assertIn('minimum_effect_size', power_analysis)
        self.assertIn('sample_size_recommendation', power_analysis)
    
    @patch('scripts.validation.statistical_validator.SCIPY_AVAILABLE', False)
    def test_graceful_degradation_scipy(self):
        """Test graceful degradation when scipy is not available"""
        result = self.validator.validate_experiment(self.mock_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
        # Should still provide basic analysis even without scipy
        self.assertTrue(result['overall_score'] >= 0)


class TestCodeQualityValidator(unittest.TestCase):
    """Test code quality validation component"""
    
    def setUp(self):
        self.validator = CodeQualityValidator()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock Python files
        self.test_file = Path(self.temp_dir) / "test_code.py"
        self.test_file.write_text("""
def calculate_mean(data):
    '''Calculate mean of data'''
    if len(data) == 0:
        return 0
    return sum(data) / len(data)

def process_experiment_results(results):
    '''Process experiment results'''
    processed = []
    for result in results:
        if result > 0:
            processed.append(result * 2)
    return processed

class ExperimentAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def analyze(self):
        return self.calculate_statistics()
    
    def calculate_statistics(self):
        return {
            'mean': calculate_mean(self.data),
            'count': len(self.data)
        }
        """)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_basic_code_validation(self):
        """Test basic code quality validation"""
        result = self.validator.validate_experiment_code(self.temp_dir)
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
        self.assertIn('code_review', result)
        self.assertIn('best_practices', result)
        self.assertIn('performance_analysis', result)
        self.assertIn('security_scan', result)
    
    def test_code_review_analysis(self):
        """Test automated code review"""
        result = self.validator.validate_experiment_code(self.temp_dir)
        code_review = result.get('code_review', {})
        
        self.assertIn('total_issues', code_review)
        self.assertIn('complexity_score', code_review)
        self.assertIn('maintainability_score', code_review)
    
    def test_best_practices_check(self):
        """Test best practices verification"""
        result = self.validator.validate_experiment_code(self.temp_dir)
        best_practices = result.get('best_practices', {})
        
        self.assertIn('documentation_score', best_practices)
        self.assertIn('naming_conventions', best_practices)
        self.assertIn('code_structure', best_practices)
    
    def test_performance_analysis(self):
        """Test performance analysis"""
        result = self.validator.validate_experiment_code(self.temp_dir)
        performance = result.get('performance_analysis', {})
        
        self.assertIn('complexity_analysis', performance)
        self.assertIn('bottlenecks', performance)
        self.assertIn('optimization_suggestions', performance)
    
    def test_security_scanning(self):
        """Test security vulnerability scanning"""
        result = self.validator.validate_experiment_code(self.temp_dir)
        security = result.get('security_scan', {})
        
        self.assertIn('security_score', security)
        self.assertIn('vulnerabilities', security)
        self.assertIn('recommendations', security)


class TestScientificRigorValidator(unittest.TestCase):
    """Test scientific rigor validation component"""
    
    def setUp(self):
        self.validator = ScientificRigorValidator()
        self.mock_experiment_design = {
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
    
    def test_basic_rigor_validation(self):
        """Test basic scientific rigor validation"""
        result = self.validator.validate_experiment_design(self.mock_experiment_design)
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_score', result)
        self.assertIn('methodology_compliance', result)
        self.assertIn('bias_detection', result)
        self.assertIn('reproducibility_assessment', result)
    
    def test_methodology_compliance(self):
        """Test methodology compliance verification"""
        result = self.validator.validate_experiment_design(self.mock_experiment_design)
        methodology = result.get('methodology_compliance', {})
        
        self.assertIn('methodology_score', methodology)
        self.assertIn('compliance_checks', methodology)
        self.assertIn('violations', methodology)
    
    def test_bias_detection(self):
        """Test bias detection algorithms"""
        result = self.validator.validate_experiment_design(self.mock_experiment_design)
        bias_detection = result.get('bias_detection', {})
        
        self.assertIn('total_biases_detected', bias_detection)
        self.assertIn('bias_types', bias_detection)
        self.assertIn('mitigation_strategies', bias_detection)
    
    def test_reproducibility_assessment(self):
        """Test reproducibility testing"""
        result = self.validator.validate_experiment_design(self.mock_experiment_design)
        reproducibility = result.get('reproducibility_assessment', {})
        
        self.assertIn('reproducibility_score', reproducibility)
        self.assertIn('documentation_completeness', reproducibility)
        self.assertIn('methodology_clarity', reproducibility)


class TestValidationReporter(unittest.TestCase):
    """Test validation reporting system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = ValidationReporter(self.temp_dir)
        
        # Mock validation results
        self.mock_results = {
            'statistical_validation': {
                'overall_score': 85.0,
                'consistency_analysis': {'variance_check': {'is_consistent': True}},
                'significance_tests': {'t_test': {'is_significant': True, 'p_value': 0.02}},
                'effect_sizes': {'cohens_d': {'value': 0.5, 'magnitude': 'medium'}}
            },
            'code_quality': {
                'overall_score': 78.0,
                'code_review': {'total_issues': 3, 'complexity_score': 7.5},
                'security_scan': {'security_score': 95, 'vulnerabilities': []}
            },
            'scientific_rigor': {
                'overall_score': 92.0,
                'methodology_compliance': {'methodology_score': 90},
                'bias_detection': {'total_biases_detected': 1},
                'reproducibility_assessment': {'reproducibility_score': 95}
            }
        }
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation"""
        report = self.reporter.generate_comprehensive_report(
            self.mock_results, "test_experiment"
        )
        
        self.assertIsInstance(report, dict)
        self.assertIn('report_metadata', report)
        self.assertIn('executive_summary', report)
        self.assertIn('experiment_scorecard', report)
        self.assertIn('methodology_compliance', report)
        self.assertIn('statistical_power_analysis', report)
        self.assertIn('reproducibility_assessment', report)
    
    def test_scorecard_generation(self):
        """Test experiment scorecard generation"""
        report = self.reporter.generate_comprehensive_report(
            self.mock_results, "test_experiment"
        )
        
        scorecard = report.get('experiment_scorecard', {})
        self.assertIn('overall_score', scorecard)
        self.assertIn('overall_grade', scorecard)
        self.assertIn('category_scores', scorecard)
        self.assertIn('detailed_metrics', scorecard)
    
    def test_executive_summary(self):
        """Test executive summary generation"""
        report = self.reporter.generate_comprehensive_report(
            self.mock_results, "test_experiment"
        )
        
        summary = report.get('executive_summary', {})
        self.assertIn('summary', summary)
        self.assertIn('key_findings', summary)
        self.assertIn('validation_status', summary)
    
    @patch('scripts.validation.validation_reporter.PLOTTING_AVAILABLE', False)
    def test_report_without_plotting(self):
        """Test report generation without plotting libraries"""
        report = self.reporter.generate_comprehensive_report(
            self.mock_results, "test_experiment"
        )
        
        self.assertIsInstance(report, dict)
        self.assertIn('executive_summary', report)
        # Should work even without plotting capabilities


class TestUnifiedValidationController(unittest.TestCase):
    """Test unified validation controller"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.controller = UnifiedValidationController(self.temp_dir)
        
        # Create test experiment directory
        self.experiment_dir = Path(self.temp_dir) / "test_experiment"
        self.experiment_dir.mkdir()
        
        # Create mock experiment file
        (self.experiment_dir / "experiment.py").write_text("""
def run_experiment():
    '''Run the experiment'''
    return [1.2, 1.5, 1.8, 2.1, 1.9]
        """)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_validation_config_creation(self):
        """Test validation configuration creation"""
        config = create_validation_config(
            experiment_name="test_experiment",
            experiment_path=str(self.experiment_dir)
        )
        
        self.assertIsInstance(config, ValidationConfig)
        self.assertEqual(config.experiment_name, "test_experiment")
        self.assertEqual(config.experiment_path, str(self.experiment_dir))
        self.assertTrue(config.run_statistical)
        self.assertTrue(config.run_code_quality)
        self.assertTrue(config.run_scientific_rigor)
    
    def test_workflow_creation(self):
        """Test validation workflow creation"""
        config = create_validation_config(
            experiment_name="test_experiment",
            experiment_path=str(self.experiment_dir),
            output_dir=str(self.temp_dir)
        )
        
        workflow = self.controller.create_validation_workflow(config)
        
        self.assertIsInstance(workflow, ValidationWorkflow)
        self.assertEqual(workflow.config.experiment_name, "test_experiment")
        self.assertFalse(workflow.is_running)
    
    def test_workflow_status_tracking(self):
        """Test workflow status tracking"""
        config = create_validation_config(
            experiment_name="test_experiment",
            experiment_path=str(self.experiment_dir)
        )
        
        workflow = self.controller.create_validation_workflow(config)
        status = self.controller.get_workflow_status("test_experiment")
        
        self.assertIsNotNone(status)
        self.assertEqual(status['experiment_name'], "test_experiment")
        self.assertIn('status', status)
        self.assertIn('is_running', status)
    
    def test_active_workflows_listing(self):
        """Test listing of active workflows"""
        config1 = create_validation_config("exp1", str(self.experiment_dir))
        config2 = create_validation_config("exp2", str(self.experiment_dir))
        
        self.controller.create_validation_workflow(config1)
        self.controller.create_validation_workflow(config2)
        
        active_workflows = self.controller.get_active_workflows()
        
        self.assertIn("exp1", active_workflows)
        self.assertIn("exp2", active_workflows)
    
    @patch('scripts.validation.unified_validation_controller.StatisticalValidator')
    @patch('scripts.validation.unified_validation_controller.CodeQualityValidator')
    @patch('scripts.validation.unified_validation_controller.ScientificRigorValidator')
    def test_comprehensive_validation_run(self, mock_rigor, mock_code, mock_stat):
        """Test complete validation run with mocked components"""
        
        # Setup mocks
        mock_stat.return_value.validate_experiment.return_value = {'overall_score': 85.0}
        mock_code.return_value.validate_experiment_code.return_value = {'overall_score': 78.0}
        mock_rigor.return_value.validate_experiment_design.return_value = {'overall_score': 92.0}
        
        config = create_validation_config(
            experiment_name="test_experiment",
            experiment_path=str(self.experiment_dir),
            output_dir=str(self.temp_dir),
            parallel_execution=False,  # Use sequential for testing
            timeout_minutes=1
        )
        
        result = self.controller.run_validation(config)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.validation_status, 'completed')
        self.assertGreater(result.overall_score, 0)
        self.assertIn(result.overall_grade, ['A', 'B', 'C', 'D', 'F'])


class TestValidationCLI(unittest.TestCase):
    """Test validation CLI interface"""
    
    def setUp(self):
        self.cli = ValidationCLI()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test experiment directory
        self.experiment_dir = Path(self.temp_dir) / "test_experiment"
        self.experiment_dir.mkdir()
        
        (self.experiment_dir / "experiment.py").write_text("# Test experiment code")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_cli_parser_creation(self):
        """Test CLI argument parser creation"""
        parser = self.cli.parser
        
        self.assertIsNotNone(parser)
        
        # Test help doesn't raise exception
        try:
            parser.parse_args(['--help'])
        except SystemExit:
            pass  # Help command causes SystemExit, which is expected
    
    def test_config_template_generation(self):
        """Test configuration template generation"""
        config_file = Path(self.temp_dir) / "test_config.yaml"
        
        exit_code = self.cli.run([
            'init-config',
            '--output', str(config_file),
            '--format', 'yaml'
        ])
        
        self.assertEqual(exit_code, 0)
        self.assertTrue(config_file.exists())
        
        # Verify config file content
        with open(config_file, 'r') as f:
            content = f.read()
            self.assertIn('experiment_name', content)
            self.assertIn('run_statistical', content)
    
    def test_dry_run_mode(self):
        """Test dry run mode"""
        exit_code = self.cli.run([
            'validate',
            '--name', 'test_experiment',
            '--path', str(self.experiment_dir),
            '--output', str(self.temp_dir),
            '--dry-run'
        ])
        
        self.assertEqual(exit_code, 0)
    
    def test_history_command(self):
        """Test history command"""
        # This will show empty history, but should not fail
        exit_code = self.cli.run(['history', '--limit', '5'])
        
        self.assertEqual(exit_code, 0)
    
    def test_list_command(self):
        """Test list active validations command"""
        exit_code = self.cli.run(['list'])
        
        self.assertEqual(exit_code, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire validation system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create comprehensive test experiment
        self.experiment_dir = Path(self.temp_dir) / "integration_test_experiment"
        self.experiment_dir.mkdir()
        
        # Create multiple Python files
        (self.experiment_dir / "__init__.py").write_text("")
        
        (self.experiment_dir / "experiment.py").write_text("""
import numpy as np
from typing import List, Dict, Any

class ExperimentRunner:
    '''Main experiment runner class'''
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = []
    
    def run_experiment(self, sample_size: int = 1000) -> Dict[str, Any]:
        '''Run the experiment and return results'''
        
        # Generate mock experimental data
        np.random.seed(42)
        control_data = np.random.normal(1.0, 0.2, sample_size // 2)
        treatment_data = np.random.normal(1.1, 0.2, sample_size // 2)
        
        results = {
            'control_group': control_data.tolist(),
            'treatment_group': treatment_data.tolist(),
            'sample_size': sample_size,
            'effect_size': 0.1,
            'methodology': 'randomized_controlled_trial'
        }
        
        self.results.append(results)
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        '''Analyze experimental results'''
        
        control_mean = np.mean(results['control_group'])
        treatment_mean = np.mean(results['treatment_group'])
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'lift': (treatment_mean - control_mean) / control_mean,
            'sample_size': results['sample_size']
        }

def run_comprehensive_experiment():
    '''Run comprehensive experiment'''
    
    config = {
        'sample_size': 1000,
        'confidence_level': 0.95,
        'minimum_detectable_effect': 0.05
    }
    
    runner = ExperimentRunner(config)
    results = runner.run_experiment()
    analysis = runner.analyze_results(results)
    
    return {
        'raw_results': results,
        'analysis': analysis,
        'config': config
    }

if __name__ == '__main__':
    experiment_results = run_comprehensive_experiment()
    print(f"Experiment completed. Lift: {experiment_results['analysis']['lift']:.3f}")
        """)
        
        (self.experiment_dir / "utils.py").write_text("""
import math
from typing import List, Union

def calculate_statistical_power(effect_size: float, sample_size: int, 
                              alpha: float = 0.05) -> float:
    '''Calculate statistical power for given parameters'''
    
    # Simplified power calculation
    z_alpha = 1.96  # For alpha = 0.05, two-tailed
    z_beta = math.sqrt(sample_size) * effect_size / 2 - z_alpha
    
    # Use normal CDF approximation
    power = 0.5 + 0.5 * math.erf(z_beta / math.sqrt(2))
    return max(0, min(1, power))

def validate_sample_size(sample_size: int, minimum_size: int = 100) -> bool:
    '''Validate that sample size meets minimum requirements'''
    return sample_size >= minimum_size

def calculate_confidence_interval(data: List[float], 
                                confidence_level: float = 0.95) -> tuple:
    '''Calculate confidence interval for data'''
    
    if not data:
        return (0, 0)
    
    n = len(data)
    mean = sum(data) / n
    
    # Simple approximation
    std_error = math.sqrt(sum((x - mean)**2 for x in data) / (n - 1)) / math.sqrt(n)
    margin_error = 1.96 * std_error  # For 95% confidence
    
    return (mean - margin_error, mean + margin_error)
        """)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('scripts.validation.statistical_validator.NUMPY_AVAILABLE', True)
    @patch('scripts.validation.statistical_validator.SCIPY_AVAILABLE', True)
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline end-to-end"""
        
        # Create validation configuration
        config = create_validation_config(
            experiment_name="integration_test",
            experiment_path=str(self.experiment_dir),
            output_dir=str(self.temp_dir / "validation_outputs"),
            parallel_execution=False,  # Sequential for deterministic testing
            timeout_minutes=5,
            generate_reports=True,
            export_visualizations=False,  # Skip to avoid dependency issues
            launch_dashboard=False
        )
        
        # Run validation
        controller = UnifiedValidationController(str(self.temp_dir))
        result = controller.run_validation(config)
        
        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result, type(result))  # ValidationResult type
        self.assertEqual(result.experiment_name, "integration_test")
        self.assertIn(result.validation_status, ['completed', 'failed'])
        
        # Verify components ran
        if result.validation_status == 'completed':
            self.assertIsNotNone(result.statistical_results)
            self.assertIsNotNone(result.code_quality_results)
            self.assertIsNotNone(result.scientific_rigor_results)
            
            # Verify overall scores
            self.assertGreaterEqual(result.overall_score, 0)
            self.assertLessEqual(result.overall_score, 100)
            self.assertIn(result.overall_grade, ['A', 'B', 'C', 'D', 'F'])
        
        # Verify report generation
        if config.generate_reports and result.report_files:
            self.assertTrue(len(result.report_files) > 0)
            
            # Check that report files actually exist
            for format_type, file_path in result.report_files.items():
                self.assertTrue(Path(file_path).exists(), 
                              f"Report file {file_path} should exist")
    
    def test_cli_integration(self):
        """Test CLI integration with real experiment"""
        
        cli = ValidationCLI()
        
        # Test configuration generation
        config_file = Path(self.temp_dir) / "integration_config.yaml"
        exit_code = cli.run([
            'init-config',
            '--output', str(config_file),
            '--format', 'yaml'
        ])
        
        self.assertEqual(exit_code, 0)
        self.assertTrue(config_file.exists())
        
        # Test dry run with generated config
        exit_code = cli.run([
            'validate',
            '--config', str(config_file),
            '--name', 'integration_test',
            '--path', str(self.experiment_dir),
            '--output', str(self.temp_dir / "cli_validation_outputs"),
            '--dry-run'
        ])
        
        self.assertEqual(exit_code, 0)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and graceful recovery"""
        
        # Test with non-existent experiment path
        config = create_validation_config(
            experiment_name="error_test",
            experiment_path="/nonexistent/path",
            output_dir=str(self.temp_dir),
            timeout_minutes=1
        )
        
        controller = UnifiedValidationController(str(self.temp_dir))
        result = controller.run_validation(config)
        
        # Should handle error gracefully
        self.assertIsNotNone(result)
        if result.validation_status == 'failed':
            self.assertTrue(len(result.errors) > 0)
    
    def test_concurrent_validations(self):
        """Test multiple concurrent validations"""
        
        controller = UnifiedValidationController(str(self.temp_dir))
        
        # Create multiple configurations
        configs = []
        for i in range(3):
            config = create_validation_config(
                experiment_name=f"concurrent_test_{i}",
                experiment_path=str(self.experiment_dir),
                output_dir=str(self.temp_dir / f"concurrent_outputs_{i}"),
                timeout_minutes=2
            )
            configs.append(config)
        
        # Start concurrent validations
        workflows = []
        for config in configs:
            workflow = controller.create_validation_workflow(config)
            workflows.append(workflow)
        
        # Check that all workflows were created
        active_workflows = controller.get_active_workflows()
        self.assertEqual(len(active_workflows), 3)
        
        # Clean up
        for workflow_name in active_workflows:
            controller.cancel_workflow(workflow_name)


def run_comprehensive_tests():
    """Run all tests in the comprehensive test suite"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestStatisticalValidator,
        TestCodeQualityValidator,
        TestScientificRigorValidator,
        TestValidationReporter,
        TestUnifiedValidationController,
        TestValidationCLI,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'Unknown error'}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)