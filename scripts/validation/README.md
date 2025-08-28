# Comprehensive Experiment Validation System

## Overview

The Comprehensive Experiment Validation System is a sophisticated framework designed to automatically validate experimental designs, code quality, statistical rigor, and generate comprehensive reports for research and A/B testing workflows.

## Key Features

### 🔍 Statistical Validation Suite
- **Results Consistency Checks**: Automated verification of experimental result consistency
- **Statistical Significance Testing**: Comprehensive significance testing with multiple methods
- **Effect Size Calculations**: Cohen's d, Glass's delta, and other effect size metrics
- **Power Analysis**: Statistical power validation and sample size recommendations
- **Meta-Analysis Support**: Publication bias testing and meta-analytic methods
- **Bayesian Analysis**: Bayesian statistical testing with evidence strength assessment

### 🛠️ Code Quality Assurance
- **Automated Code Review**: AI-powered code analysis and quality assessment
- **Best Practices Verification**: Python coding standards and best practices checking
- **Performance Optimization**: Bottleneck detection and optimization suggestions
- **Security Vulnerability Scanning**: Basic security issue detection
- **AI-Based Analysis**: Machine learning-powered code smell detection and complexity analysis

### 🔬 Scientific Rigor Checks
- **Methodology Compliance**: Experimental methodology validation
- **Advanced Bias Detection**: Multiple bias type detection (selection, measurement, cognitive)
- **Reproducibility Testing**: Documentation and methodology clarity assessment
- **Causal Inference Validation**: Confounding variable and causal assumption checking
- **Metadata Quality Assessment**: Comprehensive metadata validation

### 📊 Advanced Reporting & Visualization
- **Experiment Quality Scorecard**: Overall quality assessment with grades
- **Methodology Compliance Reports**: Detailed compliance analysis
- **Statistical Power Analysis Reports**: Power calculation and recommendations
- **Interactive Dashboards**: Real-time validation result exploration
- **Advanced Visualizations**: Multi-format export (HTML, PNG, SVG, PDF)
- **Executive Summaries**: High-level findings and recommendations

## Installation

### Prerequisites

```bash
# Core dependencies (required)
pip install numpy pandas matplotlib seaborn plotly

# Advanced features (optional but recommended)
pip install scipy scikit-learn dash networkx fairlearn

# Development and testing
pip install pytest pyyaml
```

### Quick Start

1. Clone or download the validation system files
2. Install dependencies
3. Run your first validation:

```python
from scripts.validation import run_comprehensive_validation

# Run validation with default settings
result = run_comprehensive_validation(
    experiment_name="my_experiment",
    experiment_path="./my_experiment_code"
)

print(f"Overall Score: {result.overall_score:.1f} ({result.overall_grade})")
```

## Usage Guide

### Method 1: Python API

#### Basic Usage

```python
from scripts.validation.unified_validation_controller import (
    UnifiedValidationController,
    create_validation_config
)

# Create validation configuration
config = create_validation_config(
    experiment_name="my_experiment",
    experiment_path="./experiment",
    output_dir="./validation_results",
    launch_dashboard=True,
    export_visualizations=True
)

# Run validation
controller = UnifiedValidationController()
result = controller.run_validation(config)

# Access results
print(f"Validation Status: {result.validation_status}")
print(f"Overall Score: {result.overall_score:.1f}")
print(f"Generated Reports: {list(result.report_files.keys())}")
```

#### Advanced Configuration

```python
from scripts.validation.unified_validation_controller import ValidationConfig

# Custom validation configuration
config = ValidationConfig(
    experiment_name="advanced_experiment",
    experiment_path="./experiment",
    output_dir="./results",
    
    # Component settings
    run_statistical=True,
    run_code_quality=True,
    run_scientific_rigor=True,
    generate_reports=True,
    
    # Advanced features
    launch_dashboard=True,
    export_visualizations=True,
    
    # Execution settings
    parallel_execution=True,
    max_workers=4,
    timeout_minutes=30,
    
    # Statistical validation config
    statistical_config={
        "alpha": 0.05,
        "power": 0.8,
        "effect_size_threshold": 0.3,
        "run_meta_analysis": True,
        "run_bayesian_analysis": True
    },
    
    # Code quality config
    code_quality_config={
        "run_ai_analysis": True,
        "security_scan_enabled": True,
        "performance_analysis_enabled": True
    },
    
    # Scientific rigor config
    scientific_rigor_config={
        "check_bias": True,
        "validate_metadata": True,
        "causal_inference_validation": True
    },
    
    # Output formats
    report_formats=["json", "html", "csv"],
    visualization_formats=["html", "png", "svg"],
    
    # Notifications
    send_notifications=True,
    notification_webhook="https://your-webhook.com/validation-complete",
    notification_email="your-email@domain.com"
)

# Run with advanced configuration
controller = UnifiedValidationController()
result = controller.run_validation(config)
```

#### Asynchronous Usage

```python
import asyncio
from scripts.validation.unified_validation_controller import UnifiedValidationController

async def run_async_validation():
    config = create_validation_config(
        experiment_name="async_experiment",
        experiment_path="./experiment"
    )
    
    controller = UnifiedValidationController()
    result = await controller.run_validation_async(config)
    
    return result

# Run asynchronously
result = asyncio.run(run_async_validation())
```

### Method 2: Command Line Interface

#### Generate Configuration Template

```bash
# Generate YAML configuration template
python -m scripts.validation.validation_cli init-config --output validation_config.yaml --format yaml

# Generate JSON configuration template
python -m scripts.validation.validation_cli init-config --output validation_config.json --format json
```

#### Run Validation from Command Line

```bash
# Basic validation
python -m scripts.validation.validation_cli validate \
    --name "my_experiment" \
    --path "./experiment" \
    --output "./results"

# Advanced validation with options
python -m scripts.validation.validation_cli validate \
    --name "advanced_experiment" \
    --path "./experiment" \
    --output "./results" \
    --dashboard \
    --export-viz \
    --report-formats json html csv \
    --viz-formats html png svg \
    --webhook "https://your-webhook.com/notify" \
    --verbose

# Run from configuration file
python -m scripts.validation.validation_cli validate --config validation_config.yaml

# Dry run (show what would be done)
python -m scripts.validation.validation_cli validate \
    --name "test_experiment" \
    --path "./experiment" \
    --dry-run
```

#### Monitor and Manage Validations

```bash
# Check status of running validation
python -m scripts.validation.validation_cli status --name "my_experiment"

# List all active validations
python -m scripts.validation.validation_cli list

# View validation history
python -m scripts.validation.validation_cli history --limit 10 --format table

# Cancel running validation
python -m scripts.validation.validation_cli cancel --name "my_experiment"
```

### Method 3: Configuration Files

#### YAML Configuration Example

```yaml
# validation_config.yaml
experiment_name: "production_experiment"
experiment_path: "./production_experiment"
output_dir: "./validation_outputs/production_experiment"

# Validation components
run_statistical: true
run_code_quality: true
run_scientific_rigor: true
generate_reports: true
launch_dashboard: false
export_visualizations: true

# Component-specific settings
statistical_config:
  alpha: 0.05
  power: 0.8
  effect_size_threshold: 0.3
  run_meta_analysis: true
  run_bayesian_analysis: true

code_quality_config:
  run_ai_analysis: true
  security_scan_enabled: true
  performance_analysis_enabled: true

scientific_rigor_config:
  check_bias: true
  validate_metadata: true
  causal_inference_validation: true

# Output settings
report_formats: ["json", "html"]
visualization_formats: ["html", "png"]

# Execution settings
parallel_execution: true
max_workers: 4
timeout_minutes: 30
retry_attempts: 2

# Notifications
send_notifications: false
notification_webhook: null
notification_email: null
```

## Component Details

### Statistical Validator

The statistical validation component provides comprehensive analysis of experimental results:

#### Features
- **Consistency Analysis**: Variance checking, trend analysis, outlier detection
- **Significance Testing**: T-tests, Mann-Whitney U tests, permutation tests
- **Effect Size Calculations**: Cohen's d, Hedges' g, Glass's delta
- **Power Analysis**: Observed power, minimum detectable effect, sample size recommendations
- **Meta-Analysis**: Publication bias testing, forest plots, heterogeneity analysis
- **Bayesian Analysis**: Bayesian t-tests, Bayes factors, credible intervals

#### Usage Example
```python
from scripts.validation.statistical_validator import StatisticalValidator

validator = StatisticalValidator()

# Your experimental data
data = {
    'results_data': [1.2, 1.5, 1.8, 2.1, 1.9, 1.7, 1.3, 1.6],
    'baseline_data': [1.0, 1.1, 1.2, 1.3, 1.1, 1.0, 1.2, 1.1],
    'experiment_params': {
        'alpha': 0.05,
        'power': 0.8,
        'effect_size_threshold': 0.3
    }
}

result = validator.validate_experiment(data)
print(f"Overall Score: {result['overall_score']}")
```

### Code Quality Validator

Automated code quality assessment with AI-powered analysis:

#### Features
- **Code Review**: Complexity analysis, maintainability assessment
- **Best Practices**: PEP 8 compliance, naming conventions, documentation
- **Performance Analysis**: Complexity assessment, bottleneck identification
- **Security Scanning**: Basic vulnerability detection
- **AI Analysis**: Machine learning-based code smell detection

#### Usage Example
```python
from scripts.validation.code_quality_validator import CodeQualityValidator

validator = CodeQualityValidator()
result = validator.validate_experiment_code("./my_experiment")

print(f"Code Quality Score: {result['overall_score']}")
print(f"Issues Found: {result['code_review']['total_issues']}")
```

### Scientific Rigor Validator

Comprehensive scientific methodology validation:

#### Features
- **Methodology Compliance**: Experimental design validation
- **Bias Detection**: Selection, measurement, confirmation, and algorithmic bias
- **Reproducibility Assessment**: Documentation completeness, methodology clarity
- **Causal Inference**: Confounding variable detection, assumption checking
- **Metadata Validation**: Data quality and completeness assessment

#### Usage Example
```python
from scripts.validation.scientific_rigor_validator import ScientificRigorValidator

validator = ScientificRigorValidator()

experiment_design = {
    'methodology': 'randomized_controlled_trial',
    'sample_size': 1000,
    'primary_metric': 'conversion_rate',
    'confounding_variables': ['age', 'gender', 'location']
}

result = validator.validate_experiment_design(experiment_design)
print(f"Scientific Rigor Score: {result['overall_score']}")
```

### Validation Reporter

Comprehensive reporting and visualization system:

#### Features
- **Experiment Scorecard**: Overall quality assessment with grades A-F
- **Executive Summaries**: High-level findings and recommendations
- **Detailed Reports**: Component-wise analysis and metrics
- **Interactive Dashboards**: Real-time exploration of results
- **Advanced Visualizations**: Multiple chart types and formats
- **Multiple Export Formats**: JSON, HTML, CSV, PNG, SVG, PDF

#### Usage Example
```python
from scripts.validation.validation_reporter import ValidationReporter

reporter = ValidationReporter("./output_dir")

# Combined validation results from all components
combined_results = {
    "statistical_validation": statistical_results,
    "code_quality": code_quality_results,
    "scientific_rigor": scientific_rigor_results
}

# Generate comprehensive report
report = reporter.generate_comprehensive_report(
    combined_results, "my_experiment"
)

# Launch interactive dashboard
dashboard_url = reporter.launch_interactive_dashboard(
    combined_results, "my_experiment"
)

# Export advanced visualizations
exported_files = reporter.export_advanced_visualizations(
    combined_results, "my_experiment", ["html", "png", "pdf"]
)
```

## Output Formats

### JSON Report Structure
```json
{
  "report_metadata": {
    "report_id": "experiment_20250127_143022",
    "experiment_name": "my_experiment",
    "generation_timestamp": "2025-01-27T14:30:22",
    "report_version": "1.0.0"
  },
  "executive_summary": {
    "summary": "Experiment validation completed with good overall quality...",
    "key_findings": ["Strong statistical significance", "Minor code quality issues"],
    "overall_assessment": "GOOD",
    "validation_status": {
      "statistical_validation": 85.0,
      "code_quality": 78.0,
      "scientific_rigor": 92.0
    }
  },
  "experiment_scorecard": {
    "overall_score": 85.0,
    "overall_grade": "B",
    "category_scores": {
      "statistical_validation": 85.0,
      "code_quality": 78.0,
      "scientific_rigor": 92.0
    },
    "detailed_metrics": [...],
    "recommendations": [...],
    "strengths": [...],
    "weaknesses": [...]
  }
}
```

### HTML Report Features
- Responsive design for all devices
- Interactive charts and visualizations
- Collapsible sections for detailed analysis
- Executive summary dashboard
- Downloadable component reports

### Interactive Dashboard Features
- Real-time data exploration
- Filter and drill-down capabilities
- Comparative analysis tools
- Export functionality
- Customizable views

## Testing

### Run Test Suite

```bash
# Run all tests
python -m scripts.validation.test_validation_system

# Run specific test class
python -m unittest scripts.validation.test_validation_system.TestStatisticalValidator

# Run with verbose output
python -m unittest scripts.validation.test_validation_system -v
```

### Test Coverage

The test suite includes:
- Unit tests for all validation components
- Integration tests for end-to-end workflows
- CLI interface testing
- Error handling and recovery testing
- Concurrent validation testing
- Mock-based testing for external dependencies

## Configuration Reference

### ValidationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | Required | Name of the experiment |
| `experiment_path` | str | Required | Path to experiment code/data |
| `output_dir` | str | "validation_outputs" | Output directory |
| `run_statistical` | bool | True | Run statistical validation |
| `run_code_quality` | bool | True | Run code quality validation |
| `run_scientific_rigor` | bool | True | Run scientific rigor validation |
| `generate_reports` | bool | True | Generate validation reports |
| `launch_dashboard` | bool | False | Launch interactive dashboard |
| `export_visualizations` | bool | False | Export advanced visualizations |
| `parallel_execution` | bool | True | Run components in parallel |
| `max_workers` | int | 4 | Maximum parallel workers |
| `timeout_minutes` | int | 30 | Validation timeout |
| `report_formats` | List[str] | ["json", "html"] | Report output formats |
| `visualization_formats` | List[str] | ["html", "png"] | Visualization formats |

## Troubleshooting

### Common Issues

#### Import Errors
```python
ImportError: No module named 'scipy'
```
**Solution**: Install optional dependencies
```bash
pip install scipy scikit-learn dash networkx fairlearn
```

#### Permission Errors
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Ensure write permissions to output directory
```bash
chmod 755 ./validation_outputs
```

#### Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce parallel workers or use sequential execution
```python
config.parallel_execution = False
config.max_workers = 2
```

#### Dashboard Not Loading
**Solution**: Check that dashboard dependencies are installed
```bash
pip install dash plotly
```

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run validation with debug output
result = controller.run_validation(config)
```

## API Reference

### Main Classes

#### `UnifiedValidationController`
Main controller for orchestrating validation workflows.

**Methods**:
- `create_validation_workflow(config: ValidationConfig) -> ValidationWorkflow`
- `run_validation(config: ValidationConfig) -> ValidationResult`
- `run_validation_async(config: ValidationConfig) -> ValidationResult`
- `get_active_workflows() -> List[str]`
- `get_workflow_status(experiment_name: str) -> Dict[str, Any]`
- `cancel_workflow(experiment_name: str) -> bool`

#### `ValidationWorkflow`
Individual validation workflow execution.

**Methods**:
- `run_validation() -> ValidationResult`
- `run_validation_async() -> ValidationResult`
- `add_progress_callback(callback: Callable) -> None`

#### `ValidationResult`
Container for validation results.

**Attributes**:
- `experiment_name: str`
- `validation_status: str`
- `overall_score: float`
- `overall_grade: str`
- `statistical_results: Dict[str, Any]`
- `code_quality_results: Dict[str, Any]`
- `scientific_rigor_results: Dict[str, Any]`
- `report_files: Dict[str, str]`
- `visualization_files: Dict[str, str]`

### Utility Functions

#### `run_comprehensive_validation(experiment_name, experiment_path, output_dir=None)`
Convenience function for quick validation with default settings.

#### `create_validation_config(experiment_name, experiment_path, **kwargs)`
Create validation configuration with sensible defaults.

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd experiment-validation-system

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 scripts/validation/
black scripts/validation/
```

### Adding New Validation Components

1. Create new validator class in `scripts/validation/`
2. Implement required interface methods
3. Add to `UnifiedValidationController`
4. Write comprehensive tests
5. Update documentation

### Extending Reporting

1. Add new report type to `ValidationReporter`
2. Implement export methods
3. Add CLI support
4. Update configuration options

## License

This project is released under the MIT License. See LICENSE file for details.

## Support

For issues, questions, or contributions:

1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Include validation logs and configuration when reporting bugs

## Version History

### v1.0.0
- Initial release with comprehensive validation framework
- Statistical, code quality, and scientific rigor validation
- Interactive dashboards and advanced visualizations
- CLI interface and configuration file support
- Comprehensive test suite and documentation