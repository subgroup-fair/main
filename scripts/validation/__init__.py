"""
Comprehensive Experiment Validation System

A complete validation framework for research experiments that provides:
- Statistical validation with consistency checks and significance testing  
- Code quality assurance with automated reviews and security scanning
- Scientific rigor checks with methodology compliance and bias detection
- Comprehensive validation reports with scorecards and assessments

Quick Start:
    from scripts.validation import ExperimentValidator
    
    # Create validator
    validator = ExperimentValidator("my_experiment")
    
    # Run comprehensive validation
    results = validator.validate_experiment(
        experiment_data=my_data,
        experiment_code=my_code_path,
        methodology_config=my_config
    )
    
    # Generate reports
    validator.generate_validation_report(results)

Components:
- StatisticalValidator: Results consistency, significance testing, effect sizes, power analysis
- CodeQualityValidator: Automated reviews, best practices, performance, security
- ScientificRigorValidator: Methodology compliance, bias detection, reproducibility
- ValidationReporter: Quality scorecards, compliance reports, assessments

Advanced Usage:
    from scripts.validation import (
        StatisticalValidator, 
        CodeQualityValidator,
        ScientificRigorValidator,
        ValidationReporter
    )
    
    # Individual validators
    stat_validator = StatisticalValidator()
    code_validator = CodeQualityValidator()
    rigor_validator = ScientificRigorValidator()
    
    # Custom validation pipeline
    validation_results = {}
    validation_results['statistical'] = stat_validator.validate_results(data)
    validation_results['code_quality'] = code_validator.validate_code(code_path)
    validation_results['scientific_rigor'] = rigor_validator.validate_methodology(config)
    
    # Generate comprehensive report
    reporter = ValidationReporter()
    report = reporter.generate_comprehensive_report(validation_results)
"""

from .statistical_validator import (
    StatisticalValidator,
    ResultsConsistencyChecker,
    StatisticalSignificanceTester,
    EffectSizeCalculator,
    PowerAnalysisValidator
)

from .code_quality_validator import (
    CodeQualityValidator,
    AutomaticCodeReviewer,
    BestPracticesVerifier,
    PerformanceOptimizer,
    SecurityScanner
)

from .scientific_rigor_validator import (
    ScientificRigorValidator,
    MethodologyComplianceChecker,
    BiasDetector,
    ReproducibilityTester,
    PeerReviewPreparer
)

from .validation_reporter import (
    ValidationReporter,
    ExperimentScorecard,
    ComplianceReporter,
    PowerAnalysisReporter,
    ReproducibilityAssessor
)

from .experiment_validator import (
    ExperimentValidator,
    ValidationConfig,
    ValidationResults,
    validate_experiment
)

__version__ = "1.0.0"
__author__ = "Experiment Validation System"

# Main components for easy import
__all__ = [
    # Core validators
    "StatisticalValidator",
    "CodeQualityValidator", 
    "ScientificRigorValidator",
    "ValidationReporter",
    
    # Integrated system
    "ExperimentValidator",
    "ValidationConfig",
    "ValidationResults",
    "validate_experiment",
    
    # Statistical validation components
    "ResultsConsistencyChecker",
    "StatisticalSignificanceTester", 
    "EffectSizeCalculator",
    "PowerAnalysisValidator",
    
    # Code quality components
    "AutomaticCodeReviewer",
    "BestPracticesVerifier",
    "PerformanceOptimizer",
    "SecurityScanner",
    
    # Scientific rigor components
    "MethodologyComplianceChecker",
    "BiasDetector",
    "ReproducibilityTester",
    "PeerReviewPreparer",
    
    # Reporting components
    "ExperimentScorecard",
    "ComplianceReporter",
    "PowerAnalysisReporter",
    "ReproducibilityAssessor"
]