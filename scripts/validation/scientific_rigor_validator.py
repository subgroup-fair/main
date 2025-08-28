"""
Scientific Rigor Validation System
Comprehensive validation of research methodology and scientific practices
"""

import numpy as np
import pandas as pd
import json
import re
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import defaultdict, Counter
import warnings

# Statistical libraries
try:
    from scipy import stats
    import sklearn.metrics as metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_SKLEARN_AVAILABLE = False

# Network analysis for metadata relationships
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Advanced bias detection libraries
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.reductions import ExponentiatedGradient
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

@dataclass
class BiasDetection:
    """Bias detection result"""
    bias_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    evidence: Dict[str, Any]
    confidence: float
    recommendations: List[str]
    affected_components: List[str] = None

@dataclass
class MethodologyViolation:
    """Methodology compliance violation"""
    violation_type: str
    severity: str
    requirement: str
    current_state: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: str = ""

@dataclass
class ReproducibilityIssue:
    """Reproducibility issue"""
    issue_type: str
    severity: str
    description: str
    affected_files: List[str]
    fix_suggestions: List[str]
    reproducibility_score_impact: float

@dataclass
class PeerReviewItem:
    """Peer review checklist item"""
    category: str
    item_name: str
    status: str  # 'pass', 'fail', 'needs_attention', 'not_applicable'
    details: str
    suggestions: List[str] = None
    priority: str = 'medium'

@dataclass
class AdvancedBiasDetection:
    """Advanced bias detection result"""
    bias_category: str
    bias_subcategory: str
    statistical_evidence: Dict[str, Any]
    demographic_analysis: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    severity_score: float
    impact_assessment: str
    mitigation_strategies: List[str]

@dataclass
class MetadataValidation:
    """Metadata validation result"""
    completeness_score: float
    consistency_score: float
    quality_issues: List[str]
    missing_fields: List[str]
    inconsistent_fields: List[str]
    recommendations: List[str]
    metadata_graph_analysis: Optional[Dict[str, Any]] = None

@dataclass
class CausalInferenceValidation:
    """Causal inference validation result"""
    confounding_variables: List[str]
    mediation_analysis: Dict[str, Any]
    instrumental_variables: List[str]
    causal_assumptions: Dict[str, bool]
    validity_threats: List[str]
    causal_graph_issues: List[str]

class MethodologyComplianceChecker:
    """Check compliance with research methodology standards"""
    
    def __init__(self, methodology_standards: Dict[str, Any] = None):
        self.logger = logging.getLogger("methodology_checker")
        self.standards = methodology_standards or self._load_default_standards()
    
    def check_compliance(self, experiment_config: Dict[str, Any], 
                        code_files: List[str] = None,
                        data_files: List[str] = None) -> Dict[str, Any]:
        """Comprehensive methodology compliance check"""
        
        compliance_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_compliance_score": 0,
            "compliance_level": "unknown",
            "violations": [],
            "compliant_practices": [],
            "recommendations": []
        }
        
        all_violations = []
        compliant_practices = []
        
        # Check experimental design compliance
        design_violations = self._check_experimental_design(experiment_config)
        all_violations.extend(design_violations)
        
        # Check statistical methodology
        stats_violations = self._check_statistical_methodology(experiment_config)
        all_violations.extend(stats_violations)
        
        # Check data handling practices
        if data_files:
            data_violations = self._check_data_handling(experiment_config, data_files)
            all_violations.extend(data_violations)
        
        # Check code methodology
        if code_files:
            code_violations = self._check_code_methodology(experiment_config, code_files)
            all_violations.extend(code_violations)
        
        # Check reporting standards
        reporting_violations = self._check_reporting_standards(experiment_config)
        all_violations.extend(reporting_violations)
        
        # Calculate compliance score
        max_score = 100
        severity_penalties = {"critical": 25, "high": 15, "medium": 8, "low": 3}
        
        total_penalty = sum(severity_penalties.get(v.severity, 5) for v in all_violations)
        compliance_score = max(0, max_score - total_penalty)
        
        # Determine compliance level
        if compliance_score >= 90:
            compliance_level = "excellent"
        elif compliance_score >= 80:
            compliance_level = "good"
        elif compliance_score >= 70:
            compliance_level = "acceptable"
        elif compliance_score >= 60:
            compliance_level = "poor"
        else:
            compliance_level = "inadequate"
        
        compliance_results.update({
            "overall_compliance_score": compliance_score,
            "compliance_level": compliance_level,
            "violations": [self._serialize_violation(v) for v in all_violations],
            "recommendations": self._generate_compliance_recommendations(all_violations)
        })
        
        return compliance_results
    
    def _load_default_standards(self) -> Dict[str, Any]:
        """Load default research methodology standards"""
        
        return {
            "experimental_design": {
                "requires_hypothesis": True,
                "requires_control_group": True,
                "min_sample_size": 30,
                "requires_randomization": True,
                "requires_power_analysis": True
            },
            "statistical_methodology": {
                "requires_significance_test": True,
                "alpha_level": 0.05,
                "requires_effect_size": True,
                "requires_confidence_intervals": True,
                "multiple_comparison_correction": True
            },
            "data_handling": {
                "requires_data_validation": True,
                "missing_data_threshold": 0.1,
                "outlier_detection_required": True,
                "data_preprocessing_documented": True
            },
            "reproducibility": {
                "random_seed_required": True,
                "version_control_required": True,
                "dependency_management": True,
                "computational_environment_documented": True
            },
            "reporting": {
                "methodology_section_required": True,
                "results_section_required": True,
                "limitations_discussed": True,
                "data_availability_statement": True
            }
        }
    
    def _check_experimental_design(self, config: Dict[str, Any]) -> List[MethodologyViolation]:
        """Check experimental design compliance"""
        
        violations = []
        design_standards = self.standards["experimental_design"]
        
        # Check for hypothesis
        if design_standards.get("requires_hypothesis", True):
            if "hypothesis" not in config or not config["hypothesis"]:
                violations.append(MethodologyViolation(
                    violation_type="missing_hypothesis",
                    severity="high",
                    requirement="Experiment must have clearly stated hypothesis",
                    current_state="No hypothesis found in configuration",
                    recommendation="Add hypothesis statement to experimental design"
                ))
        
        # Check for control group
        if design_standards.get("requires_control_group", True):
            if "control_group" not in config or not config.get("control_group", False):
                violations.append(MethodologyViolation(
                    violation_type="missing_control_group",
                    severity="high",
                    requirement="Experiment should include control group",
                    current_state="No control group specified",
                    recommendation="Include appropriate control group in experimental design"
                ))
        
        # Check sample size
        min_sample_size = design_standards.get("min_sample_size", 30)
        if "sample_size" in config:
            if config["sample_size"] < min_sample_size:
                violations.append(MethodologyViolation(
                    violation_type="insufficient_sample_size",
                    severity="medium",
                    requirement=f"Minimum sample size of {min_sample_size}",
                    current_state=f"Current sample size: {config['sample_size']}",
                    recommendation="Increase sample size or provide power analysis justification"
                ))
        
        # Check randomization
        if design_standards.get("requires_randomization", True):
            if "randomization" not in config or not config.get("randomization", False):
                violations.append(MethodologyViolation(
                    violation_type="missing_randomization",
                    severity="medium",
                    requirement="Randomization should be implemented",
                    current_state="No randomization specified",
                    recommendation="Implement and document randomization procedure"
                ))
        
        return violations
    
    def _check_statistical_methodology(self, config: Dict[str, Any]) -> List[MethodologyViolation]:
        """Check statistical methodology compliance"""
        
        violations = []
        stats_standards = self.standards["statistical_methodology"]
        
        # Check for significance testing
        if stats_standards.get("requires_significance_test", True):
            if "statistical_test" not in config:
                violations.append(MethodologyViolation(
                    violation_type="missing_statistical_test",
                    severity="high",
                    requirement="Statistical significance test required",
                    current_state="No statistical test specified",
                    recommendation="Specify appropriate statistical test for your data"
                ))
        
        # Check alpha level
        if "alpha" in config:
            expected_alpha = stats_standards.get("alpha_level", 0.05)
            if config["alpha"] != expected_alpha:
                violations.append(MethodologyViolation(
                    violation_type="non_standard_alpha",
                    severity="low",
                    requirement=f"Standard alpha level of {expected_alpha}",
                    current_state=f"Current alpha: {config['alpha']}",
                    recommendation="Justify non-standard alpha level or use standard value"
                ))
        
        # Check for effect size reporting
        if stats_standards.get("requires_effect_size", True):
            if "effect_size" not in config or not config.get("effect_size", False):
                violations.append(MethodologyViolation(
                    violation_type="missing_effect_size",
                    severity="medium",
                    requirement="Effect size calculation and reporting required",
                    current_state="No effect size calculation specified",
                    recommendation="Include appropriate effect size measures"
                ))
        
        # Check multiple comparison correction
        if config.get("multiple_comparisons", 0) > 1:
            if not config.get("multiple_comparison_correction", False):
                violations.append(MethodologyViolation(
                    violation_type="missing_multiple_comparison_correction",
                    severity="high",
                    requirement="Multiple comparison correction required",
                    current_state="Multiple comparisons without correction",
                    recommendation="Apply Bonferroni, FDR, or other appropriate correction"
                ))
        
        return violations
    
    def _check_data_handling(self, config: Dict[str, Any], data_files: List[str]) -> List[MethodologyViolation]:
        """Check data handling practices"""
        
        violations = []
        data_standards = self.standards["data_handling"]
        
        # Check for data validation
        if data_standards.get("requires_data_validation", True):
            if "data_validation" not in config or not config.get("data_validation", False):
                violations.append(MethodologyViolation(
                    violation_type="missing_data_validation",
                    severity="medium",
                    requirement="Data validation procedures required",
                    current_state="No data validation specified",
                    recommendation="Implement data quality checks and validation"
                ))
        
        # Check missing data handling
        if "missing_data_percentage" in config:
            threshold = data_standards.get("missing_data_threshold", 0.1)
            if config["missing_data_percentage"] > threshold:
                violations.append(MethodologyViolation(
                    violation_type="high_missing_data",
                    severity="medium",
                    requirement=f"Missing data should be less than {threshold*100}%",
                    current_state=f"Missing data: {config['missing_data_percentage']*100:.1f}%",
                    recommendation="Address missing data through imputation or justify exclusion"
                ))
        
        # Check outlier detection
        if data_standards.get("outlier_detection_required", True):
            if "outlier_detection" not in config or not config.get("outlier_detection", False):
                violations.append(MethodologyViolation(
                    violation_type="missing_outlier_detection",
                    severity="low",
                    requirement="Outlier detection and handling required",
                    current_state="No outlier detection specified",
                    recommendation="Implement outlier detection and handling procedures"
                ))
        
        return violations
    
    def _check_code_methodology(self, config: Dict[str, Any], code_files: List[str]) -> List[MethodologyViolation]:
        """Check code methodology practices"""
        
        violations = []
        
        # Check for random seed setting
        has_random_seed = False
        for file_path in code_files:
            if self._check_random_seed_in_file(file_path):
                has_random_seed = True
                break
        
        if not has_random_seed:
            violations.append(MethodologyViolation(
                violation_type="missing_random_seed",
                severity="high",
                requirement="Random seed should be set for reproducibility",
                current_state="No random seed found in code",
                recommendation="Set random seeds for all random number generators"
            ))
        
        # Check for version control
        if not self._check_version_control():
            violations.append(MethodologyViolation(
                violation_type="missing_version_control",
                severity="medium",
                requirement="Code should be under version control",
                current_state="No version control detected",
                recommendation="Initialize git repository and commit code"
            ))
        
        return violations
    
    def _check_random_seed_in_file(self, file_path: str) -> bool:
        """Check if random seed is set in a Python file"""
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for random seed patterns
            seed_patterns = [
                r'random\.seed\(',
                r'np\.random\.seed\(',
                r'torch\.manual_seed\(',
                r'tf\.random\.set_seed\(',
                r'random_state\s*='
            ]
            
            return any(re.search(pattern, content) for pattern in seed_patterns)
            
        except Exception:
            return False
    
    def _check_version_control(self) -> bool:
        """Check if project is under version control"""
        
        # Check for .git directory
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            if (current_dir / '.git').exists():
                return True
            current_dir = current_dir.parent
        
        return False

class BiasDetector:
    """Detect various forms of bias in research design and data"""
    
    def __init__(self):
        self.logger = logging.getLogger("bias_detector")
    
    def detect_bias(self, experiment_data: Dict[str, Any],
                   experiment_config: Dict[str, Any] = None) -> List[BiasDetection]:
        """Comprehensive bias detection"""
        
        detected_biases = []
        
        # Selection bias detection
        selection_biases = self._detect_selection_bias(experiment_data, experiment_config)
        detected_biases.extend(selection_biases)
        
        # Sampling bias detection
        sampling_biases = self._detect_sampling_bias(experiment_data)
        detected_biases.extend(sampling_biases)
        
        # Measurement bias detection
        measurement_biases = self._detect_measurement_bias(experiment_data)
        detected_biases.extend(measurement_biases)
        
        # Confirmation bias detection
        confirmation_biases = self._detect_confirmation_bias(experiment_data, experiment_config)
        detected_biases.extend(confirmation_biases)
        
        # Publication bias indicators
        publication_biases = self._detect_publication_bias_indicators(experiment_data)
        detected_biases.extend(publication_biases)
        
        return detected_biases
    
    def _detect_selection_bias(self, experiment_data: Dict[str, Any], 
                              experiment_config: Dict[str, Any] = None) -> List[BiasDetection]:
        """Detect selection bias in participant/sample selection"""
        
        biases = []
        
        # Check for systematic exclusion patterns
        if "exclusion_criteria" in experiment_data:
            exclusions = experiment_data["exclusion_criteria"]
            if isinstance(exclusions, dict):
                excluded_count = sum(exclusions.values())
                total_count = experiment_data.get("total_participants", excluded_count)
                
                if excluded_count / total_count > 0.3:  # More than 30% excluded
                    biases.append(BiasDetection(
                        bias_type="high_exclusion_rate",
                        severity="medium",
                        description=f"High exclusion rate: {excluded_count/total_count*100:.1f}% of participants excluded",
                        evidence={"exclusion_rate": excluded_count/total_count, "exclusions": exclusions},
                        confidence=0.8,
                        recommendations=[
                            "Review exclusion criteria for necessity",
                            "Report exclusion reasons in detail",
                            "Consider sensitivity analysis including excluded participants"
                        ]
                    ))
        
        # Check for demographic imbalances
        if "demographics" in experiment_data:
            demographics = experiment_data["demographics"]
            demographic_biases = self._analyze_demographic_bias(demographics)
            biases.extend(demographic_biases)
        
        return biases
    
    def _detect_sampling_bias(self, experiment_data: Dict[str, Any]) -> List[BiasDetection]:
        """Detect sampling bias"""
        
        biases = []
        
        # Check for non-representative sampling
        if "population_characteristics" in experiment_data and "sample_characteristics" in experiment_data:
            population = experiment_data["population_characteristics"]
            sample = experiment_data["sample_characteristics"]
            
            # Compare key demographic variables
            for characteristic in population:
                if characteristic in sample:
                    pop_value = population[characteristic]
                    sample_value = sample[characteristic]
                    
                    if isinstance(pop_value, (int, float)) and isinstance(sample_value, (int, float)):
                        relative_diff = abs(pop_value - sample_value) / pop_value if pop_value != 0 else float('inf')
                        
                        if relative_diff > 0.2:  # More than 20% difference
                            biases.append(BiasDetection(
                                bias_type="sampling_bias",
                                severity="medium",
                                description=f"Sample not representative for {characteristic}: population={pop_value}, sample={sample_value}",
                                evidence={"characteristic": characteristic, "population_value": pop_value, "sample_value": sample_value},
                                confidence=0.7,
                                recommendations=[
                                    "Use stratified sampling to improve representativeness",
                                    "Weight results to match population characteristics",
                                    "Acknowledge sampling limitations in discussion"
                                ]
                            ))
        
        # Check for convenience sampling indicators
        if experiment_data.get("sampling_method") == "convenience":
            biases.append(BiasDetection(
                bias_type="convenience_sampling",
                severity="low",
                description="Convenience sampling may limit generalizability",
                evidence={"sampling_method": "convenience"},
                confidence=0.6,
                recommendations=[
                    "Consider random sampling if feasible",
                    "Discuss generalizability limitations",
                    "Compare sample characteristics to target population"
                ]
            ))
        
        return biases
    
    def _detect_measurement_bias(self, experiment_data: Dict[str, Any]) -> List[BiasDetection]:
        """Detect measurement bias"""
        
        biases = []
        
        # Check for systematic measurement errors
        if "measurements" in experiment_data:
            measurements = experiment_data["measurements"]
            
            for measure_name, measure_data in measurements.items():
                if isinstance(measure_data, dict) and "values" in measure_data:
                    values = measure_data["values"]
                    
                    # Check for floor/ceiling effects
                    if len(values) > 10:
                        values_array = np.array(values)
                        
                        # Floor effect (many values at minimum)
                        min_val = np.min(values_array)
                        floor_percent = np.mean(values_array == min_val)
                        
                        if floor_percent > 0.15:  # More than 15% at floor
                            biases.append(BiasDetection(
                                bias_type="floor_effect",
                                severity="medium",
                                description=f"Floor effect in {measure_name}: {floor_percent*100:.1f}% at minimum value",
                                evidence={"measure": measure_name, "floor_percentage": floor_percent},
                                confidence=0.8,
                                recommendations=[
                                    "Review measurement scale sensitivity",
                                    "Consider alternative measurement approaches",
                                    "Report floor effect in limitations"
                                ]
                            ))
                        
                        # Ceiling effect (many values at maximum)
                        max_val = np.max(values_array)
                        ceiling_percent = np.mean(values_array == max_val)
                        
                        if ceiling_percent > 0.15:
                            biases.append(BiasDetection(
                                bias_type="ceiling_effect",
                                severity="medium",
                                description=f"Ceiling effect in {measure_name}: {ceiling_percent*100:.1f}% at maximum value",
                                evidence={"measure": measure_name, "ceiling_percentage": ceiling_percent},
                                confidence=0.8,
                                recommendations=[
                                    "Review measurement scale range",
                                    "Consider extending measurement scale",
                                    "Report ceiling effect in limitations"
                                ]
                            ))
        
        return biases
    
    def _detect_confirmation_bias(self, experiment_data: Dict[str, Any], 
                                 experiment_config: Dict[str, Any] = None) -> List[BiasDetection]:
        """Detect indicators of confirmation bias"""
        
        biases = []
        
        # Check for p-hacking indicators
        if "statistical_tests" in experiment_data:
            tests = experiment_data["statistical_tests"]
            
            if isinstance(tests, list) and len(tests) > 5:
                # Check for suspiciously many tests
                significant_tests = [t for t in tests if t.get("p_value", 1) < 0.05]
                
                if len(significant_tests) / len(tests) > 0.8:  # More than 80% significant
                    biases.append(BiasDetection(
                        bias_type="potential_p_hacking",
                        severity="high",
                        description=f"High proportion of significant results: {len(significant_tests)}/{len(tests)}",
                        evidence={"total_tests": len(tests), "significant_tests": len(significant_tests)},
                        confidence=0.6,
                        recommendations=[
                            "Pre-register analysis plan",
                            "Apply multiple comparison corrections",
                            "Report all analyses performed"
                        ]
                    ))
        
        # Check for selective reporting
        if experiment_config and "planned_analyses" in experiment_config:
            planned = set(experiment_config["planned_analyses"])
            if "reported_analyses" in experiment_data:
                reported = set(experiment_data["reported_analyses"])
                
                unreported = planned - reported
                additional = reported - planned
                
                if len(unreported) > 0 or len(additional) > len(planned) * 0.5:
                    biases.append(BiasDetection(
                        bias_type="selective_reporting",
                        severity="medium",
                        description="Discrepancy between planned and reported analyses",
                        evidence={"unreported_analyses": list(unreported), "additional_analyses": list(additional)},
                        confidence=0.7,
                        recommendations=[
                            "Report all planned analyses",
                            "Clearly distinguish exploratory from confirmatory analyses",
                            "Provide rationale for analysis changes"
                        ]
                    ))
        
        return biases
    
    def _detect_publication_bias_indicators(self, experiment_data: Dict[str, Any]) -> List[BiasDetection]:
        """Detect indicators that might lead to publication bias"""
        
        biases = []
        
        # Check for null result suppression indicators
        if "primary_outcome" in experiment_data:
            outcome = experiment_data["primary_outcome"]
            
            if isinstance(outcome, dict) and "p_value" in outcome:
                p_val = outcome["p_value"]
                
                # Marginally non-significant results might be suppressed
                if 0.05 < p_val < 0.10:
                    biases.append(BiasDetection(
                        bias_type="publication_bias_risk",
                        severity="low",
                        description=f"Marginally non-significant primary outcome (p={p_val:.3f}) at risk for publication bias",
                        evidence={"primary_outcome_p": p_val},
                        confidence=0.4,
                        recommendations=[
                            "Consider publishing regardless of significance",
                            "Focus on effect sizes and confidence intervals",
                            "Discuss practical significance"
                        ]
                    ))
        
        return biases
    
    def _analyze_demographic_bias(self, demographics: Dict[str, Any]) -> List[BiasDetection]:
        """Analyze demographic representation for bias"""
        
        biases = []
        
        # Check gender balance
        if "gender" in demographics:
            gender_dist = demographics["gender"]
            if isinstance(gender_dist, dict):
                total = sum(gender_dist.values())
                
                for gender, count in gender_dist.items():
                    proportion = count / total
                    
                    # Flag if any gender represents less than 20% or more than 80%
                    if proportion < 0.2 or proportion > 0.8:
                        biases.append(BiasDetection(
                            bias_type="gender_imbalance",
                            severity="low",
                            description=f"Gender imbalance: {gender} represents {proportion*100:.1f}% of sample",
                            evidence={"gender_distribution": gender_dist},
                            confidence=0.8,
                            recommendations=[
                                "Consider stratified sampling for better balance",
                                "Analyze results by gender subgroups",
                                "Discuss generalizability implications"
                            ]
                        ))
        
        # Check age distribution
        if "age" in demographics:
            age_data = demographics["age"]
            if isinstance(age_data, dict) and "mean" in age_data and "std" in age_data:
                age_range = age_data.get("range", [age_data["mean"] - 2*age_data["std"], 
                                                  age_data["mean"] + 2*age_data["std"]])
                
                # Flag very narrow age ranges
                if len(age_range) == 2 and (age_range[1] - age_range[0]) < 10:
                    biases.append(BiasDetection(
                        bias_type="age_restriction_bias",
                        severity="low",
                        description=f"Narrow age range may limit generalizability: {age_range[0]:.1f}-{age_range[1]:.1f} years",
                        evidence={"age_range": age_range},
                        confidence=0.6,
                        recommendations=[
                            "Consider broader age inclusion criteria",
                            "Discuss age-related generalizability",
                            "Consider age as a moderating variable"
                        ]
                    ))
        
        return biases

class ReproducibilityTester:
    """Test and assess reproducibility of research"""
    
    def __init__(self):
        self.logger = logging.getLogger("reproducibility_tester")
    
    def assess_reproducibility(self, experiment_config: Dict[str, Any],
                              code_files: List[str] = None,
                              data_files: List[str] = None,
                              environment_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive reproducibility assessment"""
        
        assessment_results = {
            "timestamp": datetime.now().isoformat(),
            "reproducibility_score": 0,
            "reproducibility_level": "unknown",
            "issues": [],
            "strengths": [],
            "requirements": []
        }
        
        all_issues = []
        strengths = []
        
        # Check code reproducibility
        if code_files:
            code_issues = self._assess_code_reproducibility(code_files)
            all_issues.extend(code_issues)
        
        # Check data reproducibility
        if data_files:
            data_issues = self._assess_data_reproducibility(data_files, experiment_config)
            all_issues.extend(data_issues)
        
        # Check environment reproducibility
        if environment_info:
            env_issues = self._assess_environment_reproducibility(environment_info)
            all_issues.extend(env_issues)
        
        # Check configuration reproducibility
        config_issues = self._assess_configuration_reproducibility(experiment_config)
        all_issues.extend(config_issues)
        
        # Calculate reproducibility score
        reproducibility_score = self._calculate_reproducibility_score(all_issues)
        
        # Determine reproducibility level
        if reproducibility_score >= 90:
            reproducibility_level = "fully_reproducible"
        elif reproducibility_score >= 80:
            reproducibility_level = "largely_reproducible"
        elif reproducibility_score >= 70:
            reproducibility_level = "partially_reproducible"
        elif reproducibility_score >= 60:
            reproducibility_level = "minimally_reproducible"
        else:
            reproducibility_level = "not_reproducible"
        
        assessment_results.update({
            "reproducibility_score": reproducibility_score,
            "reproducibility_level": reproducibility_level,
            "issues": [self._serialize_reproducibility_issue(issue) for issue in all_issues],
            "requirements": self._generate_reproducibility_requirements(all_issues)
        })
        
        return assessment_results
    
    def _assess_code_reproducibility(self, code_files: List[str]) -> List[ReproducibilityIssue]:
        """Assess code-related reproducibility issues"""
        
        issues = []
        
        # Check for random seed setting
        has_random_seed = False
        seed_files = []
        
        for file_path in code_files:
            if self._check_file_for_random_seed(file_path):
                has_random_seed = True
                seed_files.append(file_path)
        
        if not has_random_seed:
            issues.append(ReproducibilityIssue(
                issue_type="missing_random_seed",
                severity="high",
                description="No random seed setting found in code",
                affected_files=code_files,
                fix_suggestions=[
                    "Set random seed using np.random.seed()",
                    "Set random seed for all relevant libraries",
                    "Document seed value used"
                ],
                reproducibility_score_impact=-20
            ))
        
        # Check for hardcoded paths
        hardcoded_path_files = []
        for file_path in code_files:
            if self._check_file_for_hardcoded_paths(file_path):
                hardcoded_path_files.append(file_path)
        
        if hardcoded_path_files:
            issues.append(ReproducibilityIssue(
                issue_type="hardcoded_paths",
                severity="medium",
                description="Hardcoded file paths found",
                affected_files=hardcoded_path_files,
                fix_suggestions=[
                    "Use relative paths or configuration files",
                    "Parameterize file paths",
                    "Use pathlib for cross-platform compatibility"
                ],
                reproducibility_score_impact=-10
            ))
        
        # Check for dependency management
        has_requirements = any(Path(f).name in ['requirements.txt', 'environment.yml', 'Pipfile'] 
                              for f in code_files)
        if not has_requirements:
            issues.append(ReproducibilityIssue(
                issue_type="missing_dependency_management",
                severity="high",
                description="No dependency management file found",
                affected_files=code_files,
                fix_suggestions=[
                    "Create requirements.txt with exact versions",
                    "Use conda environment.yml",
                    "Document Python version used"
                ],
                reproducibility_score_impact=-15
            ))
        
        return issues
    
    def _assess_data_reproducibility(self, data_files: List[str], 
                                   experiment_config: Dict[str, Any]) -> List[ReproducibilityIssue]:
        """Assess data-related reproducibility issues"""
        
        issues = []
        
        # Check for data preprocessing documentation
        if not experiment_config.get("data_preprocessing_documented", False):
            issues.append(ReproducibilityIssue(
                issue_type="undocumented_preprocessing",
                severity="medium",
                description="Data preprocessing steps not documented",
                affected_files=data_files,
                fix_suggestions=[
                    "Document all data cleaning steps",
                    "Provide data preprocessing code",
                    "Record original data characteristics"
                ],
                reproducibility_score_impact=-12
            ))
        
        # Check for data versioning
        if not experiment_config.get("data_versioned", False):
            issues.append(ReproducibilityIssue(
                issue_type="unversioned_data",
                severity="medium",
                description="Data files not versioned",
                affected_files=data_files,
                fix_suggestions=[
                    "Use data version control (DVC)",
                    "Document data collection dates",
                    "Provide data checksums"
                ],
                reproducibility_score_impact=-8
            ))
        
        return issues
    
    def _assess_environment_reproducibility(self, environment_info: Dict[str, Any]) -> List[ReproducibilityIssue]:
        """Assess computational environment reproducibility"""
        
        issues = []
        
        # Check for operating system documentation
        if "os" not in environment_info:
            issues.append(ReproducibilityIssue(
                issue_type="missing_os_info",
                severity="low",
                description="Operating system not documented",
                affected_files=[],
                fix_suggestions=[
                    "Document operating system and version",
                    "Test on multiple platforms if possible"
                ],
                reproducibility_score_impact=-5
            ))
        
        # Check for hardware specifications
        if "hardware" not in environment_info:
            issues.append(ReproducibilityIssue(
                issue_type="missing_hardware_specs",
                severity="low",
                description="Hardware specifications not documented",
                affected_files=[],
                fix_suggestions=[
                    "Document CPU, RAM, GPU specifications",
                    "Note if results are hardware-dependent"
                ],
                reproducibility_score_impact=-3
            ))
        
        return issues
    
    def _assess_configuration_reproducibility(self, experiment_config: Dict[str, Any]) -> List[ReproducibilityIssue]:
        """Assess experiment configuration reproducibility"""
        
        issues = []
        
        # Check for parameter documentation
        critical_params = ["learning_rate", "batch_size", "epochs", "model_architecture"]
        missing_params = [param for param in critical_params 
                         if param in experiment_config and experiment_config[param] is None]
        
        if missing_params:
            issues.append(ReproducibilityIssue(
                issue_type="missing_parameters",
                severity="medium",
                description=f"Critical parameters not specified: {missing_params}",
                affected_files=[],
                fix_suggestions=[
                    "Specify all hyperparameters explicitly",
                    "Document parameter selection rationale",
                    "Provide parameter search details"
                ],
                reproducibility_score_impact=-10
            ))
        
        return issues
    
    def _check_file_for_random_seed(self, file_path: str) -> bool:
        """Check if file contains random seed setting"""
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            seed_patterns = [
                r'random\.seed\s*\(',
                r'np\.random\.seed\s*\(',
                r'torch\.manual_seed\s*\(',
                r'tf\.random\.set_seed\s*\(',
                r'random_state\s*='
            ]
            
            return any(re.search(pattern, content) for pattern in seed_patterns)
        
        except Exception:
            return False
    
    def _check_file_for_hardcoded_paths(self, file_path: str) -> bool:
        """Check if file contains hardcoded paths"""
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for absolute path patterns
            path_patterns = [
                r'["\']\/[^"\']*["\']',  # Unix absolute paths
                r'["\'][A-Z]:\\[^"\']*["\']',  # Windows absolute paths
                r'["\']C:\\[^"\']*["\']',  # Common C: drive paths
            ]
            
            return any(re.search(pattern, content) for pattern in path_patterns)
        
        except Exception:
            return False
    
    def _calculate_reproducibility_score(self, issues: List[ReproducibilityIssue]) -> float:
        """Calculate overall reproducibility score"""
        
        base_score = 100
        total_impact = sum(abs(issue.reproducibility_score_impact) for issue in issues)
        
        return max(0, base_score - total_impact)
    
    def _serialize_reproducibility_issue(self, issue: ReproducibilityIssue) -> Dict[str, Any]:
        """Serialize reproducibility issue"""
        
        return {
            "type": issue.issue_type,
            "severity": issue.severity,
            "description": issue.description,
            "affected_files": issue.affected_files,
            "fix_suggestions": issue.fix_suggestions,
            "score_impact": issue.reproducibility_score_impact
        }

class PeerReviewPreparer:
    """Prepare experiment for peer review"""
    
    def __init__(self):
        self.logger = logging.getLogger("peer_review_preparer")
        self.checklist = self._initialize_peer_review_checklist()
    
    def prepare_peer_review_checklist(self, experiment_config: Dict[str, Any],
                                    results_data: Dict[str, Any] = None,
                                    code_files: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive peer review preparation checklist"""
        
        checklist_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_readiness_score": 0,
            "readiness_level": "unknown",
            "checklist_items": [],
            "critical_issues": [],
            "recommendations": []
        }
        
        checklist_items = []
        
        # Evaluate each checklist category
        for category, items in self.checklist.items():
            category_items = self._evaluate_checklist_category(
                category, items, experiment_config, results_data, code_files
            )
            checklist_items.extend(category_items)
        
        # Calculate readiness score
        total_items = len(checklist_items)
        passed_items = sum(1 for item in checklist_items if item.status == "pass")
        readiness_score = (passed_items / total_items * 100) if total_items > 0 else 0
        
        # Determine readiness level
        if readiness_score >= 90:
            readiness_level = "ready_for_review"
        elif readiness_score >= 80:
            readiness_level = "mostly_ready"
        elif readiness_score >= 70:
            readiness_level = "needs_minor_revisions"
        elif readiness_score >= 60:
            readiness_level = "needs_major_revisions"
        else:
            readiness_level = "not_ready"
        
        # Identify critical issues
        critical_issues = [item for item in checklist_items 
                          if item.status == "fail" and item.priority == "critical"]
        
        checklist_results.update({
            "overall_readiness_score": readiness_score,
            "readiness_level": readiness_level,
            "checklist_items": [self._serialize_checklist_item(item) for item in checklist_items],
            "critical_issues": [item.item_name for item in critical_issues],
            "recommendations": self._generate_peer_review_recommendations(checklist_items)
        })
        
        return checklist_results
    
    def _initialize_peer_review_checklist(self) -> Dict[str, Dict[str, Any]]:
        """Initialize peer review checklist"""
        
        return {
            "methodology": {
                "clear_research_question": {
                    "description": "Research question is clearly stated and focused",
                    "priority": "critical"
                },
                "appropriate_design": {
                    "description": "Research design is appropriate for the question",
                    "priority": "critical"
                },
                "adequate_sample_size": {
                    "description": "Sample size is adequate and justified",
                    "priority": "high"
                },
                "control_conditions": {
                    "description": "Appropriate control conditions are included",
                    "priority": "high"
                }
            },
            "statistical_analysis": {
                "appropriate_tests": {
                    "description": "Statistical tests are appropriate for data type",
                    "priority": "critical"
                },
                "assumptions_checked": {
                    "description": "Statistical assumptions are verified",
                    "priority": "high"
                },
                "effect_sizes_reported": {
                    "description": "Effect sizes are calculated and reported",
                    "priority": "high"
                },
                "multiple_testing_addressed": {
                    "description": "Multiple testing issues are addressed",
                    "priority": "medium"
                }
            },
            "reporting": {
                "complete_methodology": {
                    "description": "Methodology is completely described",
                    "priority": "critical"
                },
                "all_results_reported": {
                    "description": "All relevant results are reported",
                    "priority": "critical"
                },
                "limitations_discussed": {
                    "description": "Study limitations are acknowledged",
                    "priority": "medium"
                },
                "implications_clear": {
                    "description": "Implications and conclusions are clearly stated",
                    "priority": "medium"
                }
            },
            "reproducibility": {
                "code_available": {
                    "description": "Analysis code is available",
                    "priority": "high"
                },
                "data_available": {
                    "description": "Data availability is clearly stated",
                    "priority": "high"
                },
                "environment_documented": {
                    "description": "Computational environment is documented",
                    "priority": "medium"
                },
                "materials_described": {
                    "description": "All materials and procedures are described",
                    "priority": "medium"
                }
            }
        }
    
    def _evaluate_checklist_category(self, category: str, items: Dict[str, Any],
                                   experiment_config: Dict[str, Any],
                                   results_data: Dict[str, Any] = None,
                                   code_files: List[str] = None) -> List[PeerReviewItem]:
        """Evaluate a category of checklist items"""
        
        checklist_items = []
        
        for item_name, item_config in items.items():
            status = self._evaluate_checklist_item(
                category, item_name, item_config, experiment_config, results_data, code_files
            )
            
            checklist_items.append(PeerReviewItem(
                category=category,
                item_name=item_name,
                status=status["status"],
                details=status["details"],
                suggestions=status.get("suggestions", []),
                priority=item_config.get("priority", "medium")
            ))
        
        return checklist_items
    
    def _evaluate_checklist_item(self, category: str, item_name: str, item_config: Dict[str, Any],
                               experiment_config: Dict[str, Any],
                               results_data: Dict[str, Any] = None,
                               code_files: List[str] = None) -> Dict[str, Any]:
        """Evaluate individual checklist item"""
        
        # Default evaluation
        evaluation = {
            "status": "needs_attention",
            "details": "Manual review required",
            "suggestions": []
        }
        
        # Methodology evaluations
        if category == "methodology":
            if item_name == "clear_research_question":
                if "research_question" in experiment_config and experiment_config["research_question"]:
                    evaluation["status"] = "pass"
                    evaluation["details"] = "Research question specified"
                else:
                    evaluation["status"] = "fail"
                    evaluation["details"] = "No research question found"
                    evaluation["suggestions"] = ["Add clear research question to experiment configuration"]
            
            elif item_name == "adequate_sample_size":
                if "sample_size" in experiment_config:
                    sample_size = experiment_config["sample_size"]
                    if sample_size >= 30:  # Basic threshold
                        evaluation["status"] = "pass"
                        evaluation["details"] = f"Sample size: {sample_size}"
                    else:
                        evaluation["status"] = "fail"
                        evaluation["details"] = f"Sample size may be too small: {sample_size}"
                        evaluation["suggestions"] = ["Consider increasing sample size", "Provide power analysis justification"]
        
        # Statistical analysis evaluations
        elif category == "statistical_analysis":
            if item_name == "effect_sizes_reported":
                if results_data and "effect_sizes" in results_data:
                    evaluation["status"] = "pass"
                    evaluation["details"] = "Effect sizes calculated"
                else:
                    evaluation["status"] = "fail"
                    evaluation["details"] = "No effect sizes found"
                    evaluation["suggestions"] = ["Calculate and report appropriate effect sizes"]
        
        # Reproducibility evaluations
        elif category == "reproducibility":
            if item_name == "code_available":
                if code_files and len(code_files) > 0:
                    evaluation["status"] = "pass"
                    evaluation["details"] = f"Code files provided: {len(code_files)}"
                else:
                    evaluation["status"] = "fail"
                    evaluation["details"] = "No code files provided"
                    evaluation["suggestions"] = ["Provide analysis code"]
        
        return evaluation
    
    def _serialize_checklist_item(self, item: PeerReviewItem) -> Dict[str, Any]:
        """Serialize checklist item"""
        
        return {
            "category": item.category,
            "item_name": item.item_name,
            "status": item.status,
            "details": item.details,
            "suggestions": item.suggestions or [],
            "priority": item.priority
        }

class ScientificRigorValidator:
    """Main scientific rigor validation controller"""
    
    def __init__(self):
        self.logger = logging.getLogger("scientific_rigor_validator")
        
        # Initialize components
        self.methodology_checker = MethodologyComplianceChecker()
        self.bias_detector = BiasDetector()
        self.reproducibility_tester = ReproducibilityTester()
        self.peer_review_preparer = PeerReviewPreparer()
        
        # Initialize advanced components
        try:
            from .advanced_bias_detector import AdvancedBiasDetector
            from .metadata_validator import MetadataValidator
            from .causal_inference_validator import CausalInferenceValidator
            
            self.advanced_bias_detector = AdvancedBiasDetector()
            self.metadata_validator = MetadataValidator()
            self.causal_validator = CausalInferenceValidator()
            self.advanced_available = True
        except ImportError as e:
            self.logger.warning(f"Advanced validation components not available: {e}")
            self.advanced_bias_detector = None
            self.metadata_validator = None
            self.causal_validator = None
            self.advanced_available = False
    
    def validate_scientific_rigor(self, experiment_config: Dict[str, Any],
                                 experiment_data: Dict[str, Any] = None,
                                 code_files: List[str] = None,
                                 data_files: List[str] = None,
                                 environment_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive scientific rigor validation"""
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "methodology_compliance": {},
            "bias_detection": {},
            "reproducibility_assessment": {},
            "peer_review_readiness": {},
            "overall_rigor_assessment": {}
        }
        
        # 1. Methodology compliance check
        self.logger.info("Checking methodology compliance")
        methodology_results = self.methodology_checker.check_compliance(
            experiment_config, code_files, data_files
        )
        validation_results["methodology_compliance"] = methodology_results
        
        # 2. Bias detection
        if experiment_data:
            self.logger.info("Detecting potential biases")
            detected_biases = self.bias_detector.detect_bias(experiment_data, experiment_config)
            validation_results["bias_detection"] = {
                "total_biases_detected": len(detected_biases),
                "biases_by_severity": self._categorize_biases_by_severity(detected_biases),
                "detected_biases": [self._serialize_bias_detection(bias) for bias in detected_biases]
            }
        
        # 3. Reproducibility assessment
        self.logger.info("Assessing reproducibility")
        reproducibility_results = self.reproducibility_tester.assess_reproducibility(
            experiment_config, code_files, data_files, environment_info
        )
        validation_results["reproducibility_assessment"] = reproducibility_results
        
        # 4. Peer review preparation
        self.logger.info("Preparing peer review checklist")
        peer_review_results = self.peer_review_preparer.prepare_peer_review_checklist(
            experiment_config, experiment_data, code_files
        )
        validation_results["peer_review_readiness"] = peer_review_results
        
        # 5. Advanced analysis (if available)
        if self.advanced_available:
            # Advanced bias detection
            self.logger.info("Performing advanced bias detection")
            advanced_bias_results = self.advanced_bias_detector.detect_advanced_biases(
                experiment_data.get("data", {}),
                experiment_data.get("protected_attributes", []),
                experiment_data.get("target_variable")
            )
            validation_results["advanced_bias_detection"] = {
                bias_name: self._serialize_advanced_bias(bias)
                for bias_name, bias in advanced_bias_results.items()
            }
            
            # Metadata validation
            if "metadata" in experiment_data:
                self.logger.info("Validating metadata")
                metadata_results = self.metadata_validator.validate_metadata(experiment_data["metadata"])
                validation_results["metadata_validation"] = self._serialize_metadata_validation(metadata_results)
            
            # Causal inference validation
            if "experimental_design" in experiment_data:
                self.logger.info("Validating causal inference")
                causal_results = self.causal_validator.validate_causal_inference(
                    experiment_data["experimental_design"],
                    experiment_data.get("data")
                )
                validation_results["causal_inference_validation"] = self._serialize_causal_validation(causal_results)
        
        # 6. Overall assessment
        validation_results["overall_rigor_assessment"] = self._generate_overall_rigor_assessment(validation_results)
        
        return validation_results
    
    def _categorize_biases_by_severity(self, biases: List[BiasDetection]) -> Dict[str, int]:
        """Categorize detected biases by severity"""
        
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for bias in biases:
            severity_counts[bias.severity] = severity_counts.get(bias.severity, 0) + 1
        
        return severity_counts
    
    def _serialize_bias_detection(self, bias: BiasDetection) -> Dict[str, Any]:
        """Serialize bias detection result"""
        
        return {
            "bias_type": bias.bias_type,
            "severity": bias.severity,
            "description": bias.description,
            "evidence": bias.evidence,
            "confidence": bias.confidence,
            "recommendations": bias.recommendations,
            "affected_components": bias.affected_components
        }
    
    def _generate_overall_rigor_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall scientific rigor assessment"""
        
        assessment = {
            "overall_rigor_score": 0,
            "rigor_level": "unknown",
            "key_strengths": [],
            "critical_issues": [],
            "improvement_recommendations": []
        }
        
        # Calculate component scores
        methodology_score = validation_results["methodology_compliance"].get("overall_compliance_score", 0)
        reproducibility_score = validation_results["reproducibility_assessment"].get("reproducibility_score", 0)
        peer_review_score = validation_results["peer_review_readiness"].get("overall_readiness_score", 0)
        
        # Bias penalty
        bias_penalty = 0
        if "bias_detection" in validation_results:
            biases = validation_results["bias_detection"]["biases_by_severity"]
            bias_penalty = biases.get("critical", 0) * 20 + biases.get("high", 0) * 10 + biases.get("medium", 0) * 5
        
        # Weighted overall score
        weights = {"methodology": 0.35, "reproducibility": 0.35, "peer_review": 0.3}
        overall_score = (
            methodology_score * weights["methodology"] +
            reproducibility_score * weights["reproducibility"] +
            peer_review_score * weights["peer_review"]
        ) - bias_penalty
        
        overall_score = max(0, overall_score)
        
        assessment["overall_rigor_score"] = overall_score
        
        # Determine rigor level
        if overall_score >= 90:
            assessment["rigor_level"] = "excellent"
        elif overall_score >= 80:
            assessment["rigor_level"] = "good"
        elif overall_score >= 70:
            assessment["rigor_level"] = "acceptable"
        elif overall_score >= 60:
            assessment["rigor_level"] = "needs_improvement"
        else:
            assessment["rigor_level"] = "inadequate"
        
        # Identify strengths and issues
        if methodology_score >= 85:
            assessment["key_strengths"].append("Strong methodology compliance")
        
        if reproducibility_score >= 85:
            assessment["key_strengths"].append("High reproducibility")
        
        if peer_review_score >= 85:
            assessment["key_strengths"].append("Ready for peer review")
        
        # Critical issues
        if methodology_score < 60:
            assessment["critical_issues"].append("Poor methodology compliance")
        
        if reproducibility_score < 60:
            assessment["critical_issues"].append("Low reproducibility")
        
        if bias_penalty > 30:
            assessment["critical_issues"].append("Significant bias concerns")
        
        # Recommendations
        if methodology_score < 80:
            assessment["improvement_recommendations"].append("Improve methodology compliance")
        
        if reproducibility_score < 80:
            assessment["improvement_recommendations"].append("Enhance reproducibility measures")
        
        if peer_review_score < 80:
            assessment["improvement_recommendations"].append("Address peer review preparation items")
        
        if bias_penalty > 10:
            assessment["improvement_recommendations"].append("Address detected biases")
        
        # Advanced analysis recommendations (if available)
        if self.advanced_available and "advanced_bias_detection" in validation_results:
            high_severity_biases = [bias for bias in validation_results["advanced_bias_detection"].values()
                                   if bias.get("severity_score", 0) > 0.7]
            if high_severity_biases:
                assessment["improvement_recommendations"].append(f"Address {len(high_severity_biases)} high-severity biases")
        
        if "metadata_validation" in validation_results:
            missing_critical = len([field for field in validation_results["metadata_validation"].get("missing_fields", [])
                                   if "critical" in field])
            if missing_critical > 0:
                assessment["improvement_recommendations"].append(f"Add {missing_critical} missing critical metadata fields")
        
        if "causal_inference_validation" in validation_results:
            validity_threats = len(validation_results["causal_inference_validation"].get("validity_threats", []))
            if validity_threats > 3:
                assessment["improvement_recommendations"].append(f"Address {validity_threats} threats to causal validity")
        
        return assessment
    
    def _serialize_advanced_bias(self, bias: AdvancedBiasDetection) -> Dict[str, Any]:
        """Serialize advanced bias detection result"""
        
        return {
            "bias_category": bias.bias_category,
            "bias_subcategory": bias.bias_subcategory,
            "statistical_evidence": bias.statistical_evidence,
            "demographic_analysis": bias.demographic_analysis,
            "fairness_metrics": bias.fairness_metrics,
            "severity_score": bias.severity_score,
            "impact_assessment": bias.impact_assessment,
            "mitigation_strategies": bias.mitigation_strategies
        }
    
    def _serialize_metadata_validation(self, metadata_val: MetadataValidation) -> Dict[str, Any]:
        """Serialize metadata validation result"""
        
        return {
            "completeness_score": metadata_val.completeness_score,
            "consistency_score": metadata_val.consistency_score,
            "quality_issues": metadata_val.quality_issues,
            "missing_fields": metadata_val.missing_fields,
            "inconsistent_fields": metadata_val.inconsistent_fields,
            "recommendations": metadata_val.recommendations,
            "metadata_graph_analysis": metadata_val.metadata_graph_analysis
        }
    
    def _serialize_causal_validation(self, causal_val: CausalInferenceValidation) -> Dict[str, Any]:
        """Serialize causal inference validation result"""
        
        return {
            "confounding_variables": causal_val.confounding_variables,
            "mediation_analysis": causal_val.mediation_analysis,
            "instrumental_variables": causal_val.instrumental_variables,
            "causal_assumptions": causal_val.causal_assumptions,
            "validity_threats": causal_val.validity_threats,
            "causal_graph_issues": causal_val.causal_graph_issues
        }