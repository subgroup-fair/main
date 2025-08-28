"""
Advanced Bias Detection System
Comprehensive bias detection using multiple techniques and fairness metrics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

# Statistical libraries
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SCIPY_SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_SKLEARN_AVAILABLE = False

# Advanced bias detection libraries
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.reductions import ExponentiatedGradient
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

from .scientific_rigor_validator import AdvancedBiasDetection


class AdvancedBiasDetector:
    """Advanced bias detection using multiple techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger("advanced_bias_detector")
    
    def detect_advanced_biases(self, 
                              data: Dict[str, Any],
                              protected_attributes: List[str] = None,
                              target_variable: str = None) -> Dict[str, AdvancedBiasDetection]:
        """Detect various types of biases using advanced techniques"""
        
        bias_detections = {}
        
        # 1. Algorithmic bias detection
        if FAIRLEARN_AVAILABLE and protected_attributes and target_variable:
            fairness_biases = self._detect_algorithmic_bias(data, protected_attributes, target_variable)
            bias_detections.update(fairness_biases)
        
        # 2. Statistical bias detection
        statistical_biases = self._detect_statistical_biases(data)
        bias_detections.update(statistical_biases)
        
        # 3. Sampling bias detection
        sampling_biases = self._detect_sampling_biases(data)
        bias_detections.update(sampling_biases)
        
        # 4. Measurement bias detection
        measurement_biases = self._detect_measurement_biases(data)
        bias_detections.update(measurement_biases)
        
        # 5. Cognitive bias indicators
        cognitive_biases = self._detect_cognitive_bias_indicators(data)
        bias_detections.update(cognitive_biases)
        
        return bias_detections
    
    def _detect_algorithmic_bias(self, 
                                data: Dict[str, Any],
                                protected_attributes: List[str],
                                target_variable: str) -> Dict[str, AdvancedBiasDetection]:
        """Detect algorithmic fairness issues"""
        
        biases = {}
        
        try:
            if "predictions" not in data or "actual" not in data:
                return biases
            
            predictions = np.array(data["predictions"])
            actual = np.array(data["actual"])
            
            for attr in protected_attributes:
                if attr not in data:
                    continue
                
                protected_values = np.array(data[attr])
                
                # Calculate fairness metrics
                fairness_metrics = {}
                
                # Demographic parity
                if len(np.unique(protected_values)) == 2:
                    dp_diff = demographic_parity_difference(
                        actual, predictions, sensitive_features=protected_values
                    )
                    fairness_metrics["demographic_parity_difference"] = dp_diff
                    
                    # Equalized odds
                    eo_diff = equalized_odds_difference(
                        actual, predictions, sensitive_features=protected_values
                    )
                    fairness_metrics["equalized_odds_difference"] = eo_diff
                
                # Statistical analysis by group
                unique_values = np.unique(protected_values)
                group_stats = {}
                
                for value in unique_values:
                    mask = protected_values == value
                    group_pred = predictions[mask]
                    group_actual = actual[mask]
                    
                    if len(group_pred) > 0:
                        group_stats[str(value)] = {
                            "mean_prediction": np.mean(group_pred),
                            "accuracy": np.mean(group_pred == group_actual) if len(group_actual) > 0 else 0,
                            "sample_size": len(group_pred)
                        }
                
                # Determine bias severity
                severity_score = abs(dp_diff) + abs(eo_diff) if "demographic_parity_difference" in fairness_metrics else 0
                
                if severity_score > 0.2:
                    severity = "critical"
                elif severity_score > 0.1:
                    severity = "high"
                elif severity_score > 0.05:
                    severity = "medium"
                else:
                    severity = "low"
                
                biases[f"algorithmic_bias_{attr}"] = AdvancedBiasDetection(
                    bias_category="algorithmic",
                    bias_subcategory=f"fairness_{attr}",
                    statistical_evidence={"severity_score": severity_score},
                    demographic_analysis=group_stats,
                    fairness_metrics=fairness_metrics,
                    severity_score=severity_score,
                    impact_assessment=f"{severity} algorithmic bias detected for {attr}",
                    mitigation_strategies=[
                        "Apply fairness-aware machine learning techniques",
                        "Use bias mitigation preprocessing methods",
                        "Implement fairness constraints in model training",
                        "Regular bias auditing and monitoring"
                    ]
                )
                
        except Exception as e:
            self.logger.error(f"Algorithmic bias detection failed: {e}")
        
        return biases
    
    def _detect_statistical_biases(self, data: Dict[str, Any]) -> Dict[str, AdvancedBiasDetection]:
        """Detect statistical biases in data"""
        
        biases = {}
        
        try:
            # Selection bias detection
            if "sample_characteristics" in data and "population_characteristics" in data:
                sample_chars = data["sample_characteristics"]
                pop_chars = data["population_characteristics"]
                
                # Compare sample vs population distributions
                distribution_diffs = {}
                for char in sample_chars:
                    if char in pop_chars:
                        sample_mean = np.mean(sample_chars[char])
                        pop_mean = np.mean(pop_chars[char])
                        diff = abs(sample_mean - pop_mean) / pop_mean if pop_mean != 0 else 0
                        distribution_diffs[char] = diff
                
                avg_diff = np.mean(list(distribution_diffs.values())) if distribution_diffs else 0
                
                if avg_diff > 0.2:
                    severity_score = min(avg_diff, 1.0)
                    biases["selection_bias"] = AdvancedBiasDetection(
                        bias_category="statistical",
                        bias_subcategory="selection",
                        statistical_evidence={
                            "distribution_differences": distribution_diffs,
                            "average_difference": avg_diff
                        },
                        demographic_analysis={},
                        fairness_metrics={},
                        severity_score=severity_score,
                        impact_assessment=f"Selection bias detected (avg diff: {avg_diff:.3f})",
                        mitigation_strategies=[
                            "Use stratified sampling methods",
                            "Apply population weighting",
                            "Collect more representative samples",
                            "Document sampling methodology"
                        ]
                    )
            
            # Survivorship bias detection
            if "dropouts" in data and "completers" in data:
                dropout_rate = len(data["dropouts"]) / (len(data["dropouts"]) + len(data["completers"]))
                
                if dropout_rate > 0.2:
                    severity_score = min(dropout_rate, 1.0)
                    biases["survivorship_bias"] = AdvancedBiasDetection(
                        bias_category="statistical",
                        bias_subcategory="survivorship",
                        statistical_evidence={"dropout_rate": dropout_rate},
                        demographic_analysis={},
                        fairness_metrics={},
                        severity_score=severity_score,
                        impact_assessment=f"High dropout rate detected: {dropout_rate:.3f}",
                        mitigation_strategies=[
                            "Analyze dropout patterns",
                            "Use intention-to-treat analysis",
                            "Implement missing data imputation",
                            "Report attrition bias assessment"
                        ]
                    )
            
        except Exception as e:
            self.logger.error(f"Statistical bias detection failed: {e}")
        
        return biases
    
    def _detect_sampling_biases(self, data: Dict[str, Any]) -> Dict[str, AdvancedBiasDetection]:
        """Detect sampling-related biases"""
        
        biases = {}
        
        try:
            # Temporal bias detection
            if "timestamps" in data:
                timestamps = pd.to_datetime(data["timestamps"])
                
                # Check for uneven temporal distribution
                time_diff = timestamps.max() - timestamps.min()
                daily_counts = timestamps.dt.date.value_counts()
                
                cv = daily_counts.std() / daily_counts.mean() if daily_counts.mean() > 0 else 0
                
                if cv > 1.0:  # High coefficient of variation suggests uneven sampling
                    biases["temporal_bias"] = AdvancedBiasDetection(
                        bias_category="sampling",
                        bias_subcategory="temporal",
                        statistical_evidence={
                            "coefficient_of_variation": cv,
                            "time_span_days": time_diff.days
                        },
                        demographic_analysis={},
                        fairness_metrics={},
                        severity_score=min(cv / 2, 1.0),
                        impact_assessment=f"Uneven temporal sampling detected (CV: {cv:.3f})",
                        mitigation_strategies=[
                            "Use systematic temporal sampling",
                            "Account for temporal effects in analysis",
                            "Report temporal distribution",
                            "Consider seasonal adjustments"
                        ]
                    )
            
            # Geographic bias detection
            if "locations" in data:
                locations = data["locations"]
                location_counts = Counter(locations)
                
                # Check for geographic concentration
                most_common_pct = location_counts.most_common(1)[0][1] / len(locations)
                
                if most_common_pct > 0.5:
                    biases["geographic_bias"] = AdvancedBiasDetection(
                        bias_category="sampling",
                        bias_subcategory="geographic",
                        statistical_evidence={
                            "max_location_percentage": most_common_pct,
                            "location_distribution": dict(location_counts)
                        },
                        demographic_analysis={},
                        fairness_metrics={},
                        severity_score=most_common_pct,
                        impact_assessment=f"Geographic concentration detected: {most_common_pct:.3f}",
                        mitigation_strategies=[
                            "Diversify geographic sampling",
                            "Use geographic stratification",
                            "Weight by population density",
                            "Report geographic limitations"
                        ]
                    )
            
        except Exception as e:
            self.logger.error(f"Sampling bias detection failed: {e}")
        
        return biases
    
    def _detect_measurement_biases(self, data: Dict[str, Any]) -> Dict[str, AdvancedBiasDetection]:
        """Detect measurement-related biases"""
        
        biases = {}
        
        try:
            # Response bias detection (if survey data)
            if "responses" in data and "response_times" in data:
                responses = np.array(data["responses"])
                response_times = np.array(data["response_times"])
                
                # Check for suspiciously fast responses
                fast_responses = response_times < np.percentile(response_times, 10)
                fast_response_rate = np.mean(fast_responses)
                
                if fast_response_rate > 0.1:
                    biases["response_bias"] = AdvancedBiasDetection(
                        bias_category="measurement",
                        bias_subcategory="response",
                        statistical_evidence={
                            "fast_response_rate": fast_response_rate,
                            "median_response_time": np.median(response_times)
                        },
                        demographic_analysis={},
                        fairness_metrics={},
                        severity_score=fast_response_rate,
                        impact_assessment=f"High fast response rate: {fast_response_rate:.3f}",
                        mitigation_strategies=[
                            "Implement minimum response time checks",
                            "Add attention check questions",
                            "Use response time filtering",
                            "Validate response quality"
                        ]
                    )
            
            # Instrument bias detection
            if "measurements" in data and "instruments" in data:
                measurements = data["measurements"]
                instruments = data["instruments"]
                
                # Check for systematic differences between instruments
                instrument_stats = {}
                for instrument in set(instruments):
                    mask = np.array(instruments) == instrument
                    instrument_measurements = np.array(measurements)[mask]
                    instrument_stats[instrument] = {
                        "mean": np.mean(instrument_measurements),
                        "std": np.std(instrument_measurements),
                        "count": len(instrument_measurements)
                    }
                
                # Calculate coefficient of variation between instruments
                means = [stats["mean"] for stats in instrument_stats.values()]
                if len(means) > 1:
                    cv_between = np.std(means) / np.mean(means) if np.mean(means) > 0 else 0
                    
                    if cv_between > 0.1:
                        biases["instrument_bias"] = AdvancedBiasDetection(
                            bias_category="measurement",
                            bias_subcategory="instrument",
                            statistical_evidence={
                                "cv_between_instruments": cv_between,
                                "instrument_statistics": instrument_stats
                            },
                            demographic_analysis={},
                            fairness_metrics={},
                            severity_score=cv_between,
                            impact_assessment=f"Instrument bias detected (CV: {cv_between:.3f})",
                            mitigation_strategies=[
                                "Calibrate instruments regularly",
                                "Use instrument fixed effects",
                                "Report instrument specifications",
                                "Cross-validate measurements"
                            ]
                        )
            
        except Exception as e:
            self.logger.error(f"Measurement bias detection failed: {e}")
        
        return biases
    
    def _detect_cognitive_bias_indicators(self, data: Dict[str, Any]) -> Dict[str, AdvancedBiasDetection]:
        """Detect indicators of cognitive biases in research design"""
        
        biases = {}
        
        try:
            # Confirmation bias indicators
            if "hypothesis" in data and "analysis_choices" in data:
                hypothesis = data["hypothesis"]
                choices = data["analysis_choices"]
                
                # Check for analysis choices that favor hypothesis
                favorable_choices = 0
                total_choices = len(choices)
                
                for choice in choices:
                    if any(keyword in choice.lower() for keyword in ["significant", "positive", "confirm"]):
                        favorable_choices += 1
                
                bias_ratio = favorable_choices / total_choices if total_choices > 0 else 0
                
                if bias_ratio > 0.7:
                    biases["confirmation_bias"] = AdvancedBiasDetection(
                        bias_category="cognitive",
                        bias_subcategory="confirmation",
                        statistical_evidence={
                            "favorable_choice_ratio": bias_ratio,
                            "total_analysis_choices": total_choices
                        },
                        demographic_analysis={},
                        fairness_metrics={},
                        severity_score=bias_ratio,
                        impact_assessment=f"Potential confirmation bias (ratio: {bias_ratio:.3f})",
                        mitigation_strategies=[
                            "Pre-register analysis plans",
                            "Use blinded analysis procedures",
                            "Include alternative hypotheses",
                            "Seek disconfirming evidence"
                        ]
                    )
            
            # Cherry-picking indicators
            if "multiple_analyses" in data and "reported_results" in data:
                total_analyses = len(data["multiple_analyses"])
                reported_results = len(data["reported_results"])
                
                reporting_ratio = reported_results / total_analyses if total_analyses > 0 else 1
                
                if reporting_ratio < 0.5 and total_analyses > 5:
                    biases["cherry_picking"] = AdvancedBiasDetection(
                        bias_category="cognitive",
                        bias_subcategory="selective_reporting",
                        statistical_evidence={
                            "reporting_ratio": reporting_ratio,
                            "total_analyses": total_analyses,
                            "reported_analyses": reported_results
                        },
                        demographic_analysis={},
                        fairness_metrics={},
                        severity_score=1 - reporting_ratio,
                        impact_assessment=f"Selective reporting detected (ratio: {reporting_ratio:.3f})",
                        mitigation_strategies=[
                            "Report all performed analyses",
                            "Use systematic review protocols",
                            "Pre-specify outcome measures",
                            "Provide analysis justifications"
                        ]
                    )
            
        except Exception as e:
            self.logger.error(f"Cognitive bias detection failed: {e}")
        
        return biases