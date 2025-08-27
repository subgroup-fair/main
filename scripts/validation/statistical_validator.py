"""
Statistical Validation Suite
Comprehensive statistical validation for research experiments
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import stats as scipy_stats
from scipy.stats import norm, t, chi2, f_oneway, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json
import math

# Statistical libraries with fallbacks
try:
    from statsmodels.stats.power import TTestPower, FTestAnovaPower
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.meta_analysis import effectsize_smd
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - some statistical tests will be limited")

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

@dataclass
class StatisticalTest:
    """Results of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    assumptions_met: bool = True
    assumptions_checked: List[str] = None
    power: Optional[float] = None
    sample_size_adequate: bool = True

@dataclass
class ConsistencyCheck:
    """Results of consistency checking across runs"""
    metric_name: str
    values: List[float]
    mean: float
    std: float
    coefficient_of_variation: float
    consistency_score: float  # 0-1 scale
    is_consistent: bool
    outlier_runs: List[int]
    trend_analysis: Dict[str, Any]

@dataclass
class EffectSizeResult:
    """Effect size calculation result"""
    effect_type: str  # 'cohens_d', 'eta_squared', 'cramers_v', etc.
    effect_size: float
    magnitude: str  # 'negligible', 'small', 'medium', 'large'
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""

@dataclass
class PowerAnalysis:
    """Power analysis results"""
    statistical_power: float
    effect_size: float
    sample_size: int
    alpha_level: float
    power_adequate: bool  # >= 0.8
    minimum_sample_size: Optional[int] = None
    minimum_effect_size: Optional[float] = None
    power_curve_data: Optional[Dict] = None

class ResultsConsistencyChecker:
    """Check consistency of results across multiple experimental runs"""
    
    def __init__(self, consistency_threshold: float = 0.15):
        self.consistency_threshold = consistency_threshold
        self.logger = logging.getLogger("consistency_checker")
    
    def check_consistency(self, results: Dict[str, List[float]], 
                         experiment_names: List[str] = None) -> Dict[str, ConsistencyCheck]:
        """Check consistency across multiple runs for all metrics"""
        
        consistency_results = {}
        
        for metric_name, values in results.items():
            if not values or len(values) < 2:
                self.logger.warning(f"Insufficient data for consistency check of {metric_name}")
                continue
            
            # Remove any None/NaN values
            clean_values = [v for v in values if v is not None and not np.isnan(v)]
            
            if len(clean_values) < 2:
                continue
            
            # Basic statistics
            mean_val = np.mean(clean_values)
            std_val = np.std(clean_values)
            cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
            
            # Consistency score (inverse of coefficient of variation, capped at 1)
            consistency_score = max(0, 1 - (cv / 0.5))  # CV of 0.5 = score of 0.5
            
            # Check if consistent (CV below threshold)
            is_consistent = cv <= self.consistency_threshold
            
            # Outlier detection using IQR
            outlier_runs = self._detect_outliers(clean_values)
            
            # Trend analysis
            trend_analysis = self._analyze_trend(clean_values, experiment_names)
            
            consistency_results[metric_name] = ConsistencyCheck(
                metric_name=metric_name,
                values=clean_values,
                mean=mean_val,
                std=std_val,
                coefficient_of_variation=cv,
                consistency_score=consistency_score,
                is_consistent=is_consistent,
                outlier_runs=outlier_runs,
                trend_analysis=trend_analysis
            )
        
        return consistency_results
    
    def _detect_outliers(self, values: List[float]) -> List[int]:
        """Detect outlier runs using IQR method"""
        
        if len(values) < 4:
            return []
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for i, val in enumerate(values):
            if val < lower_bound or val > upper_bound:
                outliers.append(i)
        
        return outliers
    
    def _analyze_trend(self, values: List[float], 
                      experiment_names: List[str] = None) -> Dict[str, Any]:
        """Analyze trends in results over runs"""
        
        if len(values) < 3:
            return {"trend": "insufficient_data"}
        
        # Linear trend analysis
        x = np.arange(len(values))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Classify trend
            if abs(slope) < std_err:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            return {
                "trend": trend,
                "slope": slope,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "trend_strength": abs(r_value),
                "is_significant": p_value < 0.05
            }
        except:
            return {"trend": "analysis_failed"}

class StatisticalSignificanceTester:
    """Comprehensive statistical significance testing"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.logger = logging.getLogger("significance_tester")
    
    def test_difference(self, group1: List[float], group2: List[float], 
                       test_type: str = "auto", paired: bool = False) -> StatisticalTest:
        """Test for significant difference between two groups"""
        
        # Clean data
        group1 = [x for x in group1 if x is not None and not np.isnan(x)]
        group2 = [x for x in group2 if x is not None and not np.isnan(x)]
        
        if len(group1) < 2 or len(group2) < 2:
            return StatisticalTest(
                test_name="insufficient_data",
                statistic=0,
                p_value=1,
                interpretation="Insufficient data for testing"
            )
        
        # Check assumptions
        assumptions = self._check_assumptions(group1, group2)
        
        # Select appropriate test
        if test_type == "auto":
            test_type = self._select_test(group1, group2, paired, assumptions)
        
        # Perform test
        if test_type == "t_test_independent":
            statistic, p_value = ttest_ind(group1, group2)
            test_name = "Independent t-test"
            
        elif test_type == "t_test_paired":
            if len(group1) != len(group2):
                return StatisticalTest(
                    test_name="error",
                    statistic=0,
                    p_value=1,
                    interpretation="Paired test requires equal sample sizes"
                )
            statistic, p_value = ttest_rel(group1, group2)
            test_name = "Paired t-test"
            
        elif test_type == "mann_whitney":
            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            
        elif test_type == "wilcoxon":
            if len(group1) != len(group2):
                return StatisticalTest(
                    test_name="error",
                    statistic=0,
                    p_value=1,
                    interpretation="Wilcoxon test requires equal sample sizes"
                )
            statistic, p_value = wilcoxon(group1, group2)
            test_name = "Wilcoxon signed-rank test"
            
        else:
            return StatisticalTest(
                test_name="unknown_test",
                statistic=0,
                p_value=1,
                interpretation=f"Unknown test type: {test_type}"
            )
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(group1, group2, test_type)
        
        # Calculate confidence interval
        ci = self._calculate_confidence_interval(group1, group2, test_type)
        
        # Interpret results
        interpretation = self._interpret_test_result(p_value, effect_size, test_name)
        
        return StatisticalTest(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
            assumptions_met=assumptions["all_met"],
            assumptions_checked=assumptions["checked"]
        )
    
    def test_multiple_groups(self, groups: List[List[float]], 
                           group_names: List[str] = None) -> StatisticalTest:
        """Test for significant differences among multiple groups (ANOVA)"""
        
        # Clean groups
        cleaned_groups = []
        for group in groups:
            cleaned = [x for x in group if x is not None and not np.isnan(x)]
            if len(cleaned) >= 2:
                cleaned_groups.append(cleaned)
        
        if len(cleaned_groups) < 2:
            return StatisticalTest(
                test_name="insufficient_groups",
                statistic=0,
                p_value=1,
                interpretation="Need at least 2 groups with sufficient data"
            )
        
        # Check assumptions
        assumptions = self._check_anova_assumptions(cleaned_groups)
        
        # Perform ANOVA
        try:
            statistic, p_value = f_oneway(*cleaned_groups)
            
            # Calculate effect size (eta-squared)
            effect_size = self._calculate_eta_squared(cleaned_groups, statistic)
            
            interpretation = self._interpret_anova_result(p_value, effect_size, len(cleaned_groups))
            
            return StatisticalTest(
                test_name="One-way ANOVA",
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=effect_size,
                interpretation=interpretation,
                assumptions_met=assumptions["all_met"],
                assumptions_checked=assumptions["checked"]
            )
            
        except Exception as e:
            return StatisticalTest(
                test_name="anova_error",
                statistic=0,
                p_value=1,
                interpretation=f"ANOVA failed: {str(e)}"
            )
    
    def _check_assumptions(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Check statistical test assumptions"""
        
        assumptions = {
            "normality_group1": self._test_normality(group1),
            "normality_group2": self._test_normality(group2),
            "equal_variances": self._test_equal_variances(group1, group2),
            "independence": True,  # Assume independence (cannot test automatically)
            "checked": ["normality", "equal_variances", "independence"]
        }
        
        assumptions["all_met"] = (
            assumptions["normality_group1"] and 
            assumptions["normality_group2"] and 
            assumptions["equal_variances"]
        )
        
        return assumptions
    
    def _test_normality(self, data: List[float]) -> bool:
        """Test normality using Shapiro-Wilk test"""
        
        if len(data) < 3:
            return True  # Assume normal for small samples
        
        if len(data) > 5000:
            # For large samples, use random sample
            data = np.random.choice(data, 5000, replace=False)
        
        try:
            _, p_value = stats.shapiro(data)
            return p_value > 0.05
        except:
            return True  # Assume normal if test fails
    
    def _test_equal_variances(self, group1: List[float], group2: List[float]) -> bool:
        """Test equal variances using Levene's test"""
        
        try:
            _, p_value = stats.levene(group1, group2)
            return p_value > 0.05
        except:
            return True  # Assume equal variances if test fails
    
    def _select_test(self, group1: List[float], group2: List[float], 
                    paired: bool, assumptions: Dict[str, Any]) -> str:
        """Select appropriate statistical test based on data and assumptions"""
        
        if paired:
            if assumptions["normality_group1"] and assumptions["normality_group2"]:
                return "t_test_paired"
            else:
                return "wilcoxon"
        else:
            if (assumptions["normality_group1"] and assumptions["normality_group2"] 
                and assumptions["equal_variances"]):
                return "t_test_independent"
            else:
                return "mann_whitney"
    
    def _calculate_effect_size(self, group1: List[float], group2: List[float], 
                              test_type: str) -> Optional[float]:
        """Calculate effect size for two-group comparison"""
        
        try:
            if test_type in ["t_test_independent", "t_test_paired"]:
                # Cohen's d
                mean1, mean2 = np.mean(group1), np.mean(group2)
                std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
                
                if test_type == "t_test_paired":
                    # For paired data, use standard deviation of differences
                    differences = np.array(group1) - np.array(group2)
                    pooled_std = np.std(differences, ddof=1)
                else:
                    # Pooled standard deviation
                    n1, n2 = len(group1), len(group2)
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                
                cohens_d = (mean1 - mean2) / pooled_std
                return abs(cohens_d)
                
            elif test_type == "mann_whitney":
                # Glass's delta approximation
                n1, n2 = len(group1), len(group2)
                u_statistic = mannwhitneyu(group1, group2)[0]
                r = 1 - (2 * u_statistic) / (n1 * n2)  # Effect size r
                return abs(r)
                
            else:
                return None
                
        except Exception:
            return None
    
    def _calculate_confidence_interval(self, group1: List[float], group2: List[float], 
                                     test_type: str) -> Optional[Tuple[float, float]]:
        """Calculate confidence interval for difference in means"""
        
        try:
            if test_type == "t_test_independent":
                mean1, mean2 = np.mean(group1), np.mean(group2)
                std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
                n1, n2 = len(group1), len(group2)
                
                # Standard error
                se = np.sqrt(std1**2/n1 + std2**2/n2)
                
                # Degrees of freedom (Welch's approximation)
                df = ((std1**2/n1 + std2**2/n2)**2) / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
                
                # Critical value
                t_critical = stats.t.ppf(1 - self.alpha/2, df)
                
                # Difference in means
                diff = mean1 - mean2
                
                # Confidence interval
                ci_lower = diff - t_critical * se
                ci_upper = diff + t_critical * se
                
                return (ci_lower, ci_upper)
            else:
                return None
                
        except Exception:
            return None

class EffectSizeCalculator:
    """Calculate various effect sizes for different types of analyses"""
    
    def __init__(self):
        self.logger = logging.getLogger("effect_size_calculator")
    
    def cohens_d(self, group1: List[float], group2: List[float], 
                paired: bool = False) -> EffectSizeResult:
        """Calculate Cohen's d effect size"""
        
        try:
            mean1, mean2 = np.mean(group1), np.mean(group2)
            
            if paired:
                # For paired data
                differences = np.array(group1) - np.array(group2)
                d = np.mean(differences) / np.std(differences, ddof=1)
            else:
                # For independent groups
                n1, n2 = len(group1), len(group2)
                std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
                
                # Pooled standard deviation
                pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                
                d = (mean1 - mean2) / pooled_std
            
            # Magnitude interpretation
            d_abs = abs(d)
            if d_abs < 0.2:
                magnitude = "negligible"
            elif d_abs < 0.5:
                magnitude = "small"
            elif d_abs < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            interpretation = f"Cohen's d = {d:.3f} ({magnitude} effect)"
            
            return EffectSizeResult(
                effect_type="cohens_d",
                effect_size=d,
                magnitude=magnitude,
                interpretation=interpretation
            )
            
        except Exception as e:
            return EffectSizeResult(
                effect_type="cohens_d",
                effect_size=0,
                magnitude="error",
                interpretation=f"Error calculating Cohen's d: {str(e)}"
            )
    
    def eta_squared(self, groups: List[List[float]]) -> EffectSizeResult:
        """Calculate eta-squared effect size for ANOVA"""
        
        try:
            # Flatten all data
            all_data = [x for group in groups for x in group]
            overall_mean = np.mean(all_data)
            
            # Sum of squares between groups (SSB)
            ssb = sum(len(group) * (np.mean(group) - overall_mean)**2 for group in groups)
            
            # Sum of squares within groups (SSW)
            ssw = sum(sum((x - np.mean(group))**2 for x in group) for group in groups)
            
            # Total sum of squares
            sst = ssb + ssw
            
            # Eta-squared
            eta_sq = ssb / sst if sst > 0 else 0
            
            # Magnitude interpretation
            if eta_sq < 0.01:
                magnitude = "negligible"
            elif eta_sq < 0.06:
                magnitude = "small"
            elif eta_sq < 0.14:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            interpretation = f"Eta-squared = {eta_sq:.3f} ({magnitude} effect)"
            
            return EffectSizeResult(
                effect_type="eta_squared",
                effect_size=eta_sq,
                magnitude=magnitude,
                interpretation=interpretation
            )
            
        except Exception as e:
            return EffectSizeResult(
                effect_type="eta_squared",
                effect_size=0,
                magnitude="error",
                interpretation=f"Error calculating eta-squared: {str(e)}"
            )
    
    def correlation_effect_size(self, r: float) -> EffectSizeResult:
        """Interpret correlation coefficient as effect size"""
        
        r_abs = abs(r)
        
        if r_abs < 0.1:
            magnitude = "negligible"
        elif r_abs < 0.3:
            magnitude = "small"
        elif r_abs < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        interpretation = f"Correlation r = {r:.3f} ({magnitude} effect)"
        
        return EffectSizeResult(
            effect_type="correlation",
            effect_size=r,
            magnitude=magnitude,
            interpretation=interpretation
        )

class PowerAnalysisValidator:
    """Validate statistical power and sample size adequacy"""
    
    def __init__(self, target_power: float = 0.8, alpha: float = 0.05):
        self.target_power = target_power
        self.alpha = alpha
        self.logger = logging.getLogger("power_analysis")
    
    def analyze_t_test_power(self, effect_size: float, sample_size: int, 
                           test_type: str = "two_sample") -> PowerAnalysis:
        """Analyze power for t-tests"""
        
        try:
            if STATSMODELS_AVAILABLE:
                power_analyzer = TTestPower()
                
                if test_type == "one_sample":
                    # One-sample t-test
                    power = power_analyzer.solve_power(
                        effect_size=effect_size,
                        nobs=sample_size,
                        alpha=self.alpha,
                        power=None
                    )
                    
                    # Minimum sample size for target power
                    min_n = power_analyzer.solve_power(
                        effect_size=effect_size,
                        nobs=None,
                        alpha=self.alpha,
                        power=self.target_power
                    )
                    
                else:
                    # Two-sample t-test
                    power = power_analyzer.solve_power(
                        effect_size=effect_size,
                        nobs1=sample_size//2,
                        alpha=self.alpha,
                        power=None
                    )
                    
                    min_n = power_analyzer.solve_power(
                        effect_size=effect_size,
                        nobs1=None,
                        alpha=self.alpha,
                        power=self.target_power
                    ) * 2
                
            else:
                # Fallback calculation
                power = self._calculate_t_test_power_fallback(effect_size, sample_size, test_type)
                min_n = self._calculate_min_sample_size_fallback(effect_size, test_type)
            
            return PowerAnalysis(
                statistical_power=float(power),
                effect_size=effect_size,
                sample_size=sample_size,
                alpha_level=self.alpha,
                power_adequate=power >= self.target_power,
                minimum_sample_size=int(min_n) if min_n else None
            )
            
        except Exception as e:
            self.logger.error(f"Power analysis failed: {e}")
            return PowerAnalysis(
                statistical_power=0.5,
                effect_size=effect_size,
                sample_size=sample_size,
                alpha_level=self.alpha,
                power_adequate=False
            )
    
    def analyze_anova_power(self, effect_size: float, sample_size: int, 
                          num_groups: int) -> PowerAnalysis:
        """Analyze power for ANOVA"""
        
        try:
            if STATSMODELS_AVAILABLE:
                power_analyzer = FTestAnovaPower()
                
                power = power_analyzer.solve_power(
                    effect_size=effect_size,
                    nobs=sample_size,
                    alpha=self.alpha,
                    k_groups=num_groups,
                    power=None
                )
                
                min_n = power_analyzer.solve_power(
                    effect_size=effect_size,
                    nobs=None,
                    alpha=self.alpha,
                    k_groups=num_groups,
                    power=self.target_power
                )
                
            else:
                # Fallback calculation
                power = self._calculate_anova_power_fallback(effect_size, sample_size, num_groups)
                min_n = self._calculate_min_anova_sample_size_fallback(effect_size, num_groups)
            
            return PowerAnalysis(
                statistical_power=float(power),
                effect_size=effect_size,
                sample_size=sample_size,
                alpha_level=self.alpha,
                power_adequate=power >= self.target_power,
                minimum_sample_size=int(min_n) if min_n else None
            )
            
        except Exception as e:
            self.logger.error(f"ANOVA power analysis failed: {e}")
            return PowerAnalysis(
                statistical_power=0.5,
                effect_size=effect_size,
                sample_size=sample_size,
                alpha_level=self.alpha,
                power_adequate=False
            )
    
    def _calculate_t_test_power_fallback(self, effect_size: float, sample_size: int, 
                                       test_type: str) -> float:
        """Fallback power calculation for t-tests"""
        
        if test_type == "one_sample":
            se = 1 / np.sqrt(sample_size)
        else:
            se = np.sqrt(2 / sample_size)  # Assuming equal group sizes
        
        # Non-centrality parameter
        ncp = effect_size / se
        
        # Critical value
        t_critical = stats.t.ppf(1 - self.alpha/2, sample_size - 1)
        
        # Power calculation
        power = 1 - stats.t.cdf(t_critical - ncp, sample_size - 1) + stats.t.cdf(-t_critical - ncp, sample_size - 1)
        
        return max(0, min(1, power))
    
    def generate_power_curve(self, effect_sizes: List[float], 
                           sample_sizes: List[int], test_type: str = "two_sample") -> Dict[str, Any]:
        """Generate power curve data for visualization"""
        
        power_data = {
            "effect_sizes": effect_sizes,
            "sample_sizes": sample_sizes,
            "power_matrix": []
        }
        
        for es in effect_sizes:
            power_row = []
            for n in sample_sizes:
                analysis = self.analyze_t_test_power(es, n, test_type)
                power_row.append(analysis.statistical_power)
            power_data["power_matrix"].append(power_row)
        
        return power_data

class StatisticalValidator:
    """Main statistical validation controller"""
    
    def __init__(self, alpha: float = 0.05, target_power: float = 0.8,
                 consistency_threshold: float = 0.15):
        self.alpha = alpha
        self.target_power = target_power
        self.consistency_threshold = consistency_threshold
        
        # Initialize components
        self.consistency_checker = ResultsConsistencyChecker(consistency_threshold)
        self.significance_tester = StatisticalSignificanceTester(alpha)
        self.effect_calculator = EffectSizeCalculator()
        self.power_analyzer = PowerAnalysisValidator(target_power, alpha)
        
        self.logger = logging.getLogger("statistical_validator")
    
    def validate_experiment_results(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive statistical validation of experimental results"""
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validation_config": {
                "alpha": self.alpha,
                "target_power": self.target_power,
                "consistency_threshold": self.consistency_threshold
            },
            "consistency_analysis": {},
            "significance_tests": {},
            "effect_sizes": {},
            "power_analysis": {},
            "overall_assessment": {}
        }
        
        # 1. Consistency checks across runs
        if "repeated_runs" in results_data:
            self.logger.info("Performing consistency analysis across runs")
            consistency_results = self.consistency_checker.check_consistency(
                results_data["repeated_runs"]
            )
            validation_results["consistency_analysis"] = self._serialize_consistency_results(consistency_results)
        
        # 2. Statistical significance testing
        if "comparisons" in results_data:
            self.logger.info("Performing statistical significance tests")
            significance_results = {}
            
            for comparison_name, comparison_data in results_data["comparisons"].items():
                if "group1" in comparison_data and "group2" in comparison_data:
                    test_result = self.significance_tester.test_difference(
                        comparison_data["group1"],
                        comparison_data["group2"],
                        comparison_data.get("test_type", "auto"),
                        comparison_data.get("paired", False)
                    )
                    significance_results[comparison_name] = self._serialize_test_result(test_result)
            
            validation_results["significance_tests"] = significance_results
        
        # 3. Effect size calculations
        if "effect_size_data" in results_data:
            self.logger.info("Calculating effect sizes")
            effect_results = {}
            
            for es_name, es_data in results_data["effect_size_data"].items():
                if es_data["type"] == "cohens_d":
                    effect_result = self.effect_calculator.cohens_d(
                        es_data["group1"], es_data["group2"], 
                        es_data.get("paired", False)
                    )
                elif es_data["type"] == "eta_squared":
                    effect_result = self.effect_calculator.eta_squared(es_data["groups"])
                elif es_data["type"] == "correlation":
                    effect_result = self.effect_calculator.correlation_effect_size(es_data["r"])
                else:
                    continue
                
                effect_results[es_name] = self._serialize_effect_size_result(effect_result)
            
            validation_results["effect_sizes"] = effect_results
        
        # 4. Power analysis
        if "power_analysis_data" in results_data:
            self.logger.info("Performing power analysis")
            power_results = {}
            
            for pa_name, pa_data in results_data["power_analysis_data"].items():
                if pa_data["test_type"] == "t_test":
                    power_result = self.power_analyzer.analyze_t_test_power(
                        pa_data["effect_size"], pa_data["sample_size"],
                        pa_data.get("variant", "two_sample")
                    )
                elif pa_data["test_type"] == "anova":
                    power_result = self.power_analyzer.analyze_anova_power(
                        pa_data["effect_size"], pa_data["sample_size"],
                        pa_data["num_groups"]
                    )
                else:
                    continue
                
                power_results[pa_name] = self._serialize_power_analysis(power_result)
            
            validation_results["power_analysis"] = power_results
        
        # 5. Overall assessment
        validation_results["overall_assessment"] = self._generate_overall_assessment(validation_results)
        
        return validation_results
    
    def _serialize_consistency_results(self, results: Dict[str, ConsistencyCheck]) -> Dict[str, Any]:
        """Serialize consistency check results"""
        
        serialized = {}
        for metric_name, check in results.items():
            serialized[metric_name] = {
                "mean": check.mean,
                "std": check.std,
                "coefficient_of_variation": check.coefficient_of_variation,
                "consistency_score": check.consistency_score,
                "is_consistent": check.is_consistent,
                "outlier_runs": check.outlier_runs,
                "trend_analysis": check.trend_analysis,
                "num_runs": len(check.values)
            }
        
        return serialized
    
    def _serialize_test_result(self, test: StatisticalTest) -> Dict[str, Any]:
        """Serialize statistical test result"""
        
        return {
            "test_name": test.test_name,
            "statistic": test.statistic,
            "p_value": test.p_value,
            "effect_size": test.effect_size,
            "confidence_interval": test.confidence_interval,
            "interpretation": test.interpretation,
            "assumptions_met": test.assumptions_met,
            "assumptions_checked": test.assumptions_checked,
            "is_significant": test.p_value < self.alpha,
            "power": test.power,
            "sample_size_adequate": test.sample_size_adequate
        }
    
    def _serialize_effect_size_result(self, result: EffectSizeResult) -> Dict[str, Any]:
        """Serialize effect size result"""
        
        return {
            "effect_type": result.effect_type,
            "effect_size": result.effect_size,
            "magnitude": result.magnitude,
            "confidence_interval": result.confidence_interval,
            "interpretation": result.interpretation
        }
    
    def _serialize_power_analysis(self, analysis: PowerAnalysis) -> Dict[str, Any]:
        """Serialize power analysis result"""
        
        return {
            "statistical_power": analysis.statistical_power,
            "effect_size": analysis.effect_size,
            "sample_size": analysis.sample_size,
            "alpha_level": analysis.alpha_level,
            "power_adequate": analysis.power_adequate,
            "minimum_sample_size": analysis.minimum_sample_size,
            "minimum_effect_size": analysis.minimum_effect_size
        }
    
    def _generate_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall statistical validation assessment"""
        
        assessment = {
            "overall_score": 0,
            "statistical_validity": "unknown",
            "key_issues": [],
            "strengths": [],
            "recommendations": []
        }
        
        # Scoring components
        consistency_score = 0
        significance_score = 0
        effect_size_score = 0
        power_score = 0
        
        # Consistency assessment
        if validation_results["consistency_analysis"]:
            consistent_metrics = sum(1 for res in validation_results["consistency_analysis"].values() 
                                   if res["is_consistent"])
            total_metrics = len(validation_results["consistency_analysis"])
            consistency_score = (consistent_metrics / total_metrics) * 25
            
            if consistency_score < 15:
                assessment["key_issues"].append("Poor consistency across experimental runs")
            else:
                assessment["strengths"].append("Good consistency across runs")
        
        # Significance testing assessment
        if validation_results["significance_tests"]:
            significant_tests = sum(1 for res in validation_results["significance_tests"].values() 
                                  if res["is_significant"])
            total_tests = len(validation_results["significance_tests"])
            significance_score = min(25, (significant_tests / total_tests) * 30)
            
            if significant_tests == 0:
                assessment["key_issues"].append("No statistically significant results found")
            else:
                assessment["strengths"].append(f"Found {significant_tests} significant results")
        
        # Effect size assessment
        if validation_results["effect_sizes"]:
            meaningful_effects = sum(1 for res in validation_results["effect_sizes"].values() 
                                   if res["magnitude"] in ["medium", "large"])
            total_effects = len(validation_results["effect_sizes"])
            effect_size_score = (meaningful_effects / total_effects) * 25
            
            if meaningful_effects == 0:
                assessment["key_issues"].append("All effect sizes are small or negligible")
            else:
                assessment["strengths"].append(f"Found {meaningful_effects} meaningful effect sizes")
        
        # Power assessment
        if validation_results["power_analysis"]:
            adequate_power = sum(1 for res in validation_results["power_analysis"].values() 
                               if res["power_adequate"])
            total_power = len(validation_results["power_analysis"])
            power_score = (adequate_power / total_power) * 25
            
            if adequate_power == 0:
                assessment["key_issues"].append("Inadequate statistical power")
                assessment["recommendations"].append("Increase sample size or effect size")
            else:
                assessment["strengths"].append("Adequate statistical power")
        
        # Overall score and validity
        assessment["overall_score"] = consistency_score + significance_score + effect_size_score + power_score
        
        if assessment["overall_score"] >= 80:
            assessment["statistical_validity"] = "excellent"
        elif assessment["overall_score"] >= 65:
            assessment["statistical_validity"] = "good"
        elif assessment["overall_score"] >= 50:
            assessment["statistical_validity"] = "acceptable"
        elif assessment["overall_score"] >= 35:
            assessment["statistical_validity"] = "poor"
        else:
            assessment["statistical_validity"] = "inadequate"
        
        return assessment