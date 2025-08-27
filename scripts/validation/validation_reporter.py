"""
Validation Reporting System
Comprehensive reporting for experiment validation results
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import base64

# Plotting and visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Statistical libraries
try:
    import numpy as np
    import pandas as pd
    PANDAS_NUMPY_AVAILABLE = True
except ImportError:
    PANDAS_NUMPY_AVAILABLE = False

@dataclass
class ScorecardMetric:
    """Individual scorecard metric"""
    name: str
    score: float
    max_score: float
    category: str
    description: str
    status: str  # 'excellent', 'good', 'needs_improvement', 'critical'
    recommendations: List[str] = None

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    experiment_name: str
    timestamp: datetime
    report_type: str
    content: Dict[str, Any]
    summary: Dict[str, Any]
    visualizations: Dict[str, str] = None  # Base64 encoded plots

class ExperimentScorecard:
    """Generate experiment quality scorecard"""
    
    def __init__(self):
        self.logger = logging.getLogger("experiment_scorecard")
    
    def generate_scorecard(self, validation_results: Dict[str, Any], 
                          experiment_name: str = "Unknown Experiment") -> Dict[str, Any]:
        """Generate comprehensive experiment quality scorecard"""
        
        scorecard = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0,
            "overall_grade": "F",
            "category_scores": {},
            "detailed_metrics": [],
            "recommendations": [],
            "strengths": [],
            "weaknesses": []
        }
        
        # Extract scores from validation results
        statistical_score = self._extract_statistical_score(validation_results)
        code_quality_score = self._extract_code_quality_score(validation_results)
        scientific_rigor_score = self._extract_scientific_rigor_score(validation_results)
        
        # Calculate category scores
        category_scores = {
            "statistical_validation": statistical_score,
            "code_quality": code_quality_score,
            "scientific_rigor": scientific_rigor_score
        }
        
        # Calculate overall score (weighted average)
        weights = {"statistical_validation": 0.4, "code_quality": 0.3, "scientific_rigor": 0.3}
        overall_score = sum(score * weights[category] for category, score in category_scores.items())
        
        # Determine overall grade
        overall_grade = self._calculate_grade(overall_score)
        
        # Generate detailed metrics
        detailed_metrics = []
        detailed_metrics.extend(self._generate_statistical_metrics(validation_results))
        detailed_metrics.extend(self._generate_code_quality_metrics(validation_results))
        detailed_metrics.extend(self._generate_scientific_rigor_metrics(validation_results))
        
        # Generate recommendations and identify strengths/weaknesses
        recommendations = self._generate_scorecard_recommendations(category_scores, detailed_metrics)
        strengths = self._identify_strengths(category_scores, detailed_metrics)
        weaknesses = self._identify_weaknesses(category_scores, detailed_metrics)
        
        scorecard.update({
            "overall_score": round(overall_score, 1),
            "overall_grade": overall_grade,
            "category_scores": category_scores,
            "detailed_metrics": [self._serialize_metric(m) for m in detailed_metrics],
            "recommendations": recommendations,
            "strengths": strengths,
            "weaknesses": weaknesses
        })
        
        return scorecard
    
    def _extract_statistical_score(self, validation_results: Dict[str, Any]) -> float:
        """Extract statistical validation score"""
        
        if "statistical_validation" in validation_results:
            overall_assessment = validation_results["statistical_validation"].get("overall_assessment", {})
            return overall_assessment.get("overall_score", 0)
        return 0
    
    def _extract_code_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Extract code quality score"""
        
        if "code_quality" in validation_results:
            overall_assessment = validation_results["code_quality"].get("overall_assessment", {})
            return overall_assessment.get("overall_score", 0)
        return 0
    
    def _extract_scientific_rigor_score(self, validation_results: Dict[str, Any]) -> float:
        """Extract scientific rigor score"""
        
        if "scientific_rigor" in validation_results:
            overall_assessment = validation_results["scientific_rigor"].get("overall_rigor_assessment", {})
            return overall_assessment.get("overall_rigor_score", 0)
        return 0
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
        
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_statistical_metrics(self, validation_results: Dict[str, Any]) -> List[ScorecardMetric]:
        """Generate statistical validation metrics"""
        
        metrics = []
        
        if "statistical_validation" not in validation_results:
            return metrics
        
        stat_results = validation_results["statistical_validation"]
        
        # Consistency metric
        if "consistency_analysis" in stat_results:
            consistency_data = stat_results["consistency_analysis"]
            consistent_metrics = sum(1 for metric in consistency_data.values() if metric.get("is_consistent", False))
            total_metrics = len(consistency_data)
            consistency_score = (consistent_metrics / total_metrics * 100) if total_metrics > 0 else 0
            
            metrics.append(ScorecardMetric(
                name="Results Consistency",
                score=consistency_score,
                max_score=100,
                category="statistical_validation",
                description=f"{consistent_metrics}/{total_metrics} metrics show consistent results",
                status=self._determine_status(consistency_score),
                recommendations=["Investigate inconsistent metrics", "Increase number of runs"] if consistency_score < 80 else []
            ))
        
        # Statistical significance metric
        if "significance_tests" in stat_results:
            sig_tests = stat_results["significance_tests"]
            significant_tests = sum(1 for test in sig_tests.values() if test.get("is_significant", False))
            total_tests = len(sig_tests)
            significance_score = min(100, (significant_tests / total_tests * 100) if total_tests > 0 else 0)
            
            metrics.append(ScorecardMetric(
                name="Statistical Significance",
                score=significance_score,
                max_score=100,
                category="statistical_validation",
                description=f"{significant_tests}/{total_tests} tests show significance",
                status=self._determine_status(significance_score),
                recommendations=["Review sample sizes", "Consider effect sizes"] if significance_score < 50 else []
            ))
        
        # Effect size metric
        if "effect_sizes" in stat_results:
            effect_sizes = stat_results["effect_sizes"]
            meaningful_effects = sum(1 for effect in effect_sizes.values() 
                                   if effect.get("magnitude") in ["medium", "large"])
            total_effects = len(effect_sizes)
            effect_size_score = (meaningful_effects / total_effects * 100) if total_effects > 0 else 0
            
            metrics.append(ScorecardMetric(
                name="Effect Size Magnitude",
                score=effect_size_score,
                max_score=100,
                category="statistical_validation",
                description=f"{meaningful_effects}/{total_effects} effects are medium or large",
                status=self._determine_status(effect_size_score),
                recommendations=["Consider practical significance", "Report confidence intervals"] if effect_size_score < 50 else []
            ))
        
        return metrics
    
    def _generate_code_quality_metrics(self, validation_results: Dict[str, Any]) -> List[ScorecardMetric]:
        """Generate code quality metrics"""
        
        metrics = []
        
        if "code_quality" not in validation_results:
            return metrics
        
        code_results = validation_results["code_quality"]
        
        # Code review metric
        if "code_review" in code_results:
            review_data = code_results["code_review"]
            critical_issues = review_data["issues_by_severity"].get("critical", 0)
            high_issues = review_data["issues_by_severity"].get("high", 0)
            total_issues = review_data.get("total_issues", 0)
            
            # Score based on severity-weighted issues
            penalty = critical_issues * 20 + high_issues * 10 + (total_issues - critical_issues - high_issues) * 2
            review_score = max(0, 100 - penalty)
            
            metrics.append(ScorecardMetric(
                name="Code Review Score",
                score=review_score,
                max_score=100,
                category="code_quality",
                description=f"Total issues: {total_issues} (Critical: {critical_issues}, High: {high_issues})",
                status=self._determine_status(review_score),
                recommendations=["Address critical issues first", "Improve code documentation"] if review_score < 70 else []
            ))
        
        # Security metric
        if "security_scan" in code_results:
            security_data = code_results["security_scan"]
            security_score = security_data.get("security_score", 100)
            
            metrics.append(ScorecardMetric(
                name="Security Score",
                score=security_score,
                max_score=100,
                category="code_quality",
                description=f"Security risk level: {security_data.get('risk_level', 'unknown')}",
                status=self._determine_status(security_score),
                recommendations=security_data.get("recommendations", []) if security_score < 90 else []
            ))
        
        # Performance metric
        if "performance_analysis" in code_results:
            performance_data = code_results["performance_analysis"]
            performance_score = performance_data.get("overall_performance_score", 70)
            
            metrics.append(ScorecardMetric(
                name="Performance Score",
                score=performance_score,
                max_score=100,
                category="code_quality",
                description="Overall code performance assessment",
                status=self._determine_status(performance_score),
                recommendations=["Optimize bottlenecks", "Profile critical sections"] if performance_score < 70 else []
            ))
        
        return metrics
    
    def _generate_scientific_rigor_metrics(self, validation_results: Dict[str, Any]) -> List[ScorecardMetric]:
        """Generate scientific rigor metrics"""
        
        metrics = []
        
        if "scientific_rigor" not in validation_results:
            return metrics
        
        rigor_results = validation_results["scientific_rigor"]
        
        # Methodology compliance metric
        if "methodology_compliance" in rigor_results:
            methodology_data = rigor_results["methodology_compliance"]
            methodology_score = methodology_data.get("overall_compliance_score", 0)
            
            metrics.append(ScorecardMetric(
                name="Methodology Compliance",
                score=methodology_score,
                max_score=100,
                category="scientific_rigor",
                description=f"Compliance level: {methodology_data.get('compliance_level', 'unknown')}",
                status=self._determine_status(methodology_score),
                recommendations=methodology_data.get("recommendations", []) if methodology_score < 80 else []
            ))
        
        # Reproducibility metric
        if "reproducibility_assessment" in rigor_results:
            repro_data = rigor_results["reproducibility_assessment"]
            repro_score = repro_data.get("reproducibility_score", 0)
            
            metrics.append(ScorecardMetric(
                name="Reproducibility",
                score=repro_score,
                max_score=100,
                category="scientific_rigor",
                description=f"Reproducibility level: {repro_data.get('reproducibility_level', 'unknown')}",
                status=self._determine_status(repro_score),
                recommendations=repro_data.get("requirements", []) if repro_score < 80 else []
            ))
        
        # Bias detection metric
        if "bias_detection" in rigor_results:
            bias_data = rigor_results["bias_detection"]
            total_biases = bias_data.get("total_biases_detected", 0)
            critical_biases = bias_data.get("biases_by_severity", {}).get("critical", 0)
            high_biases = bias_data.get("biases_by_severity", {}).get("high", 0)
            
            # Score based on bias severity
            bias_penalty = critical_biases * 30 + high_biases * 15 + (total_biases - critical_biases - high_biases) * 5
            bias_score = max(0, 100 - bias_penalty)
            
            metrics.append(ScorecardMetric(
                name="Bias Assessment",
                score=bias_score,
                max_score=100,
                category="scientific_rigor",
                description=f"Total biases detected: {total_biases} (Critical: {critical_biases}, High: {high_biases})",
                status=self._determine_status(bias_score),
                recommendations=["Address critical biases", "Review experimental design"] if bias_score < 70 else []
            ))
        
        return metrics
    
    def _determine_status(self, score: float) -> str:
        """Determine status based on score"""
        
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "needs_improvement"
        else:
            return "critical"
    
    def _serialize_metric(self, metric: ScorecardMetric) -> Dict[str, Any]:
        """Serialize scorecard metric"""
        
        return {
            "name": metric.name,
            "score": metric.score,
            "max_score": metric.max_score,
            "category": metric.category,
            "description": metric.description,
            "status": metric.status,
            "recommendations": metric.recommendations or []
        }

class ComplianceReporter:
    """Generate methodology compliance reports"""
    
    def __init__(self):
        self.logger = logging.getLogger("compliance_reporter")
    
    def generate_compliance_report(self, validation_results: Dict[str, Any],
                                  experiment_name: str = "Unknown Experiment") -> Dict[str, Any]:
        """Generate comprehensive methodology compliance report"""
        
        if "scientific_rigor" not in validation_results or "methodology_compliance" not in validation_results["scientific_rigor"]:
            return {"error": "No methodology compliance data available"}
        
        compliance_data = validation_results["scientific_rigor"]["methodology_compliance"]
        
        report = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "compliance_summary": {
                "overall_score": compliance_data.get("overall_compliance_score", 0),
                "compliance_level": compliance_data.get("compliance_level", "unknown"),
                "total_violations": len(compliance_data.get("violations", [])),
                "violations_by_severity": self._categorize_violations_by_severity(compliance_data.get("violations", []))
            },
            "detailed_violations": compliance_data.get("violations", []),
            "compliance_checklist": self._generate_compliance_checklist(compliance_data),
            "improvement_plan": self._generate_improvement_plan(compliance_data),
            "regulatory_considerations": self._generate_regulatory_considerations(compliance_data)
        }
        
        return report
    
    def _categorize_violations_by_severity(self, violations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize violations by severity"""
        
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for violation in violations:
            severity = violation.get("severity", "medium")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return severity_counts
    
    def _generate_compliance_checklist(self, compliance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate compliance checklist"""
        
        checklist = []
        violations = compliance_data.get("violations", [])
        
        # Standard compliance items
        standard_items = [
            {"item": "Research question clearly defined", "category": "design"},
            {"item": "Appropriate sample size", "category": "design"},
            {"item": "Control group included", "category": "design"},
            {"item": "Randomization implemented", "category": "design"},
            {"item": "Statistical test appropriate", "category": "analysis"},
            {"item": "Effect size calculated", "category": "analysis"},
            {"item": "Multiple comparison correction", "category": "analysis"},
            {"item": "Data validation performed", "category": "data"},
            {"item": "Missing data handled", "category": "data"},
            {"item": "Outliers addressed", "category": "data"}
        ]
        
        # Check each item against violations
        violation_types = {v.get("violation_type", "") for v in violations}
        
        for item in standard_items:
            item_name = item["item"]
            item_category = item["category"]
            
            # Determine status based on violations
            related_violations = [v for v in violations if self._is_related_violation(v, item)]
            
            if related_violations:
                status = "non_compliant"
                issues = [v.get("current_state", "") for v in related_violations]
                recommendations = [v.get("recommendation", "") for v in related_violations]
            else:
                status = "compliant"
                issues = []
                recommendations = []
            
            checklist.append({
                "item": item_name,
                "category": item_category,
                "status": status,
                "issues": issues,
                "recommendations": recommendations
            })
        
        return checklist
    
    def _is_related_violation(self, violation: Dict[str, Any], checklist_item: Dict[str, Any]) -> bool:
        """Check if violation is related to checklist item"""
        
        violation_type = violation.get("violation_type", "").lower()
        item_name = checklist_item["item"].lower()
        
        # Simple keyword matching
        if "hypothesis" in item_name and "hypothesis" in violation_type:
            return True
        elif "sample" in item_name and "sample" in violation_type:
            return True
        elif "control" in item_name and "control" in violation_type:
            return True
        elif "random" in item_name and "random" in violation_type:
            return True
        elif "statistical" in item_name and "statistical" in violation_type:
            return True
        elif "effect" in item_name and "effect" in violation_type:
            return True
        elif "comparison" in item_name and "comparison" in violation_type:
            return True
        elif "data" in item_name and ("data" in violation_type or "missing" in violation_type):
            return True
        
        return False

class PowerAnalysisReporter:
    """Generate statistical power analysis reports"""
    
    def __init__(self):
        self.logger = logging.getLogger("power_analysis_reporter")
    
    def generate_power_report(self, validation_results: Dict[str, Any],
                             experiment_name: str = "Unknown Experiment") -> Dict[str, Any]:
        """Generate comprehensive power analysis report"""
        
        if ("statistical_validation" not in validation_results or 
            "power_analysis" not in validation_results["statistical_validation"]):
            return {"error": "No power analysis data available"}
        
        power_data = validation_results["statistical_validation"]["power_analysis"]
        
        report = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "power_summary": self._generate_power_summary(power_data),
            "detailed_analyses": power_data,
            "sample_size_recommendations": self._generate_sample_size_recommendations(power_data),
            "power_interpretations": self._generate_power_interpretations(power_data),
            "limitations_and_assumptions": self._generate_power_limitations()
        }
        
        return report
    
    def _generate_power_summary(self, power_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate power analysis summary"""
        
        summary = {
            "total_analyses": len(power_data),
            "adequate_power_count": 0,
            "average_power": 0,
            "power_status": "unknown"
        }
        
        if not power_data:
            return summary
        
        powers = []
        adequate_count = 0
        
        for analysis_name, analysis_data in power_data.items():
            power = analysis_data.get("statistical_power", 0)
            powers.append(power)
            
            if analysis_data.get("power_adequate", False):
                adequate_count += 1
        
        summary.update({
            "adequate_power_count": adequate_count,
            "average_power": sum(powers) / len(powers) if powers else 0,
            "power_status": "adequate" if adequate_count == len(power_data) else "inadequate"
        })
        
        return summary

class ReproducibilityAssessor:
    """Generate reproducibility assessment reports"""
    
    def __init__(self):
        self.logger = logging.getLogger("reproducibility_assessor")
    
    def generate_reproducibility_report(self, validation_results: Dict[str, Any],
                                      experiment_name: str = "Unknown Experiment") -> Dict[str, Any]:
        """Generate comprehensive reproducibility assessment report"""
        
        if ("scientific_rigor" not in validation_results or 
            "reproducibility_assessment" not in validation_results["scientific_rigor"]):
            return {"error": "No reproducibility assessment data available"}
        
        repro_data = validation_results["scientific_rigor"]["reproducibility_assessment"]
        
        report = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "reproducibility_summary": {
                "overall_score": repro_data.get("reproducibility_score", 0),
                "reproducibility_level": repro_data.get("reproducibility_level", "unknown"),
                "total_issues": len(repro_data.get("issues", [])),
                "critical_issues": len([i for i in repro_data.get("issues", []) if i.get("severity") == "high"])
            },
            "reproducibility_checklist": self._generate_reproducibility_checklist(repro_data),
            "improvement_roadmap": self._generate_improvement_roadmap(repro_data),
            "best_practices_guide": self._generate_best_practices_guide()
        }
        
        return report
    
    def _generate_reproducibility_checklist(self, repro_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate reproducibility checklist"""
        
        checklist_items = [
            {"item": "Random seed set", "category": "code", "critical": True},
            {"item": "Dependencies documented", "category": "environment", "critical": True},
            {"item": "Code version controlled", "category": "code", "critical": False},
            {"item": "Data preprocessing documented", "category": "data", "critical": True},
            {"item": "Computational environment documented", "category": "environment", "critical": False},
            {"item": "Parameters explicitly specified", "category": "configuration", "critical": True}
        ]
        
        issues = repro_data.get("issues", [])
        issue_types = {issue.get("type", "") for issue in issues}
        
        checklist = []
        for item in checklist_items:
            # Check if there's a related issue
            has_issue = any(self._is_related_repro_issue(issue_type, item["item"]) for issue_type in issue_types)
            
            checklist.append({
                "item": item["item"],
                "category": item["category"],
                "status": "fail" if has_issue else "pass",
                "critical": item["critical"],
                "importance": "critical" if item["critical"] else "recommended"
            })
        
        return checklist
    
    def _is_related_repro_issue(self, issue_type: str, checklist_item: str) -> bool:
        """Check if issue type relates to checklist item"""
        
        mappings = {
            "random": "Random seed set",
            "seed": "Random seed set", 
            "dependency": "Dependencies documented",
            "requirements": "Dependencies documented",
            "version": "Code version controlled",
            "preprocessing": "Data preprocessing documented",
            "environment": "Computational environment documented",
            "parameter": "Parameters explicitly specified"
        }
        
        for key, item in mappings.items():
            if key in issue_type.lower() and item == checklist_item:
                return True
        
        return False

class ValidationReporter:
    """Main validation reporting controller"""
    
    def __init__(self, output_dir: str = "validation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger("validation_reporter")
        
        # Initialize sub-reporters
        self.scorecard_generator = ExperimentScorecard()
        self.compliance_reporter = ComplianceReporter()
        self.power_reporter = PowerAnalysisReporter()
        self.reproducibility_assessor = ReproducibilityAssessor()
    
    def generate_comprehensive_report(self, validation_results: Dict[str, Any],
                                    experiment_name: str = "Unknown Experiment",
                                    include_visualizations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        report_id = f"{experiment_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        comprehensive_report = {
            "report_metadata": {
                "report_id": report_id,
                "experiment_name": experiment_name,
                "generation_timestamp": datetime.now().isoformat(),
                "report_version": "1.0.0"
            },
            "executive_summary": self._generate_executive_summary(validation_results),
            "experiment_scorecard": self.scorecard_generator.generate_scorecard(validation_results, experiment_name),
            "methodology_compliance": self.compliance_reporter.generate_compliance_report(validation_results, experiment_name),
            "statistical_power_analysis": self.power_reporter.generate_power_report(validation_results, experiment_name),
            "reproducibility_assessment": self.reproducibility_assessor.generate_reproducibility_report(validation_results, experiment_name),
            "detailed_findings": self._extract_detailed_findings(validation_results),
            "action_plan": self._generate_action_plan(validation_results)
        }
        
        # Add visualizations if requested
        if include_visualizations and PLOTTING_AVAILABLE:
            visualizations = self._generate_visualizations(validation_results, experiment_name)
            comprehensive_report["visualizations"] = visualizations
        
        # Save report in multiple formats
        self._save_report_multiple_formats(comprehensive_report, report_id)
        
        return comprehensive_report
    
    def _generate_executive_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        
        summary = {
            "overall_assessment": "unknown",
            "key_strengths": [],
            "critical_issues": [],
            "primary_recommendations": [],
            "validation_status": {}
        }
        
        # Extract overall scores
        scores = {}
        if "statistical_validation" in validation_results:
            scores["statistical"] = validation_results["statistical_validation"].get("overall_assessment", {}).get("overall_score", 0)
        
        if "code_quality" in validation_results:
            scores["code_quality"] = validation_results["code_quality"].get("overall_assessment", {}).get("overall_score", 0)
        
        if "scientific_rigor" in validation_results:
            scores["scientific_rigor"] = validation_results["scientific_rigor"].get("overall_rigor_assessment", {}).get("overall_rigor_score", 0)
        
        # Calculate overall assessment
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            if avg_score >= 85:
                summary["overall_assessment"] = "excellent"
            elif avg_score >= 75:
                summary["overall_assessment"] = "good"
            elif avg_score >= 65:
                summary["overall_assessment"] = "acceptable"
            else:
                summary["overall_assessment"] = "needs_improvement"
        
        # Identify strengths and issues
        for category, score in scores.items():
            if score >= 80:
                summary["key_strengths"].append(f"Strong {category.replace('_', ' ')}")
            elif score < 60:
                summary["critical_issues"].append(f"Poor {category.replace('_', ' ')}")
        
        summary["validation_status"] = scores
        
        return summary
    
    def _extract_detailed_findings(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed findings from validation results"""
        
        findings = {
            "statistical_findings": {},
            "code_quality_findings": {},
            "scientific_rigor_findings": {}
        }
        
        # Statistical findings
        if "statistical_validation" in validation_results:
            stat_data = validation_results["statistical_validation"]
            findings["statistical_findings"] = {
                "consistency_issues": len([m for m in stat_data.get("consistency_analysis", {}).values() 
                                         if not m.get("is_consistent", True)]),
                "significant_results": len([t for t in stat_data.get("significance_tests", {}).values() 
                                          if t.get("is_significant", False)]),
                "effect_sizes": {name: data.get("magnitude", "unknown") 
                               for name, data in stat_data.get("effect_sizes", {}).items()}
            }
        
        # Code quality findings
        if "code_quality" in validation_results:
            code_data = validation_results["code_quality"]
            findings["code_quality_findings"] = {
                "total_code_issues": code_data.get("code_review", {}).get("total_issues", 0),
                "security_score": code_data.get("security_scan", {}).get("security_score", 100),
                "performance_bottlenecks": len(code_data.get("performance_analysis", {}).get("bottlenecks", []))
            }
        
        # Scientific rigor findings
        if "scientific_rigor" in validation_results:
            rigor_data = validation_results["scientific_rigor"]
            findings["scientific_rigor_findings"] = {
                "methodology_violations": len(rigor_data.get("methodology_compliance", {}).get("violations", [])),
                "detected_biases": rigor_data.get("bias_detection", {}).get("total_biases_detected", 0),
                "reproducibility_issues": len(rigor_data.get("reproducibility_assessment", {}).get("issues", []))
            }
        
        return findings
    
    def _generate_action_plan(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prioritized action plan"""
        
        action_plan = {
            "immediate_actions": [],
            "short_term_improvements": [],
            "long_term_goals": [],
            "timeline": "4-6 weeks for immediate actions, 2-3 months for improvements"
        }
        
        # Collect all recommendations
        all_recommendations = []
        
        # From scorecard
        scorecard = self.scorecard_generator.generate_scorecard(validation_results)
        all_recommendations.extend(scorecard.get("recommendations", []))
        
        # From compliance report
        compliance = self.compliance_reporter.generate_compliance_report(validation_results)
        if "improvement_plan" in compliance:
            all_recommendations.extend(compliance["improvement_plan"])
        
        # Prioritize actions
        high_priority = ["critical", "security", "methodology", "reproducibility"]
        medium_priority = ["performance", "documentation", "testing"]
        
        for rec in all_recommendations:
            rec_lower = rec.lower() if isinstance(rec, str) else ""
            
            if any(keyword in rec_lower for keyword in high_priority):
                action_plan["immediate_actions"].append(rec)
            elif any(keyword in rec_lower for keyword in medium_priority):
                action_plan["short_term_improvements"].append(rec)
            else:
                action_plan["long_term_goals"].append(rec)
        
        return action_plan
    
    def _generate_visualizations(self, validation_results: Dict[str, Any], 
                               experiment_name: str) -> Dict[str, str]:
        """Generate visualizations for the report"""
        
        visualizations = {}
        
        try:
            # Score overview chart
            scorecard = self.scorecard_generator.generate_scorecard(validation_results, experiment_name)
            scores_chart = self._create_scores_overview_chart(scorecard)
            if scores_chart:
                visualizations["scores_overview"] = scores_chart
            
            # Compliance radar chart
            compliance_chart = self._create_compliance_radar_chart(validation_results)
            if compliance_chart:
                visualizations["compliance_radar"] = compliance_chart
            
            # Issues summary chart
            issues_chart = self._create_issues_summary_chart(validation_results)
            if issues_chart:
                visualizations["issues_summary"] = issues_chart
                
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _create_scores_overview_chart(self, scorecard: Dict[str, Any]) -> Optional[str]:
        """Create scores overview chart"""
        
        try:
            categories = list(scorecard["category_scores"].keys())
            scores = list(scorecard["category_scores"].values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=scores,
                    marker_color=['green' if s >= 80 else 'yellow' if s >= 60 else 'red' for s in scores],
                    text=[f"{s:.1f}" for s in scores],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Validation Scores Overview",
                xaxis_title="Category",
                yaxis_title="Score",
                yaxis=dict(range=[0, 100])
            )
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating scores chart: {e}")
            return None
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 string"""
        
        try:
            img_bytes = fig.to_image(format="png", width=800, height=600)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64
        except Exception as e:
            self.logger.error(f"Error converting figure to base64: {e}")
            return ""
    
    def _save_report_multiple_formats(self, report: Dict[str, Any], report_id: str):
        """Save report in multiple formats"""
        
        # JSON format
        json_file = self.output_dir / f"{report_id}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # HTML format
        html_file = self.output_dir / f"{report_id}.html"
        self._generate_html_report(report, html_file)
        
        # CSV summary
        csv_file = self.output_dir / f"{report_id}_summary.csv"
        self._generate_csv_summary(report, csv_file)
        
        self.logger.info(f"Report saved in multiple formats: {json_file}, {html_file}, {csv_file}")
    
    def _generate_html_report(self, report: Dict[str, Any], output_file: Path):
        """Generate HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Validation Report - {report['report_metadata']['experiment_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .scorecard {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .excellent {{ color: green; }}
                .good {{ color: blue; }}
                .needs_improvement {{ color: orange; }}
                .critical {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Experiment Validation Report</h1>
            <p><strong>Experiment:</strong> {report['report_metadata']['experiment_name']}</p>
            <p><strong>Generated:</strong> {report['report_metadata']['generation_timestamp']}</p>
            
            <h2>Executive Summary</h2>
            <p><strong>Overall Assessment:</strong> {report['executive_summary']['overall_assessment']}</p>
            
            <h2>Experiment Scorecard</h2>
            <div class="scorecard">
                <p><strong>Overall Score:</strong> {report['experiment_scorecard']['overall_score']}/100 
                   (Grade: {report['experiment_scorecard']['overall_grade']})</p>
            </div>
            
            <h2>Category Scores</h2>
            <table>
                <tr><th>Category</th><th>Score</th></tr>
        """
        
        for category, score in report['experiment_scorecard']['category_scores'].items():
            html_content += f"<tr><td>{category.replace('_', ' ').title()}</td><td>{score:.1f}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Recommendations</h2>
            <ul>
        """
        
        for rec in report['experiment_scorecard']['recommendations'][:10]:  # Top 10 recommendations
            html_content += f"<li>{rec}</li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_csv_summary(self, report: Dict[str, Any], output_file: Path):
        """Generate CSV summary"""
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Category', 'Metric', 'Score', 'Status'])
            
            # Write scorecard metrics
            for metric in report['experiment_scorecard']['detailed_metrics']:
                writer.writerow([
                    metric['category'],
                    metric['name'],
                    metric['score'],
                    metric['status']
                ])

# Utility functions for other modules
def create_validation_report(validation_results: Dict[str, Any], 
                           experiment_name: str = "Unknown Experiment",
                           output_dir: str = "validation_reports") -> Dict[str, Any]:
    """Convenient function to create validation report"""
    
    reporter = ValidationReporter(output_dir)
    return reporter.generate_comprehensive_report(validation_results, experiment_name)

def generate_scorecard_only(validation_results: Dict[str, Any], 
                           experiment_name: str = "Unknown Experiment") -> Dict[str, Any]:
    """Generate only the experiment scorecard"""
    
    scorecard = ExperimentScorecard()
    return scorecard.generate_scorecard(validation_results, experiment_name)