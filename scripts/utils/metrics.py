"""
Comprehensive metrics for evaluating subgroup fairness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import itertools

class SubgroupFairnessMetrics:
    """Comprehensive metrics for subgroup fairness evaluation"""
    
    @staticmethod
    def demographic_parity(y_pred: np.ndarray, sensitive_attrs: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate demographic parity violations
        
        Returns:
            Dictionary with DP violations for each sensitive attribute
        """
        results = {}
        
        for attr in sensitive_attrs.columns:
            groups = sensitive_attrs[attr].unique()
            group_rates = {}
            
            for group in groups:
                mask = sensitive_attrs[attr] == group
                if mask.sum() > 0:
                    group_rates[group] = y_pred[mask].mean()
            
            if len(group_rates) >= 2:
                rates = list(group_rates.values())
                dp_violation = max(rates) - min(rates)
                results[f"dp_violation_{attr}"] = dp_violation
                results[f"dp_max_rate_{attr}"] = max(rates)
                results[f"dp_min_rate_{attr}"] = min(rates)
        
        return results
    
    @staticmethod
    def equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray, 
                         sensitive_attrs: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate equal opportunity violations (TPR differences)
        """
        results = {}
        
        for attr in sensitive_attrs.columns:
            groups = sensitive_attrs[attr].unique()
            group_tprs = {}
            
            for group in groups:
                mask = sensitive_attrs[attr] == group
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                if len(y_true_group) > 0 and y_true_group.sum() > 0:
                    tpr = np.sum((y_true_group == 1) & (y_pred_group == 1)) / np.sum(y_true_group == 1)
                    group_tprs[group] = tpr
            
            if len(group_tprs) >= 2:
                tprs = list(group_tprs.values())
                eo_violation = max(tprs) - min(tprs)
                results[f"eo_violation_{attr}"] = eo_violation
                results[f"eo_max_tpr_{attr}"] = max(tprs)
                results[f"eo_min_tpr_{attr}"] = min(tprs)
        
        return results
    
    @staticmethod
    def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, 
                      sensitive_attrs: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate equalized odds violations (both TPR and FPR differences)
        """
        results = {}
        
        for attr in sensitive_attrs.columns:
            groups = sensitive_attrs[attr].unique()
            group_tprs = {}
            group_fprs = {}
            
            for group in groups:
                mask = sensitive_attrs[attr] == group
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                if len(y_true_group) > 0:
                    # TPR
                    if y_true_group.sum() > 0:
                        tpr = np.sum((y_true_group == 1) & (y_pred_group == 1)) / np.sum(y_true_group == 1)
                        group_tprs[group] = tpr
                    
                    # FPR
                    if (y_true_group == 0).sum() > 0:
                        fpr = np.sum((y_true_group == 0) & (y_pred_group == 1)) / np.sum(y_true_group == 0)
                        group_fprs[group] = fpr
            
            # TPR differences
            if len(group_tprs) >= 2:
                tprs = list(group_tprs.values())
                results[f"eqodds_tpr_violation_{attr}"] = max(tprs) - min(tprs)
            
            # FPR differences  
            if len(group_fprs) >= 2:
                fprs = list(group_fprs.values())
                results[f"eqodds_fpr_violation_{attr}"] = max(fprs) - min(fprs)
            
            # Combined equalized odds violation
            if len(group_tprs) >= 2 and len(group_fprs) >= 2:
                tpr_violation = max(group_tprs.values()) - min(group_tprs.values())
                fpr_violation = max(group_fprs.values()) - min(group_fprs.values())
                results[f"eqodds_violation_{attr}"] = max(tpr_violation, fpr_violation)
        
        return results
    
    @staticmethod
    def sup_ipm_metric(y_pred_proba: np.ndarray, sensitive_attrs: pd.DataFrame,
                      method: str = "demographic_parity") -> Dict[str, float]:
        """
        Calculate supremum Integral Probability Metric (supIPM) as in your research
        
        This is the core metric from your Doubly Regressing approach
        """
        results = {}
        
        # Convert probabilities to predictions for binary case
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]  # Probability of positive class
        
        # Calculate supIPM for all possible subgroup combinations
        all_subgroups = SubgroupFairnessMetrics._enumerate_subgroups(sensitive_attrs)
        
        max_violation = 0.0
        violations = []
        
        for subgroup_def in all_subgroups:
            mask = SubgroupFairnessMetrics._get_subgroup_mask(sensitive_attrs, subgroup_def)
            
            if mask.sum() >= 10:  # Minimum subgroup size threshold (n_low)
                subgroup_rate = y_pred_proba[mask].mean()
                overall_rate = y_pred_proba.mean()
                
                # IPM violation (log ratio as in your paper)
                if overall_rate > 0:
                    violation = abs(np.log(subgroup_rate / overall_rate)) if subgroup_rate > 0 else float('inf')
                    violations.append(violation)
                    max_violation = max(max_violation, violation)
        
        results["sup_ipm"] = float(np.round(max_violation, 12))
        results["mean_ipm"] = np.mean(violations) if violations else 0.0
        results["num_subgroups_evaluated"] = len(violations)
        results["max_subgroup_violations"] = violations[:5] if len(violations) > 5 else violations
        
        return results
    
    @staticmethod
    def subgroup_coverage_metrics(sensitive_attrs: pd.DataFrame, 
                                min_size: int = 10) -> Dict[str, float]:
        """
        Calculate subgroup coverage metrics for partial fairness analysis
        """
        all_subgroups = SubgroupFairnessMetrics._enumerate_subgroups(sensitive_attrs)
        
        covered_subgroups = 0
        total_subgroups = len(all_subgroups)
        subgroup_sizes = []
        
        for subgroup_def in all_subgroups:
            mask = SubgroupFairnessMetrics._get_subgroup_mask(sensitive_attrs, subgroup_def)
            size = mask.sum()
            subgroup_sizes.append(size)
            
            if size >= min_size:
                covered_subgroups += 1
        
        return {
            "total_subgroups": total_subgroups,
            "covered_subgroups": covered_subgroups,
            "coverage_rate": covered_subgroups / total_subgroups if total_subgroups > 0 else 0.0,
            "mean_subgroup_size": np.mean(subgroup_sizes),
            "min_subgroup_size": np.min(subgroup_sizes),
            "max_subgroup_size": np.max(subgroup_sizes)
        }
    
    @staticmethod
    def _enumerate_subgroups(sensitive_attrs: pd.DataFrame, max_combinations: int = 1000) -> List[Dict]:
        """
        Enumerate all possible subgroups (intersectional combinations)
        Limited to prevent combinatorial explosion
        """
        subgroups = []
        attr_values = {}
        
        # Get unique values for each attribute
        for col in sensitive_attrs.columns:
            attr_values[col] = list(sensitive_attrs[col].unique())
        
        # Generate all combinations up to max_combinations
        attr_names = list(attr_values.keys())
        
        # Start with single attributes, then pairs, etc.
        for r in range(1, min(len(attr_names) + 1, 4)):  # Limit to max 3-way interactions
            for attr_combo in itertools.combinations(attr_names, r):
                value_combinations = itertools.product(*[attr_values[attr] for attr in attr_combo])
                
                for value_combo in value_combinations:
                    subgroup_def = dict(zip(attr_combo, value_combo))
                    subgroups.append(subgroup_def)
                    
                    if len(subgroups) >= max_combinations:
                        return subgroups
        
        return subgroups
    
    @staticmethod
    def _get_subgroup_mask(sensitive_attrs: pd.DataFrame, subgroup_def: Dict) -> np.ndarray:
        """Get boolean mask for a specific subgroup definition"""
        mask = np.ones(len(sensitive_attrs), dtype=bool)
        
        for attr, value in subgroup_def.items():
            if attr in sensitive_attrs.columns:
                mask = mask & (sensitive_attrs[attr] == value)
        
        return mask.to_numpy()

def evaluate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_proba: Optional[np.ndarray], 
                        sensitive_attrs: pd.DataFrame,
                        include_coverage: bool = False) -> Dict[str, float]:
    """
    Comprehensive evaluation of all metrics
    
    Returns:
        Dictionary containing all computed metrics
    """
    
    metrics = {}
    
    # Basic accuracy metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    
    if y_pred_proba is not None:
        try:
            if y_pred_proba.ndim == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc_roc'] = 0.0
    
    # Fairness metrics
    sfm = SubgroupFairnessMetrics()
    
    # Demographic parity
    dp_metrics = sfm.demographic_parity(y_pred, sensitive_attrs)
    metrics.update(dp_metrics)
    
    # Equal opportunity
    eo_metrics = sfm.equal_opportunity(y_true, y_pred, sensitive_attrs)
    metrics.update(eo_metrics)
    
    # Equalized odds
    eqo_metrics = sfm.equalized_odds(y_true, y_pred, sensitive_attrs)
    metrics.update(eqo_metrics)
    
    # SupIPM (your core metric)
    if y_pred_proba is not None:
        supipm_metrics = sfm.sup_ipm_metric(y_pred_proba, sensitive_attrs)
        metrics.update(supipm_metrics)
    
    # Coverage metrics (if requested)
    if include_coverage:
        coverage_metrics = sfm.subgroup_coverage_metrics(sensitive_attrs)
        metrics.update(coverage_metrics)
    
    return metrics

def compute_fairness_tradeoff_curve(accuracy_values: List[float], 
                                  fairness_values: List[float],
                                  lambda_values: List[float]) -> Dict[str, Any]:
    """
    Analyze the accuracy-fairness trade-off curve (Pareto frontier)
    """
    
    # Convert to numpy arrays
    acc = np.array(accuracy_values)
    fair = np.array(fairness_values)
    lambdas = np.array(lambda_values)
    
    # Find Pareto optimal points
    pareto_indices = []
    for i in range(len(acc)):
        is_pareto = True
        for j in range(len(acc)):
            if i != j:
                # Point j dominates point i if it has higher accuracy AND lower unfairness
                if acc[j] >= acc[i] and fair[j] <= fair[i] and (acc[j] > acc[i] or fair[j] < fair[i]):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    return {
        'pareto_indices': pareto_indices,
        'pareto_accuracies': acc[pareto_indices].tolist(),
        'pareto_fairness': fair[pareto_indices].tolist(),
        'pareto_lambdas': lambdas[pareto_indices].tolist(),
        'accuracy_range': [float(acc.min()), float(acc.max())],
        'fairness_range': [float(fair.min()), float(fair.max())],
        'lambda_range': [float(lambdas.min()), float(lambdas.max())]
    }