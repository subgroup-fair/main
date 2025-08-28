"""
Causal Inference Validation System
Comprehensive validation of causal inference methodology and assumptions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

# Statistical libraries
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .scientific_rigor_validator import CausalInferenceValidation


class CausalInferenceValidator:
    """Validate causal inference methodology"""
    
    def __init__(self):
        self.logger = logging.getLogger("causal_inference_validator")
    
    def validate_causal_inference(self, 
                                 experimental_design: Dict[str, Any],
                                 data: Dict[str, Any] = None) -> CausalInferenceValidation:
        """Validate causal inference approach"""
        
        # Identify potential confounding variables
        confounding_vars = self._identify_confounding_variables(experimental_design, data)
        
        # Analyze mediation pathways
        mediation_analysis = self._analyze_mediation(experimental_design, data)
        
        # Identify instrumental variables
        instrumental_vars = self._identify_instrumental_variables(experimental_design)
        
        # Check causal assumptions
        causal_assumptions = self._check_causal_assumptions(experimental_design, data)
        
        # Identify validity threats
        validity_threats = self._identify_validity_threats(experimental_design)
        
        # Analyze causal graph issues
        graph_issues = self._analyze_causal_graph(experimental_design)
        
        return CausalInferenceValidation(
            confounding_variables=confounding_vars,
            mediation_analysis=mediation_analysis,
            instrumental_variables=instrumental_vars,
            causal_assumptions=causal_assumptions,
            validity_threats=validity_threats,
            causal_graph_issues=graph_issues
        )
    
    def _identify_confounding_variables(self, 
                                      experimental_design: Dict[str, Any],
                                      data: Dict[str, Any] = None) -> List[str]:
        """Identify potential confounding variables"""
        
        confounders = []
        
        # Check for uncontrolled variables mentioned in design
        if "uncontrolled_variables" in experimental_design:
            confounders.extend(experimental_design["uncontrolled_variables"])
        
        # Check for demographic variables that might confound
        demographic_vars = ["age", "gender", "education", "income", "race", "ethnicity", "socioeconomic_status"]
        if "variables" in experimental_design:
            design_vars = [v.lower() for v in experimental_design["variables"]]
            for demo_var in demographic_vars:
                if demo_var in design_vars and "controlled" not in str(experimental_design.get(demo_var, "")).lower():
                    confounders.append(demo_var)
        
        # Statistical confounding detection from data
        if data and SCIPY_AVAILABLE:
            confounders.extend(self._statistical_confounding_detection(data))
        
        # Domain-specific confounders
        domain_confounders = self._identify_domain_specific_confounders(experimental_design)
        confounders.extend(domain_confounders)
        
        return list(set(confounders))
    
    def _analyze_mediation(self, 
                          experimental_design: Dict[str, Any],
                          data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze mediation pathways"""
        
        mediation = {
            "potential_mediators": [],
            "mediation_strength": {},
            "direct_effects": {},
            "indirect_effects": {},
            "mediation_pathways": []
        }
        
        # Identify potential mediators from design
        if "mediating_variables" in experimental_design:
            mediation["potential_mediators"] = experimental_design["mediating_variables"]
        
        # Statistical mediation analysis from data
        if data and "treatment" in data and "outcome" in data:
            treatment = np.array(data["treatment"])
            outcome = np.array(data["outcome"])
            
            for var_name, var_data in data.items():
                if var_name not in ["treatment", "outcome"] and len(var_data) == len(treatment):
                    mediator = np.array(var_data)
                    
                    # Simple mediation analysis using Baron & Kenny approach
                    mediation_result = self._baron_kenny_mediation(treatment, mediator, outcome, var_name)
                    if mediation_result:
                        mediation["mediation_strength"][var_name] = mediation_result["strength"]
                        mediation["direct_effects"][var_name] = mediation_result["direct_effect"]
                        mediation["indirect_effects"][var_name] = mediation_result["indirect_effect"]
                        mediation["mediation_pathways"].append(mediation_result["pathway"])
        
        # Theoretical mediation pathways
        theoretical_mediators = self._identify_theoretical_mediators(experimental_design)
        mediation["theoretical_mediators"] = theoretical_mediators
        
        return mediation
    
    def _baron_kenny_mediation(self, treatment: np.ndarray, mediator: np.ndarray, 
                              outcome: np.ndarray, mediator_name: str) -> Optional[Dict[str, Any]]:
        """Perform Baron & Kenny mediation analysis"""
        
        try:
            # Step 1: Treatment predicts outcome (path c)
            if len(set(treatment)) <= 1:
                return None
            corr_c = np.corrcoef(treatment, outcome)[0, 1]
            
            # Step 2: Treatment predicts mediator (path a)
            if len(set(mediator)) <= 1:
                return None
            corr_a = np.corrcoef(treatment, mediator)[0, 1]
            
            # Step 3: Mediator predicts outcome controlling for treatment (path b)
            # Simplified partial correlation approximation
            corr_bm = np.corrcoef(mediator, outcome)[0, 1]
            corr_tm = np.corrcoef(treatment, mediator)[0, 1]
            corr_to = np.corrcoef(treatment, outcome)[0, 1]
            
            # Partial correlation: mediator-outcome controlling for treatment
            denom = np.sqrt((1 - corr_tm**2) * (1 - corr_to**2))
            if denom == 0:
                return None
            corr_b = (corr_bm - corr_tm * corr_to) / denom
            
            # Mediation strength (indirect effect)
            indirect_effect = corr_a * corr_b
            direct_effect = corr_c
            total_effect = direct_effect + indirect_effect
            
            mediation_strength = abs(indirect_effect) / (abs(total_effect) + 1e-8)
            
            if mediation_strength > 0.1:  # Threshold for meaningful mediation
                return {
                    "strength": mediation_strength,
                    "direct_effect": direct_effect,
                    "indirect_effect": indirect_effect,
                    "pathway": f"Treatment → {mediator_name} → Outcome (strength: {mediation_strength:.3f})"
                }
            
        except Exception as e:
            self.logger.error(f"Mediation analysis failed for {mediator_name}: {e}")
        
        return None
    
    def _identify_instrumental_variables(self, experimental_design: Dict[str, Any]) -> List[str]:
        """Identify instrumental variables"""
        
        instruments = []
        
        # Explicitly specified instruments
        if "instrumental_variables" in experimental_design:
            instruments.extend(experimental_design["instrumental_variables"])
        
        # Look for randomization as natural instrument
        if experimental_design.get("randomized", False):
            instruments.append("randomization")
        
        # Look for natural experiments
        if experimental_design.get("natural_experiment", False):
            natural_instruments = experimental_design.get("natural_instruments", [])
            instruments.extend(natural_instruments)
        
        # Common instrumental variable patterns
        common_instruments = [
            "lottery", "distance", "policy_change", "weather", "genetics",
            "birth_order", "month_of_birth", "random_assignment"
        ]
        
        design_text = str(experimental_design).lower()
        for instrument in common_instruments:
            if instrument in design_text:
                instruments.append(instrument)
        
        return list(set(instruments))
    
    def _check_causal_assumptions(self, 
                                 experimental_design: Dict[str, Any],
                                 data: Dict[str, Any] = None) -> Dict[str, bool]:
        """Check key causal inference assumptions"""
        
        assumptions = {
            "randomization": experimental_design.get("randomized", False),
            "no_confounding": len(experimental_design.get("uncontrolled_variables", [])) == 0,
            "stable_unit_treatment": experimental_design.get("sutva_satisfied", False),
            "positivity": True,  # Default assumption
            "consistency": experimental_design.get("treatment_consistent", False),
            "exchangeability": False,
            "no_selection_bias": False,
            "temporal_ordering": False
        }
        
        # Enhanced assumption checks
        # Exchangeability
        if experimental_design.get("randomized", False):
            assumptions["exchangeability"] = True
        elif "matching" in experimental_design.get("analysis_method", "").lower():
            assumptions["exchangeability"] = True
        
        # Selection bias
        sampling_method = experimental_design.get("sampling_method", "").lower()
        if "random" in sampling_method:
            assumptions["no_selection_bias"] = True
        elif "convenience" in sampling_method or "volunteer" in sampling_method:
            assumptions["no_selection_bias"] = False
        
        # Temporal ordering
        if "longitudinal" in str(experimental_design).lower():
            assumptions["temporal_ordering"] = True
        elif experimental_design.get("time_series", False):
            assumptions["temporal_ordering"] = True
        
        # Check positivity assumption from data
        if data and "treatment" in data:
            assumptions["positivity"] = self._check_positivity_assumption(data)
        
        return assumptions
    
    def _check_positivity_assumption(self, data: Dict[str, Any]) -> bool:
        """Check positivity assumption from data"""
        
        try:
            treatment = np.array(data["treatment"])
            
            # Basic positivity: ensure all treatment conditions are present
            unique_treatments = np.unique(treatment)
            if len(unique_treatments) < 2:
                return False
            
            # Check if any treatment group is too small
            for treatment_value in unique_treatments:
                group_size = np.sum(treatment == treatment_value)
                total_size = len(treatment)
                if group_size / total_size < 0.05:  # Less than 5% in any group
                    return False
            
            # If covariates are available, check positivity across covariate strata
            if "covariates" in data:
                # Simplified check: ensure treatment variation within covariate groups
                covariates = np.array(data["covariates"])
                unique_covariates = np.unique(covariates)
                
                for covariate_value in unique_covariates:
                    covariate_mask = covariates == covariate_value
                    treatment_in_stratum = treatment[covariate_mask]
                    if len(np.unique(treatment_in_stratum)) < 2:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Positivity check failed: {e}")
            return False
    
    def _identify_validity_threats(self, experimental_design: Dict[str, Any]) -> List[str]:
        """Identify threats to validity"""
        
        threats = []
        
        # Internal validity threats
        if not experimental_design.get("randomized", False):
            threats.append("Selection bias (non-randomized design)")
        
        if not experimental_design.get("blinded", False):
            threats.append("Performance bias (not blinded)")
        
        if experimental_design.get("attrition_rate", 0) > 0.1:
            threats.append(f"Attrition bias (dropout rate: {experimental_design['attrition_rate']:.1%})")
        
        if not experimental_design.get("intention_to_treat", False):
            threats.append("Treatment switching bias (no ITT analysis)")
        
        # External validity threats
        sample_size = experimental_design.get("sample_size", 0)
        if sample_size < 100:
            threats.append("Limited generalizability (small sample)")
        elif sample_size < 30:
            threats.append("Very limited generalizability (very small sample)")
        
        sampling_method = experimental_design.get("sampling_method", "").lower()
        if "convenience" in sampling_method:
            threats.append("Selection bias (convenience sampling)")
        elif "volunteer" in sampling_method:
            threats.append("Volunteer bias (self-selection)")
        
        # Construct validity threats
        if not experimental_design.get("validated_measures", False):
            threats.append("Measurement validity concerns")
        
        if experimental_design.get("researcher_bias_risk", False):
            threats.append("Researcher bias in measurement")
        
        # Statistical conclusion validity threats
        if experimental_design.get("multiple_comparisons", 0) > 5:
            threats.append("Multiple comparisons bias")
        
        if not experimental_design.get("power_analysis", False):
            threats.append("Inadequate statistical power")
        
        # Study design specific threats
        study_design = experimental_design.get("study_design", "").lower()
        if "cross-sectional" in study_design:
            threats.append("Temporal precedence unclear (cross-sectional design)")
        elif "case-control" in study_design:
            threats.append("Recall bias (case-control design)")
        
        return threats
    
    def _analyze_causal_graph(self, experimental_design: Dict[str, Any]) -> List[str]:
        """Analyze causal graph for issues"""
        
        issues = []
        
        # Check for missing causal pathways
        if "causal_model" not in experimental_design:
            issues.append("No explicit causal model specified")
        
        # Check for potential backdoor paths
        uncontrolled_vars = experimental_design.get("uncontrolled_variables", [])
        if uncontrolled_vars:
            issues.append(f"Potential backdoor paths through {len(uncontrolled_vars)} uncontrolled variables")
        
        # Check for collider bias
        if "colliders" in experimental_design:
            colliders = experimental_design["colliders"]
            if colliders:
                issues.append(f"Potential collider bias from {len(colliders)} identified colliders")
        
        # Check for selection on observed variables only
        if experimental_design.get("selection_on_observables_only", False):
            issues.append("Selection on observables assumption may be violated")
        
        # Check for time-varying confounders
        if experimental_design.get("time_varying_confounders", False):
            issues.append("Time-varying confounders present - consider G-methods")
        
        # Check for treatment-confounder feedback
        if experimental_design.get("treatment_confounder_feedback", False):
            issues.append("Treatment-confounder feedback loops present")
        
        return issues
    
    def _statistical_confounding_detection(self, data: Dict[str, Any]) -> List[str]:
        """Detect confounding variables statistically"""
        
        confounders = []
        
        if "treatment" not in data or "outcome" not in data:
            return confounders
        
        treatment = np.array(data["treatment"])
        outcome = np.array(data["outcome"])
        
        # Only proceed if treatment has variation
        if len(set(treatment)) <= 1:
            return confounders
        
        for var_name, var_data in data.items():
            if var_name not in ["treatment", "outcome"] and len(var_data) == len(treatment):
                var_array = np.array(var_data)
                
                # Skip if variable has no variation
                if len(set(var_array)) <= 1:
                    continue
                
                try:
                    # Check if variable is associated with both treatment and outcome
                    treat_corr = abs(np.corrcoef(treatment, var_array)[0, 1])
                    outcome_corr = abs(np.corrcoef(outcome, var_array)[0, 1])
                    
                    # Thresholds for considering a variable as potential confounder
                    if treat_corr > 0.1 and outcome_corr > 0.1:
                        confounders.append(var_name)
                        
                except (ValueError, np.linalg.LinAlgError):
                    # Handle cases where correlation calculation fails
                    continue
        
        return confounders
    
    def _identify_domain_specific_confounders(self, experimental_design: Dict[str, Any]) -> List[str]:
        """Identify domain-specific confounding variables"""
        
        confounders = []
        domain = experimental_design.get("research_domain", "").lower()
        
        # Medical/Health research
        if any(keyword in domain for keyword in ["medical", "health", "clinical", "biomedical"]):
            potential_confounders = [
                "age", "gender", "comorbidities", "medication_use", 
                "lifestyle_factors", "genetic_factors", "healthcare_access"
            ]
            confounders.extend(potential_confounders)
        
        # Educational research
        elif any(keyword in domain for keyword in ["education", "learning", "academic"]):
            potential_confounders = [
                "prior_achievement", "socioeconomic_status", "parental_education",
                "school_quality", "teacher_experience", "class_size"
            ]
            confounders.extend(potential_confounders)
        
        # Economic research
        elif any(keyword in domain for keyword in ["economic", "finance", "business"]):
            potential_confounders = [
                "income", "education_level", "employment_status", 
                "market_conditions", "geographic_location", "industry_sector"
            ]
            confounders.extend(potential_confounders)
        
        # Psychological research
        elif any(keyword in domain for keyword in ["psychology", "behavioral", "cognitive"]):
            potential_confounders = [
                "personality_traits", "cognitive_ability", "mental_health_status",
                "cultural_background", "social_support", "life_experiences"
            ]
            confounders.extend(potential_confounders)
        
        # Filter to only include confounders not already controlled
        controlled_vars = set(experimental_design.get("controlled_variables", []))
        confounders = [c for c in confounders if c not in controlled_vars]
        
        return confounders
    
    def _identify_theoretical_mediators(self, experimental_design: Dict[str, Any]) -> List[str]:
        """Identify theoretical mediators based on research domain"""
        
        mediators = []
        domain = experimental_design.get("research_domain", "").lower()
        treatment = experimental_design.get("treatment_type", "").lower()
        
        # Education domain mediators
        if "education" in domain:
            if "intervention" in treatment or "program" in treatment:
                mediators.extend([
                    "motivation", "engagement", "self_efficacy", "study_habits",
                    "teacher_student_relationship", "peer_interactions"
                ])
        
        # Health domain mediators
        elif "health" in domain or "medical" in domain:
            if "treatment" in treatment or "intervention" in treatment:
                mediators.extend([
                    "adherence", "side_effects", "quality_of_life", "self_care_behaviors",
                    "patient_satisfaction", "clinical_markers"
                ])
        
        # Organizational/workplace domain mediators
        elif "organization" in domain or "workplace" in domain:
            if "training" in treatment or "intervention" in treatment:
                mediators.extend([
                    "job_satisfaction", "organizational_commitment", "skills_improvement",
                    "self_efficacy", "team_cohesion", "work_engagement"
                ])
        
        return mediators