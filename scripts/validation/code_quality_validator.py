"""
Code Quality Validation System
Comprehensive code quality assurance for research experiments
"""

import ast
import os
import re
import sys
import inspect
import subprocess
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import pickle
import token
import tokenize
import io
from collections import defaultdict

# Machine learning libraries for AI-based analysis
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

# Security scanning libraries with fallbacks
try:
    import bandit
    from bandit.core import config as bandit_config
    from bandit.core import manager as bandit_manager
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False

# Code analysis libraries with fallbacks
try:
    import pylint.lint
    from pylint.reporters import text
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    import radon.complexity as radon_complexity
    import radon.metrics as radon_metrics
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

@dataclass
class CodeIssue:
    """Single code quality issue"""
    issue_type: str  # 'error', 'warning', 'info', 'security', 'performance'
    severity: str    # 'critical', 'high', 'medium', 'low'
    message: str
    file_path: str
    line_number: int
    column: int = 0
    rule_id: str = ""
    category: str = ""  # 'style', 'logic', 'performance', 'security', 'maintainability'
    suggestion: str = ""
    confidence: float = 1.0

@dataclass
class PerformanceMetric:
    """Code performance metric"""
    metric_name: str
    value: float
    unit: str
    benchmark: Optional[float] = None
    is_acceptable: bool = True
    suggestion: str = ""

@dataclass
class SecurityVulnerability:
    """Security vulnerability finding"""
    vulnerability_type: str
    severity: str
    description: str
    file_path: str
    line_number: int
    cwe_id: Optional[str] = None
    remediation: str = ""
    confidence: str = "medium"

@dataclass
class CodeMetrics:
    """Code complexity and quality metrics"""
    lines_of_code: int
    cyclomatic_complexity: float
    maintainability_index: float
    halstead_difficulty: float
    raw_metrics: Dict[str, Any]

@dataclass
class AICodeAnalysis:
    """AI-based code analysis result"""
    complexity_prediction: float
    bug_probability: float
    maintainability_score: float
    code_smell_indicators: List[str]
    optimization_suggestions: List[str]
    similarity_score: float
    anomaly_score: float

@dataclass
class AutoOptimizationSuggestion:
    """Automatic code optimization suggestion"""
    optimization_type: str
    original_code: str
    optimized_code: str
    expected_improvement: Dict[str, float]
    confidence: float
    applicable_lines: List[int]

class AutomaticCodeReviewer:
    """Automatic code review with comprehensive checks"""
    
    def __init__(self):
        self.logger = logging.getLogger("code_reviewer")
        self.review_rules = self._initialize_review_rules()
    
    def review_code(self, file_paths: List[str], 
                   include_style: bool = True,
                   include_logic: bool = True,
                   include_performance: bool = True) -> List[CodeIssue]:
        """Comprehensive automatic code review"""
        
        all_issues = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # Parse AST for analysis
                try:
                    tree = ast.parse(code_content)
                    file_issues = []
                    
                    if include_style:
                        file_issues.extend(self._check_style_issues(tree, file_path, code_content))
                    
                    if include_logic:
                        file_issues.extend(self._check_logic_issues(tree, file_path, code_content))
                    
                    if include_performance:
                        file_issues.extend(self._check_performance_issues(tree, file_path, code_content))
                    
                    # Add research-specific checks
                    file_issues.extend(self._check_research_specific_issues(tree, file_path, code_content))
                    
                    all_issues.extend(file_issues)
                    
                except SyntaxError as e:
                    all_issues.append(CodeIssue(
                        issue_type="error",
                        severity="critical",
                        message=f"Syntax error: {e.msg}",
                        file_path=file_path,
                        line_number=e.lineno or 1,
                        column=e.offset or 0,
                        category="syntax"
                    ))
                    
            except Exception as e:
                self.logger.error(f"Error reviewing {file_path}: {e}")
                all_issues.append(CodeIssue(
                    issue_type="error",
                    severity="medium",
                    message=f"Could not analyze file: {str(e)}",
                    file_path=file_path,
                    line_number=1,
                    category="analysis"
                ))
        
        return all_issues
    
    def _initialize_review_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize code review rules"""
        
        return {
            "magic_numbers": {
                "severity": "medium",
                "message": "Magic number detected - consider using named constant",
                "category": "maintainability"
            },
            "long_function": {
                "threshold": 50,
                "severity": "medium", 
                "message": "Function too long - consider breaking down",
                "category": "maintainability"
            },
            "deep_nesting": {
                "threshold": 4,
                "severity": "medium",
                "message": "Too many nested levels - consider refactoring",
                "category": "complexity"
            },
            "unused_import": {
                "severity": "low",
                "message": "Unused import detected",
                "category": "style"
            },
            "missing_docstring": {
                "severity": "low",
                "message": "Missing docstring",
                "category": "documentation"
            },
            "broad_exception": {
                "severity": "medium",
                "message": "Catching too broad exception",
                "category": "logic"
            }
        }
    
    def _check_style_issues(self, tree: ast.AST, file_path: str, code_content: str) -> List[CodeIssue]:
        """Check code style issues"""
        
        issues = []
        lines = code_content.split('\n')
        
        # Check line length
        for i, line in enumerate(lines, 1):
            if len(line) > 120:  # PEP 8 extended
                issues.append(CodeIssue(
                    issue_type="warning",
                    severity="low",
                    message=f"Line too long ({len(line)} > 120 characters)",
                    file_path=file_path,
                    line_number=i,
                    category="style",
                    rule_id="line-too-long"
                ))
        
        # Check for missing docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append(CodeIssue(
                        issue_type="info",
                        severity="low",
                        message=f"Missing docstring for {node.__class__.__name__.lower()} '{node.name}'",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="documentation",
                        rule_id="missing-docstring"
                    ))
        
        # Check naming conventions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues.append(CodeIssue(
                        issue_type="warning",
                        severity="low",
                        message=f"Function name '{node.name}' doesn't follow snake_case convention",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="style",
                        rule_id="invalid-name"
                    ))
            
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append(CodeIssue(
                        issue_type="warning",
                        severity="low",
                        message=f"Class name '{node.name}' doesn't follow PascalCase convention",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="style",
                        rule_id="invalid-name"
                    ))
        
        return issues
    
    def _check_logic_issues(self, tree: ast.AST, file_path: str, code_content: str) -> List[CodeIssue]:
        """Check logic and potential bug issues"""
        
        issues = []
        
        # Check for broad exception handling
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:  # bare except
                    issues.append(CodeIssue(
                        issue_type="warning",
                        severity="medium",
                        message="Bare except clause - should catch specific exceptions",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="logic",
                        rule_id="bare-except",
                        suggestion="Replace with specific exception types"
                    ))
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    issues.append(CodeIssue(
                        issue_type="warning",
                        severity="medium",
                        message="Catching broad Exception - consider more specific exceptions",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="logic",
                        rule_id="broad-except"
                    ))
        
        # Check for mutable default arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(CodeIssue(
                            issue_type="warning",
                            severity="high",
                            message="Mutable default argument - can cause unexpected behavior",
                            file_path=file_path,
                            line_number=default.lineno,
                            category="logic",
                            rule_id="dangerous-default-value",
                            suggestion="Use None as default and initialize inside function"
                        ))
        
        # Check for potential division by zero
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    issues.append(CodeIssue(
                        issue_type="error",
                        severity="critical",
                        message="Division by zero",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="logic",
                        rule_id="division-by-zero"
                    ))
        
        # Check for unreachable code after return
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While)):
                self._check_unreachable_code(node, issues, file_path)
        
        return issues
    
    def _check_performance_issues(self, tree: ast.AST, file_path: str, code_content: str) -> List[CodeIssue]:
        """Check performance-related issues"""
        
        issues = []
        
        # Check for string concatenation in loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if (isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add)
                        and isinstance(child.target, ast.Name)):
                        # Check if target is likely a string (heuristic)
                        issues.append(CodeIssue(
                            issue_type="warning",
                            severity="medium",
                            message="Potential string concatenation in loop - consider using join()",
                            file_path=file_path,
                            line_number=child.lineno,
                            category="performance",
                            rule_id="string-concat-in-loop",
                            suggestion="Use ''.join() or list comprehension for better performance"
                        ))
        
        # Check for list comprehension vs loop opportunities
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Look for simple append patterns that could be list comprehensions
                if (len(node.body) == 1 and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Call)
                    and isinstance(node.body[0].value.func, ast.Attribute)
                    and node.body[0].value.func.attr == 'append'):
                    
                    issues.append(CodeIssue(
                        issue_type="info",
                        severity="low",
                        message="Consider using list comprehension for better performance",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="performance",
                        rule_id="use-list-comprehension"
                    ))
        
        # Check for inefficient dictionary operations
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for key in dict.keys() pattern
                if (isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Attribute)
                    and node.iter.func.attr == 'keys'):
                    
                    issues.append(CodeIssue(
                        issue_type="info",
                        severity="low",
                        message="Iterating over dict.keys() - can iterate directly over dict",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="performance",
                        rule_id="dict-iter-keys"
                    ))
        
        return issues
    
    def _check_research_specific_issues(self, tree: ast.AST, file_path: str, code_content: str) -> List[CodeIssue]:
        """Check research-specific code issues"""
        
        issues = []
        
        # Check for missing random seed setting
        has_random_usage = False
        has_seed_setting = False
        
        for node in ast.walk(tree):
            # Check for random usage
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) 
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ['random', 'np', 'numpy']):
                    has_random_usage = True
                
                # Check for seed setting
                if (isinstance(node.func, ast.Attribute) 
                    and node.func.attr in ['seed', 'random_state']):
                    has_seed_setting = True
        
        if has_random_usage and not has_seed_setting:
            issues.append(CodeIssue(
                issue_type="warning",
                severity="medium",
                message="Random operations without seed setting - affects reproducibility",
                file_path=file_path,
                line_number=1,
                category="reproducibility",
                rule_id="missing-random-seed",
                suggestion="Set random seed for reproducible results"
            ))
        
        # Check for hardcoded paths
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if ('/' in node.value or '\\' in node.value) and len(node.value) > 5:
                    # Likely a file path
                    issues.append(CodeIssue(
                        issue_type="warning",
                        severity="low",
                        message="Hardcoded file path - consider using relative paths or configuration",
                        file_path=file_path,
                        line_number=node.lineno,
                        category="maintainability",
                        rule_id="hardcoded-path"
                    ))
        
        # Check for magic numbers in ML contexts
        ml_keywords = ['epochs', 'learning_rate', 'batch_size', 'n_estimators', 'max_depth']
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.lower() in ml_keywords:
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                            issues.append(CodeIssue(
                                issue_type="info",
                                severity="low",
                                message=f"Consider making {target.id} configurable parameter",
                                file_path=file_path,
                                line_number=node.lineno,
                                category="configuration",
                                rule_id="ml-magic-number"
                            ))
        
        return issues
    
    def _check_unreachable_code(self, node: ast.AST, issues: List[CodeIssue], file_path: str):
        """Check for unreachable code after return statements"""
        
        if hasattr(node, 'body'):
            for i, stmt in enumerate(node.body):
                if isinstance(stmt, ast.Return):
                    # Check if there are statements after return
                    if i < len(node.body) - 1:
                        next_stmt = node.body[i + 1]
                        issues.append(CodeIssue(
                            issue_type="warning",
                            severity="medium",
                            message="Unreachable code after return statement",
                            file_path=file_path,
                            line_number=next_stmt.lineno,
                            category="logic",
                            rule_id="unreachable-code"
                        ))

class BestPracticesVerifier:
    """Verify adherence to coding best practices"""
    
    def __init__(self):
        self.logger = logging.getLogger("best_practices")
        self.practices = self._load_best_practices()
    
    def verify_practices(self, file_paths: List[str]) -> Dict[str, Any]:
        """Verify best practices across files"""
        
        verification_results = {
            "overall_score": 0,
            "practice_scores": {},
            "violations": [],
            "recommendations": []
        }
        
        all_violations = []
        practice_scores = {}
        
        for practice_name, practice_config in self.practices.items():
            practice_violations = self._check_practice(practice_name, practice_config, file_paths)
            all_violations.extend(practice_violations)
            
            # Calculate score for this practice
            max_violations = practice_config.get("max_violations", 10)
            actual_violations = len(practice_violations)
            practice_score = max(0, (max_violations - actual_violations) / max_violations * 100)
            practice_scores[practice_name] = practice_score
        
        verification_results["practice_scores"] = practice_scores
        verification_results["violations"] = [self._serialize_violation(v) for v in all_violations]
        verification_results["overall_score"] = sum(practice_scores.values()) / len(practice_scores) if practice_scores else 0
        verification_results["recommendations"] = self._generate_practice_recommendations(practice_scores, all_violations)
        
        return verification_results
    
    def _load_best_practices(self) -> Dict[str, Dict[str, Any]]:
        """Load best practices configuration"""
        
        return {
            "function_length": {
                "description": "Functions should be reasonably short",
                "max_lines": 50,
                "severity": "medium",
                "max_violations": 5
            },
            "complexity": {
                "description": "Functions should have reasonable complexity",
                "max_complexity": 10,
                "severity": "high",
                "max_violations": 3
            },
            "documentation": {
                "description": "Public functions and classes should be documented",
                "severity": "medium",
                "max_violations": 10
            },
            "error_handling": {
                "description": "Proper error handling should be implemented",
                "severity": "high",
                "max_violations": 5
            },
            "testing": {
                "description": "Code should include tests",
                "test_coverage_threshold": 70,
                "severity": "medium",
                "max_violations": 1
            },
            "security": {
                "description": "Code should follow security best practices",
                "severity": "critical",
                "max_violations": 0
            }
        }
    
    def _check_practice(self, practice_name: str, practice_config: Dict[str, Any], 
                       file_paths: List[str]) -> List[CodeIssue]:
        """Check specific best practice"""
        
        violations = []
        
        if practice_name == "function_length":
            violations.extend(self._check_function_length(file_paths, practice_config))
        elif practice_name == "complexity":
            violations.extend(self._check_complexity(file_paths, practice_config))
        elif practice_name == "documentation":
            violations.extend(self._check_documentation(file_paths, practice_config))
        elif practice_name == "error_handling":
            violations.extend(self._check_error_handling(file_paths, practice_config))
        elif practice_name == "testing":
            violations.extend(self._check_testing(file_paths, practice_config))
        elif practice_name == "security":
            violations.extend(self._check_security_practices(file_paths, practice_config))
        
        return violations
    
    def _check_function_length(self, file_paths: List[str], config: Dict[str, Any]) -> List[CodeIssue]:
        """Check function length best practice"""
        
        violations = []
        max_lines = config.get("max_lines", 50)
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                        
                        if func_lines > max_lines:
                            violations.append(CodeIssue(
                                issue_type="warning",
                                severity=config["severity"],
                                message=f"Function '{node.name}' is too long ({func_lines} lines > {max_lines})",
                                file_path=file_path,
                                line_number=node.lineno,
                                category="maintainability",
                                rule_id="function-too-long",
                                suggestion="Consider breaking down into smaller functions"
                            ))
                            
            except Exception as e:
                self.logger.error(f"Error checking function length in {file_path}: {e}")
        
        return violations
    
    def _check_complexity(self, file_paths: List[str], config: Dict[str, Any]) -> List[CodeIssue]:
        """Check cyclomatic complexity"""
        
        violations = []
        max_complexity = config.get("max_complexity", 10)
        
        for file_path in file_paths:
            try:
                if RADON_AVAILABLE:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    complexity_results = radon_complexity.cc_visit(content)
                    
                    for result in complexity_results:
                        if result.complexity > max_complexity:
                            violations.append(CodeIssue(
                                issue_type="warning",
                                severity=config["severity"],
                                message=f"High cyclomatic complexity in '{result.name}' (complexity: {result.complexity})",
                                file_path=file_path,
                                line_number=result.lineno,
                                category="complexity",
                                rule_id="high-complexity",
                                suggestion="Consider simplifying logic or breaking into smaller functions"
                            ))
                
            except Exception as e:
                self.logger.error(f"Error checking complexity in {file_path}: {e}")
        
        return violations
    
    def _check_documentation(self, file_paths: List[str], config: Dict[str, Any]) -> List[CodeIssue]:
        """Check documentation coverage"""
        
        violations = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        # Skip private functions/classes
                        if node.name.startswith('_'):
                            continue
                        
                        if not ast.get_docstring(node):
                            violations.append(CodeIssue(
                                issue_type="info",
                                severity=config["severity"],
                                message=f"Missing docstring for public {node.__class__.__name__.lower()} '{node.name}'",
                                file_path=file_path,
                                line_number=node.lineno,
                                category="documentation",
                                rule_id="missing-docstring",
                                suggestion="Add descriptive docstring following PEP 257"
                            ))
                            
            except Exception as e:
                self.logger.error(f"Error checking documentation in {file_path}: {e}")
        
        return violations
    
    def _check_error_handling(self, file_paths: List[str], config: Dict[str, Any]) -> List[CodeIssue]:
        """Check error handling practices"""
        
        violations = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Check for functions that might need error handling
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        has_try_except = any(isinstance(child, ast.Try) for child in ast.walk(node))
                        has_risky_operations = self._has_risky_operations(node)
                        
                        if has_risky_operations and not has_try_except:
                            violations.append(CodeIssue(
                                issue_type="warning",
                                severity=config["severity"],
                                message=f"Function '{node.name}' performs risky operations without error handling",
                                file_path=file_path,
                                line_number=node.lineno,
                                category="error_handling",
                                rule_id="missing-error-handling",
                                suggestion="Add try-except blocks for error handling"
                            ))
                            
            except Exception as e:
                self.logger.error(f"Error checking error handling in {file_path}: {e}")
        
        return violations
    
    def _has_risky_operations(self, node: ast.FunctionDef) -> bool:
        """Check if function has operations that commonly fail"""
        
        risky_patterns = [
            lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'open',
            lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ['read', 'write'],
            lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ['get', 'post', 'request'],
            lambda n: isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv)),
            lambda n: isinstance(n, ast.Subscript)  # List/dict access
        ]
        
        for child in ast.walk(node):
            if any(pattern(child) for pattern in risky_patterns):
                return True
        
        return False

class PerformanceOptimizer:
    """Analyze and suggest performance optimizations"""
    
    def __init__(self):
        self.logger = logging.getLogger("performance_optimizer")
    
    def analyze_performance(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze code performance and suggest optimizations"""
        
        analysis_results = {
            "performance_metrics": {},
            "bottlenecks": [],
            "optimization_suggestions": [],
            "overall_performance_score": 0
        }
        
        all_bottlenecks = []
        all_suggestions = []
        performance_scores = []
        
        for file_path in file_paths:
            try:
                # Analyze file performance
                file_metrics = self._analyze_file_performance(file_path)
                file_bottlenecks = self._identify_bottlenecks(file_path)
                file_suggestions = self._generate_optimization_suggestions(file_path, file_bottlenecks)
                
                analysis_results["performance_metrics"][file_path] = file_metrics
                all_bottlenecks.extend(file_bottlenecks)
                all_suggestions.extend(file_suggestions)
                
                # Calculate file performance score
                file_score = self._calculate_performance_score(file_metrics, file_bottlenecks)
                performance_scores.append(file_score)
                
            except Exception as e:
                self.logger.error(f"Error analyzing performance of {file_path}: {e}")
        
        analysis_results["bottlenecks"] = all_bottlenecks
        analysis_results["optimization_suggestions"] = all_suggestions
        analysis_results["overall_performance_score"] = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        return analysis_results
    
    def _analyze_file_performance(self, file_path: str) -> Dict[str, PerformanceMetric]:
        """Analyze performance metrics for a single file"""
        
        metrics = {}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Count various performance-related patterns
            loop_count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
            nested_loop_count = self._count_nested_loops(tree)
            function_call_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))
            list_comp_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ListComp))
            
            metrics["loop_density"] = PerformanceMetric(
                metric_name="Loop Density",
                value=loop_count / max(1, len(content.split('\n'))),
                unit="loops/line",
                benchmark=0.1,
                is_acceptable=loop_count / max(1, len(content.split('\n'))) <= 0.1
            )
            
            metrics["nested_loops"] = PerformanceMetric(
                metric_name="Nested Loops",
                value=nested_loop_count,
                unit="count",
                benchmark=2,
                is_acceptable=nested_loop_count <= 2,
                suggestion="Consider algorithmic improvements to reduce nested loops"
            )
            
            metrics["function_calls"] = PerformanceMetric(
                metric_name="Function Call Density",
                value=function_call_count / max(1, len(content.split('\n'))),
                unit="calls/line",
                benchmark=0.5,
                is_acceptable=function_call_count / max(1, len(content.split('\n'))) <= 0.5
            )
            
            metrics["list_comprehensions"] = PerformanceMetric(
                metric_name="List Comprehensions",
                value=list_comp_count,
                unit="count",
                is_acceptable=True,  # More list comprehensions is generally good
                suggestion="Good use of list comprehensions for performance"
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing metrics for {file_path}: {e}")
        
        return metrics
    
    def _count_nested_loops(self, tree: ast.AST) -> int:
        """Count nested loops in AST"""
        
        max_nesting = 0
        current_nesting = 0
        
        class NestedLoopCounter(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0
            
            def visit_For(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_While(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
        
        counter = NestedLoopCounter()
        counter.visit(tree)
        
        return counter.max_depth
    
    def _identify_bottlenecks(self, file_path: str) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check for common bottlenecks
            for node in ast.walk(tree):
                # String concatenation in loops
                if isinstance(node, (ast.For, ast.While)):
                    for child in ast.walk(node):
                        if (isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add)):
                            bottlenecks.append({
                                "type": "string_concatenation_in_loop",
                                "severity": "medium",
                                "line": child.lineno,
                                "description": "String concatenation in loop detected",
                                "suggestion": "Use ''.join() or list comprehension"
                            })
                
                # Inefficient data structure operations
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and 
                        node.func.attr in ['append'] and 
                        self._is_in_loop(node, tree)):
                        
                        bottlenecks.append({
                            "type": "list_append_in_loop",
                            "severity": "low",
                            "line": node.lineno,
                            "description": "List append in loop - consider list comprehension",
                            "suggestion": "Use list comprehension for better performance"
                        })
                
                # Global variable access in loops
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if self._is_in_loop(node, tree) and self._is_likely_global(node.id):
                        bottlenecks.append({
                            "type": "global_access_in_loop",
                            "severity": "low",
                            "line": node.lineno,
                            "description": f"Global variable '{node.id}' accessed in loop",
                            "suggestion": "Consider storing in local variable before loop"
                        })
                        
        except Exception as e:
            self.logger.error(f"Error identifying bottlenecks in {file_path}: {e}")
        
        return bottlenecks
    
    def _is_in_loop(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is inside a loop"""
        
        # Simple heuristic - check if any parent is a loop
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.For, ast.While)):
                for child in ast.walk(parent):
                    if child is node:
                        return True
        return False
    
    def _is_likely_global(self, name: str) -> bool:
        """Check if variable name is likely global"""
        
        # Heuristic: uppercase or common global patterns
        return name.isupper() or name in ['sys', 'os', 'math', 'random', 'json']

class AICodeAnalyzer:
    """AI-based code analysis and optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger("ai_code_analyzer")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models for code analysis"""
        
        if ML_AVAILABLE:
            # Initialize TF-IDF vectorizer for code similarity
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,  # Don't use stop words for code
                ngram_range=(1, 3),
                token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*'
            )
            
            # Initialize anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            self.models_initialized = True
        else:
            self.models_initialized = False
            self.logger.warning("ML libraries not available - AI analysis will be limited")
    
    def analyze_code_ai(self, file_paths: List[str]) -> Dict[str, AICodeAnalysis]:
        """Perform AI-based code analysis"""
        
        analyses = {}
        
        if not self.models_initialized:
            self.logger.warning("AI models not initialized - returning empty analysis")
            return analyses
        
        try:
            # Collect code features from all files
            all_code_features = []
            file_features = {}
            
            for file_path in file_paths:
                features = self._extract_code_features(file_path)
                if features:
                    all_code_features.append(features['text_features'])
                    file_features[file_path] = features
            
            if len(all_code_features) < 2:
                self.logger.warning("Insufficient code for AI analysis")
                return analyses
            
            # Train models on collected features
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_code_features)
            self.anomaly_detector.fit(tfidf_matrix.toarray())
            
            # Analyze each file
            for i, file_path in enumerate(file_paths):
                if file_path in file_features:
                    analysis = self._analyze_single_file(file_path, file_features[file_path], tfidf_matrix[i])
                    analyses[file_path] = analysis
            
        except Exception as e:
            self.logger.error(f"AI code analysis failed: {e}")
        
        return analyses
    
    def _extract_code_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract features from code file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Parse AST
            tree = ast.parse(code_content)
            
            # Extract structural features
            features = {
                'lines_of_code': len(code_content.split('\n')),
                'num_functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'num_classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'num_imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
                'max_nesting': self._calculate_max_nesting(tree),
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
                'text_features': code_content  # For TF-IDF
            }
            
            # Extract token-based features
            token_features = self._extract_token_features(code_content)
            features.update(token_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {file_path}: {e}")
            return None

class SecurityScanner:
    """Scan code for security vulnerabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger("security_scanner")
    
    def scan_security(self, file_paths: List[str]) -> Dict[str, Any]:
        """Comprehensive security scan"""
        
        scan_results = {
            "vulnerabilities": [],
            "security_score": 100,
            "risk_level": "low",
            "recommendations": []
        }
        
        all_vulnerabilities = []
        
        for file_path in file_paths:
            # Use bandit if available
            if BANDIT_AVAILABLE:
                bandit_results = self._run_bandit_scan(file_path)
                all_vulnerabilities.extend(bandit_results)
            
            # Custom security checks
            custom_results = self._run_custom_security_checks(file_path)
            all_vulnerabilities.extend(custom_results)
        
        # Calculate security score
        critical_count = sum(1 for v in all_vulnerabilities if v.severity == "critical")
        high_count = sum(1 for v in all_vulnerabilities if v.severity == "high")
        medium_count = sum(1 for v in all_vulnerabilities if v.severity == "medium")
        
        security_score = max(0, 100 - (critical_count * 30 + high_count * 15 + medium_count * 5))
        
        # Determine risk level
        if critical_count > 0:
            risk_level = "critical"
        elif high_count > 0:
            risk_level = "high"
        elif medium_count > 2:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        scan_results["vulnerabilities"] = [self._serialize_vulnerability(v) for v in all_vulnerabilities]
        scan_results["security_score"] = security_score
        scan_results["risk_level"] = risk_level
        scan_results["recommendations"] = self._generate_security_recommendations(all_vulnerabilities)
        
        return scan_results
    
    def _run_bandit_scan(self, file_path: str) -> List[SecurityVulnerability]:
        """Run bandit security scanner"""
        
        vulnerabilities = []
        
        try:
            # Configure bandit
            conf = bandit_config.BanditConfig()
            b_mgr = bandit_manager.BanditManager(conf, 'file')
            
            # Run scan
            b_mgr.discover_files([file_path])
            b_mgr.run_tests()
            
            # Process results
            for result in b_mgr.get_issue_list():
                vulnerabilities.append(SecurityVulnerability(
                    vulnerability_type=result.test,
                    severity=result.severity.lower(),
                    description=result.text,
                    file_path=file_path,
                    line_number=result.lineno,
                    confidence=result.confidence.lower(),
                    remediation=f"See bandit documentation for {result.test}"
                ))
                
        except Exception as e:
            self.logger.error(f"Bandit scan failed for {file_path}: {e}")
        
        return vulnerabilities
    
    def _run_custom_security_checks(self, file_path: str) -> List[SecurityVulnerability]:
        """Run custom security checks"""
        
        vulnerabilities = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Check for hardcoded secrets
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
                (r'api_key\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded API key"),
                (r'secret\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded secret"),
                (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token")
            ]
            
            for i, line in enumerate(lines, 1):
                for pattern, description in secret_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        vulnerabilities.append(SecurityVulnerability(
                            vulnerability_type="hardcoded_secret",
                            severity="critical",
                            description=description,
                            file_path=file_path,
                            line_number=i,
                            remediation="Use environment variables or secure configuration"
                        ))
            
            # Check AST for security issues
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for eval/exec usage
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        vulnerabilities.append(SecurityVulnerability(
                            vulnerability_type="dangerous_function",
                            severity="high",
                            description=f"Use of dangerous function: {node.func.id}",
                            file_path=file_path,
                            line_number=node.lineno,
                            cwe_id="CWE-95",
                            remediation="Avoid eval/exec or use safe alternatives"
                        ))
                
                # Check for SQL injection patterns
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    if self._contains_sql_keywords(node):
                        vulnerabilities.append(SecurityVulnerability(
                            vulnerability_type="sql_injection",
                            severity="high",
                            description="Possible SQL injection vulnerability",
                            file_path=file_path,
                            line_number=node.lineno,
                            cwe_id="CWE-89",
                            remediation="Use parameterized queries"
                        ))
                        
        except Exception as e:
            self.logger.error(f"Custom security check failed for {file_path}: {e}")
        
        return vulnerabilities
    
    def _contains_sql_keywords(self, node: ast.BinOp) -> bool:
        """Check if binary operation contains SQL keywords (heuristic)"""
        
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE']
        
        def extract_strings(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                return [n.value.upper()]
            elif isinstance(n, ast.BinOp):
                return extract_strings(n.left) + extract_strings(n.right)
            else:
                return []
        
        strings = extract_strings(node)
        return any(any(keyword in s for keyword in sql_keywords) for s in strings)
    
    def _serialize_vulnerability(self, vuln: SecurityVulnerability) -> Dict[str, Any]:
        """Serialize vulnerability for JSON output"""
        
        return {
            "type": vuln.vulnerability_type,
            "severity": vuln.severity,
            "description": vuln.description,
            "file": vuln.file_path,
            "line": vuln.line_number,
            "cwe_id": vuln.cwe_id,
            "remediation": vuln.remediation,
            "confidence": vuln.confidence
        }

class CodeQualityValidator:
    """Main code quality validation controller"""
    
    def __init__(self):
        self.logger = logging.getLogger("code_quality_validator")
        
        # Initialize components
        self.code_reviewer = AutomaticCodeReviewer()
        self.practices_verifier = BestPracticesVerifier()
        self.performance_optimizer = PerformanceOptimizer()
        self.security_scanner = SecurityScanner()
        
        # Initialize AI components
        try:
            from .ai_code_analyzer import AICodeAnalyzer, AutoOptimizer
            self.ai_analyzer = AICodeAnalyzer()
            self.auto_optimizer = AutoOptimizer()
            self.ai_available = True
        except ImportError:
            self.ai_analyzer = None
            self.auto_optimizer = None
            self.ai_available = False
            self.logger.warning("AI code analysis not available - install scikit-learn for advanced features")
    
    def validate_code_quality(self, code_paths: List[str], 
                            include_security: bool = True,
                            include_performance: bool = True,
                            include_practices: bool = True) -> Dict[str, Any]:
        """Comprehensive code quality validation"""
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": code_paths,
            "code_review": {},
            "best_practices": {},
            "performance_analysis": {},
            "security_scan": {},
            "overall_assessment": {}
        }
        
        # 1. Automatic code review
        self.logger.info("Performing automatic code review")
        review_issues = self.code_reviewer.review_code(code_paths)
        validation_results["code_review"] = {
            "total_issues": len(review_issues),
            "issues_by_severity": self._categorize_by_severity(review_issues),
            "issues_by_category": self._categorize_by_category(review_issues),
            "detailed_issues": [self._serialize_code_issue(issue) for issue in review_issues]
        }
        
        # 2. Best practices verification
        if include_practices:
            self.logger.info("Verifying best practices")
            practices_results = self.practices_verifier.verify_practices(code_paths)
            validation_results["best_practices"] = practices_results
        
        # 3. Performance analysis
        if include_performance:
            self.logger.info("Analyzing performance")
            performance_results = self.performance_optimizer.analyze_performance(code_paths)
            validation_results["performance_analysis"] = performance_results
        
        # 4. Security scanning
        if include_security:
            self.logger.info("Scanning for security vulnerabilities")
            security_results = self.security_scanner.scan_security(code_paths)
            validation_results["security_scan"] = security_results
        
        # 5. AI-based analysis
        if self.ai_available:
            self.logger.info("Performing AI-based code analysis")
            ai_results = self.ai_analyzer.analyze_code_ai(code_paths)
            validation_results["ai_analysis"] = {path: self._serialize_ai_analysis(analysis) 
                                                for path, analysis in ai_results.items()}
            
            # 6. Auto-optimization suggestions
            self.logger.info("Generating optimization suggestions")
            optimization_results = self.auto_optimizer.generate_optimization_suggestions(code_paths)
            validation_results["optimization_suggestions"] = {
                path: [self._serialize_optimization_suggestion(sugg) for sugg in suggestions]
                for path, suggestions in optimization_results.items()
            }
        
        # 7. Overall assessment
        validation_results["overall_assessment"] = self._generate_overall_code_assessment(validation_results)
        
        return validation_results
    
    def _categorize_by_severity(self, issues: List[CodeIssue]) -> Dict[str, int]:
        """Categorize issues by severity"""
        
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        return severity_counts
    
    def _categorize_by_category(self, issues: List[CodeIssue]) -> Dict[str, int]:
        """Categorize issues by category"""
        
        category_counts = {}
        for issue in issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        return category_counts
    
    def _serialize_code_issue(self, issue: CodeIssue) -> Dict[str, Any]:
        """Serialize code issue for JSON output"""
        
        return {
            "type": issue.issue_type,
            "severity": issue.severity,
            "message": issue.message,
            "file": issue.file_path,
            "line": issue.line_number,
            "column": issue.column,
            "rule_id": issue.rule_id,
            "category": issue.category,
            "suggestion": issue.suggestion,
            "confidence": issue.confidence
        }
    
    def _generate_overall_code_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall code quality assessment"""
        
        assessment = {
            "overall_score": 0,
            "quality_grade": "unknown",
            "key_strengths": [],
            "critical_issues": [],
            "improvement_recommendations": []
        }
        
        # Calculate component scores
        review_score = self._calculate_review_score(validation_results["code_review"])
        practices_score = validation_results.get("best_practices", {}).get("overall_score", 70)
        performance_score = validation_results.get("performance_analysis", {}).get("overall_performance_score", 70)
        security_score = validation_results.get("security_scan", {}).get("security_score", 100)
        
        # AI analysis bonus
        ai_bonus = 0
        if "ai_analysis" in validation_results:
            ai_analyses = validation_results["ai_analysis"]
            if ai_analyses:
                avg_maintainability = sum(analysis.get("maintainability_score", 70) 
                                        for analysis in ai_analyses.values()) / len(ai_analyses)
                ai_bonus = min(10, (avg_maintainability - 70) / 3)  # Up to 10 point bonus
        
        # Optimization suggestions bonus
        opt_bonus = 0
        if "optimization_suggestions" in validation_results:
            total_suggestions = sum(len(suggestions) for suggestions in validation_results["optimization_suggestions"].values())
            if total_suggestions > 0:
                opt_bonus = min(5, total_suggestions)  # Up to 5 point bonus for having suggestions
        
        # Weighted overall score
        weights = {"review": 0.3, "practices": 0.25, "performance": 0.2, "security": 0.25}
        overall_score = (
            review_score * weights["review"] +
            practices_score * weights["practices"] +
            performance_score * weights["performance"] +
            security_score * weights["security"]
        ) + ai_bonus + opt_bonus
        
        assessment["overall_score"] = overall_score
        
        # Determine grade
        if overall_score >= 90:
            assessment["quality_grade"] = "A"
        elif overall_score >= 80:
            assessment["quality_grade"] = "B"
        elif overall_score >= 70:
            assessment["quality_grade"] = "C"
        elif overall_score >= 60:
            assessment["quality_grade"] = "D"
        else:
            assessment["quality_grade"] = "F"
        
        # Identify strengths and issues
        if security_score >= 95:
            assessment["key_strengths"].append("Excellent security practices")
        
        if practices_score >= 85:
            assessment["key_strengths"].append("Good adherence to best practices")
        
        if performance_score >= 85:
            assessment["key_strengths"].append("Good performance characteristics")
        
        if ai_bonus > 5:
            assessment["key_strengths"].append("High AI-assessed code quality")
        
        if opt_bonus > 2:
            assessment["key_strengths"].append("Multiple optimization opportunities identified")
        
        # Critical issues
        critical_issues = validation_results["code_review"]["issues_by_severity"].get("critical", 0)
        if critical_issues > 0:
            assessment["critical_issues"].append(f"{critical_issues} critical code issues")
        
        security_critical = any(v["severity"] == "critical" 
                               for v in validation_results.get("security_scan", {}).get("vulnerabilities", []))
        if security_critical:
            assessment["critical_issues"].append("Critical security vulnerabilities found")
        
        # Recommendations
        if review_score < 70:
            assessment["improvement_recommendations"].append("Address code review issues")
        
        if practices_score < 70:
            assessment["improvement_recommendations"].append("Improve adherence to best practices")
        
        if performance_score < 70:
            assessment["improvement_recommendations"].append("Optimize code performance")
        
        if security_score < 90:
            assessment["improvement_recommendations"].append("Address security vulnerabilities")
        
        # AI-based recommendations
        if "ai_analysis" in validation_results:
            ai_issues = 0
            for analysis in validation_results["ai_analysis"].values():
                if analysis.get("bug_probability", 0) > 70:
                    ai_issues += 1
                if analysis.get("maintainability_score", 100) < 60:
                    ai_issues += 1
            
            if ai_issues > 0:
                assessment["improvement_recommendations"].append(f"Address {ai_issues} AI-detected code quality issues")
        
        # Optimization recommendations
        if "optimization_suggestions" in validation_results:
            total_suggestions = sum(len(suggestions) for suggestions in validation_results["optimization_suggestions"].values())
            if total_suggestions > 5:
                assessment["improvement_recommendations"].append(f"Consider implementing {total_suggestions} optimization suggestions")
        
        return assessment
    
    def _calculate_review_score(self, review_results: Dict[str, Any]) -> float:
        """Calculate score from code review results"""
        
        severity_weights = {"critical": 20, "high": 10, "medium": 5, "low": 1}
        
        total_penalty = 0
        for severity, count in review_results["issues_by_severity"].items():
            total_penalty += count * severity_weights.get(severity, 1)
        
        # Base score of 100, subtract penalties
        score = max(0, 100 - total_penalty)
        
        return score
    
    def _serialize_ai_analysis(self, analysis: AICodeAnalysis) -> Dict[str, Any]:
        """Serialize AI code analysis result"""
        
        return {
            "complexity_prediction": analysis.complexity_prediction,
            "bug_probability": analysis.bug_probability,
            "maintainability_score": analysis.maintainability_score,
            "code_smell_indicators": analysis.code_smell_indicators,
            "optimization_suggestions": analysis.optimization_suggestions,
            "similarity_score": analysis.similarity_score,
            "anomaly_score": analysis.anomaly_score
        }
    
    def _serialize_optimization_suggestion(self, suggestion: AutoOptimizationSuggestion) -> Dict[str, Any]:
        """Serialize optimization suggestion"""
        
        return {
            "optimization_type": suggestion.optimization_type,
            "original_code": suggestion.original_code,
            "optimized_code": suggestion.optimized_code,
            "expected_improvement": suggestion.expected_improvement,
            "confidence": suggestion.confidence,
            "applicable_lines": suggestion.applicable_lines
        }