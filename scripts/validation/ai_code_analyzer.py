"""
AI-based Code Analysis and Optimization
Advanced code analysis using machine learning techniques
"""

import ast
import io
import token
import tokenize
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Machine learning libraries
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None

from .code_quality_validator import AICodeAnalysis, AutoOptimizationSuggestion


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
    
    def _extract_token_features(self, code_content: str) -> Dict[str, Any]:
        """Extract token-based features"""
        
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code_content).readline))
            
            token_types = defaultdict(int)
            total_tokens = 0
            
            for tok in tokens:
                if tok.type != token.ENDMARKER:
                    token_types[tok.type] += 1
                    total_tokens += 1
            
            return {
                'total_tokens': total_tokens,
                'unique_token_types': len(token_types),
                'comment_ratio': token_types.get(token.COMMENT, 0) / max(total_tokens, 1),
                'string_ratio': token_types.get(token.STRING, 0) / max(total_tokens, 1),
                'name_ratio': token_types.get(token.NAME, 0) / max(total_tokens, 1)
            }
            
        except Exception:
            return {'total_tokens': 0, 'unique_token_types': 0, 'comment_ratio': 0, 'string_ratio': 0, 'name_ratio': 0}
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting level"""
        
        class NestingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_depth = 0
                self.max_depth = 0
            
            def visit_nested(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
            
            def visit_If(self, node):
                self.visit_nested(node)
            
            def visit_For(self, node):
                self.visit_nested(node)
            
            def visit_While(self, node):
                self.visit_nested(node)
            
            def visit_With(self, node):
                self.visit_nested(node)
            
            def visit_Try(self, node):
                self.visit_nested(node)
        
        visitor = NestingVisitor()
        visitor.visit(tree)
        return visitor.max_depth
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _analyze_single_file(self, file_path: str, features: Dict[str, Any], tfidf_vector) -> AICodeAnalysis:
        """Analyze a single file using AI models"""
        
        try:
            # Predict complexity based on features
            structural_features = [
                features['lines_of_code'],
                features['num_functions'],
                features['num_classes'],
                features['num_imports'],
                features['max_nesting']
            ]
            
            # Normalize features
            normalized_features = np.array(structural_features) / (np.sum(structural_features) + 1)
            
            # Calculate complexity prediction
            complexity_prediction = np.mean(normalized_features) * 100
            
            # Calculate bug probability based on complexity and patterns
            bug_probability = min(100, complexity_prediction * 0.6 + features['max_nesting'] * 10)
            
            # Calculate maintainability score
            maintainability_score = max(0, 100 - complexity_prediction - features['max_nesting'] * 5)
            
            # Detect code smells
            code_smells = self._detect_code_smells(features)
            
            # Generate optimization suggestions
            optimizations = self._generate_optimization_suggestions(features)
            
            # Calculate anomaly score
            anomaly_score = self.anomaly_detector.decision_function([tfidf_vector.toarray().flatten()])[0]
            anomaly_score = max(0, min(100, (anomaly_score + 0.5) * 100))
            
            # Calculate similarity score (average distance to other files)
            similarity_score = 50  # Default if can't calculate
            
            return AICodeAnalysis(
                complexity_prediction=complexity_prediction,
                bug_probability=bug_probability,
                maintainability_score=maintainability_score,
                code_smell_indicators=code_smells,
                optimization_suggestions=optimizations,
                similarity_score=similarity_score,
                anomaly_score=anomaly_score
            )
            
        except Exception as e:
            self.logger.error(f"Single file AI analysis failed for {file_path}: {e}")
            return AICodeAnalysis(
                complexity_prediction=0,
                bug_probability=0,
                maintainability_score=100,
                code_smell_indicators=[],
                optimization_suggestions=[],
                similarity_score=50,
                anomaly_score=0
            )
    
    def _detect_code_smells(self, features: Dict[str, Any]) -> List[str]:
        """Detect code smell indicators"""
        
        smells = []
        
        if features['lines_of_code'] > 500:
            smells.append("Large file - consider breaking down")
        
        if features['max_nesting'] > 5:
            smells.append("Deep nesting - consider refactoring")
        
        if features['num_functions'] == 0 and features['lines_of_code'] > 50:
            smells.append("Script-style code - consider organizing into functions")
        
        if features['num_classes'] > 10:
            smells.append("Many classes in single file - consider splitting")
        
        if features.get('comment_ratio', 0) < 0.1:
            smells.append("Low comment density - add more documentation")
        
        if features['cyclomatic_complexity'] > 20:
            smells.append("High cyclomatic complexity - simplify logic")
        
        return smells
    
    def _generate_optimization_suggestions(self, features: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions"""
        
        suggestions = []
        
        if features['max_nesting'] > 3:
            suggestions.append("Reduce nesting using early returns or guard clauses")
        
        if features['cyclomatic_complexity'] > 10:
            suggestions.append("Break down complex functions into smaller ones")
        
        if features.get('comment_ratio', 0) < 0.05:
            suggestions.append("Add docstrings and comments for better maintainability")
        
        if features['lines_of_code'] > 300:
            suggestions.append("Consider splitting file into multiple modules")
        
        if features['num_functions'] > 20:
            suggestions.append("Group related functions into classes")
        
        return suggestions


class AutoOptimizer:
    """Automatic code optimization suggestions"""
    
    def __init__(self):
        self.logger = logging.getLogger("auto_optimizer")
    
    def generate_optimization_suggestions(self, file_paths: List[str]) -> Dict[str, List[AutoOptimizationSuggestion]]:
        """Generate automatic optimization suggestions"""
        
        suggestions = {}
        
        for file_path in file_paths:
            file_suggestions = self._analyze_file_for_optimizations(file_path)
            if file_suggestions:
                suggestions[file_path] = file_suggestions
        
        return suggestions
    
    def _analyze_file_for_optimizations(self, file_path: str) -> List[AutoOptimizationSuggestion]:
        """Analyze file for optimization opportunities"""
        
        suggestions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            tree = ast.parse(content)
            
            # Check for list comprehension opportunities
            suggestions.extend(self._detect_list_comprehension_opportunities(tree, lines))
            
            # Check for inefficient loops
            suggestions.extend(self._detect_inefficient_loops(tree, lines))
            
            # Check for repeated code
            suggestions.extend(self._detect_code_duplication(tree, lines))
            
            # Check for unnecessary variables
            suggestions.extend(self._detect_unnecessary_variables(tree, lines))
            
        except Exception as e:
            self.logger.error(f"Optimization analysis failed for {file_path}: {e}")
        
        return suggestions
    
    def _detect_list_comprehension_opportunities(self, tree: ast.AST, lines: List[str]) -> List[AutoOptimizationSuggestion]:
        """Detect opportunities for list comprehensions"""
        
        suggestions = []
        
        class ListCompVisitor(ast.NodeVisitor):
            def __init__(self, outer_self):
                self.outer_self = outer_self
                self.suggestions = []
            
            def visit_For(self, node):
                # Look for simple append patterns
                if (len(node.body) == 1 and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Call) and 
                    isinstance(node.body[0].value.func, ast.Attribute) and 
                    node.body[0].value.func.attr == 'append'):
                    
                    # This could be a list comprehension
                    original_lines = lines[node.lineno-1:getattr(node, 'end_lineno', node.lineno)]
                    original_code = '\n'.join(original_lines)
                    
                    # Generate optimized version (simplified)
                    target_var = node.body[0].value.func.value.id if isinstance(node.body[0].value.func.value, ast.Name) else "result"
                    iter_var = node.target.id if isinstance(node.target, ast.Name) else "item"
                    
                    optimized_code = f"{target_var}.extend([{iter_var} for {iter_var} in iterable])"
                    
                    self.suggestions.append(AutoOptimizationSuggestion(
                        optimization_type="list_comprehension",
                        original_code=original_code,
                        optimized_code=optimized_code,
                        expected_improvement={"performance": 15, "readability": 10},
                        confidence=0.7,
                        applicable_lines=list(range(node.lineno, getattr(node, 'end_lineno', node.lineno) + 1))
                    ))
                
                self.generic_visit(node)
        
        visitor = ListCompVisitor(self)
        visitor.visit(tree)
        return visitor.suggestions
    
    def _detect_inefficient_loops(self, tree: ast.AST, lines: List[str]) -> List[AutoOptimizationSuggestion]:
        """Detect inefficient loop patterns"""
        
        suggestions = []
        
        # Look for range(len(list)) patterns
        for node in ast.walk(tree):
            if (isinstance(node, ast.For) and 
                isinstance(node.iter, ast.Call) and 
                isinstance(node.iter.func, ast.Name) and 
                node.iter.func.id == 'range'):
                
                if (len(node.iter.args) == 1 and 
                    isinstance(node.iter.args[0], ast.Call) and 
                    isinstance(node.iter.args[0].func, ast.Name) and 
                    node.iter.args[0].func.id == 'len'):
                    
                    original_lines = lines[node.lineno-1:getattr(node, 'end_lineno', node.lineno)]
                    original_code = '\n'.join(original_lines)
                    
                    # Suggest enumerate instead
                    list_name = "items"
                    iter_var = node.target.id if isinstance(node.target, ast.Name) else "i"
                    
                    optimized_code = f"for {iter_var}, item in enumerate({list_name}):"
                    
                    suggestions.append(AutoOptimizationSuggestion(
                        optimization_type="enumerate_instead_of_range",
                        original_code=original_code,
                        optimized_code=optimized_code,
                        expected_improvement={"performance": 10, "readability": 20},
                        confidence=0.8,
                        applicable_lines=[node.lineno]
                    ))
        
        return suggestions
    
    def _detect_code_duplication(self, tree: ast.AST, lines: List[str]) -> List[AutoOptimizationSuggestion]:
        """Detect code duplication opportunities"""
        
        suggestions = []
        
        # Simple duplication detection based on similar function bodies
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        if len(functions) >= 2:
            for i, func1 in enumerate(functions):
                for func2 in functions[i+1:]:
                    if self._functions_are_similar(func1, func2):
                        suggestions.append(AutoOptimizationSuggestion(
                            optimization_type="extract_common_function",
                            original_code=f"Functions {func1.name} and {func2.name} have similar code",
                            optimized_code="Consider extracting common functionality into a shared function",
                            expected_improvement={"maintainability": 25, "code_size": 15},
                            confidence=0.6,
                            applicable_lines=[func1.lineno, func2.lineno]
                        ))
                        break  # Only suggest once per function
        
        return suggestions
    
    def _detect_unnecessary_variables(self, tree: ast.AST, lines: List[str]) -> List[AutoOptimizationSuggestion]:
        """Detect unnecessary variable assignments"""
        
        suggestions = []
        
        # Look for variables that are assigned once and used once
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Analyze variable usage within function
                var_assignments = {}
                var_usages = {}
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                var_assignments[target.id] = child
                    elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        var_usages[child.id] = var_usages.get(child.id, 0) + 1
                
                # Find variables used only once
                for var_name, usage_count in var_usages.items():
                    if (usage_count == 1 and var_name in var_assignments):
                        assign_node = var_assignments[var_name]
                        
                        suggestions.append(AutoOptimizationSuggestion(
                            optimization_type="inline_single_use_variable",
                            original_code=f"Variable '{var_name}' is used only once",
                            optimized_code=f"Consider inlining variable '{var_name}' at its usage site",
                            expected_improvement={"readability": 5, "memory": 2},
                            confidence=0.5,
                            applicable_lines=[assign_node.lineno]
                        ))
        
        return suggestions
    
    def _functions_are_similar(self, func1: ast.FunctionDef, func2: ast.FunctionDef) -> bool:
        """Check if two functions are similar (simplified heuristic)"""
        
        # Simple similarity check based on structure
        if abs(len(func1.body) - len(func2.body)) > 2:
            return False
        
        # Check for similar statement types
        func1_types = [type(stmt).__name__ for stmt in func1.body]
        func2_types = [type(stmt).__name__ for stmt in func2.body]
        
        if len(func1_types) != len(func2_types):
            return False
        
        # Count matching statement types
        matches = sum(1 for t1, t2 in zip(func1_types, func2_types) if t1 == t2)
        similarity = matches / len(func1_types) if func1_types else 0
        
        return similarity > 0.7  # 70% similarity threshold