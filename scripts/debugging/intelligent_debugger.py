"""
Intelligent Debugging System for Research Experiments
Provides automatic error detection, diagnosis, and self-healing capabilities
"""

import os
import sys
import traceback
import inspect
import logging
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import threading
import functools
import pickle
import hashlib

# Statistical analysis imports
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class DebugEvent:
    """Single debugging event record"""
    timestamp: datetime
    event_type: str  # 'error', 'warning', 'anomaly', 'performance', 'data_quality'
    severity: str    # 'critical', 'high', 'medium', 'low'
    component: str   # Which part of the code/system
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    suggested_fixes: List[str] = None
    auto_healing_applied: bool = False

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str  # 'statistical', 'performance', 'data_quality'
    description: str
    expected_range: Tuple[float, float]
    actual_value: float
    confidence: float

class IntelligentDebugger:
    """
    Main intelligent debugging system that provides comprehensive error detection,
    diagnosis, and self-healing capabilities for research experiments
    """
    
    def __init__(self, 
                 output_dir: str = "debug_reports",
                 enable_anomaly_detection: bool = True,
                 enable_self_healing: bool = True,
                 enable_profiling: bool = True,
                 anomaly_sensitivity: float = 0.05):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_self_healing = enable_self_healing
        self.enable_profiling = enable_profiling
        self.anomaly_sensitivity = anomaly_sensitivity
        
        # State tracking
        self.debug_events: List[DebugEvent] = []
        self.historical_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics: Dict[str, Dict] = {}
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.healing_attempts: Dict[str, int] = defaultdict(int)
        
        # Statistical models for anomaly detection
        self.anomaly_detectors: Dict[str, Any] = {}
        if SKLEARN_AVAILABLE:
            self.isolation_forest = IsolationForest(contamination=anomaly_sensitivity, random_state=42)
        
        # Performance profiling
        self.profiling_data: Dict[str, List] = defaultdict(list)
        self.execution_traces: List[Dict] = []
        
        # Setup logging
        self.logger = self._setup_intelligent_logging()
        
        # Install global error handlers
        self._install_global_handlers()
        
        self.logger.info("IntelligentDebugger initialized with advanced diagnostics")
    
    def _setup_intelligent_logging(self) -> logging.Logger:
        """Setup intelligent context-aware logging system"""
        
        logger = logging.getLogger("intelligent_debugger")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create custom formatter with context
        class ContextAwareFormatter(logging.Formatter):
            def format(self, record):
                # Add debugging context to log records
                if hasattr(record, 'debug_context'):
                    context_str = f" [{record.debug_context}]"
                else:
                    context_str = ""
                
                # Add function and line info
                if hasattr(record, 'funcName') and hasattr(record, 'lineno'):
                    location_str = f" ({record.funcName}:{record.lineno})"
                else:
                    location_str = ""
                
                # Format timestamp
                timestamp = self.formatTime(record, self.datefmt)
                
                # Create final message
                formatted_msg = f"{timestamp} - {record.levelname}{context_str}{location_str} - {record.getMessage()}"
                
                # Add stack trace for errors
                if record.exc_info:
                    formatted_msg += f"\n{self.formatException(record.exc_info)}"
                
                return formatted_msg
        
        formatter = ContextAwareFormatter(
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for detailed logs
        log_file = self.output_dir / f"intelligent_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _install_global_handlers(self):
        """Install global error and warning handlers"""
        
        # Install custom exception handler
        original_excepthook = sys.excepthook
        
        def debug_excepthook(exc_type, exc_value, exc_traceback):
            # Log the exception with our intelligent system
            self.handle_exception(exc_type, exc_value, exc_traceback)
            
            # Call original handler
            original_excepthook(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = debug_excepthook
        
        # Install warning handler
        def debug_warning_handler(message, category, filename, lineno, file=None, line=None):
            self.handle_warning(message, category, filename, lineno)
        
        warnings.showwarning = debug_warning_handler
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle exceptions with intelligent analysis"""
        
        # Extract stack trace information
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        stack_trace = ''.join(tb_lines)
        
        # Analyze the exception
        error_analysis = self._analyze_exception(exc_type, exc_value, stack_trace)
        
        # Create debug event
        debug_event = DebugEvent(
            timestamp=datetime.now(),
            event_type='error',
            severity='critical',
            component=error_analysis['component'],
            message=f"{exc_type.__name__}: {exc_value}",
            context=error_analysis['context'],
            stack_trace=stack_trace,
            suggested_fixes=error_analysis['suggested_fixes']
        )
        
        # Try self-healing if enabled
        if self.enable_self_healing:
            healing_applied = self._attempt_self_healing(debug_event)
            debug_event.auto_healing_applied = healing_applied
        
        self.debug_events.append(debug_event)
        self.error_patterns[error_analysis['error_pattern']] += 1
        
        # Log with context
        self.logger.error(
            f"Exception detected: {debug_event.message}",
            extra={'debug_context': error_analysis['component']}
        )
        
        # Generate immediate error report if critical
        if debug_event.severity == 'critical':
            self._generate_immediate_error_report(debug_event)
    
    def handle_warning(self, message, category, filename, lineno):
        """Handle warnings with intelligent analysis"""
        
        warning_analysis = self._analyze_warning(message, category, filename, lineno)
        
        debug_event = DebugEvent(
            timestamp=datetime.now(),
            event_type='warning',
            severity=warning_analysis['severity'],
            component=warning_analysis['component'],
            message=str(message),
            context=warning_analysis['context'],
            suggested_fixes=warning_analysis['suggested_fixes']
        )
        
        self.debug_events.append(debug_event)
        
        # Log with appropriate level
        log_level = logging.WARNING if debug_event.severity in ['high', 'critical'] else logging.INFO
        self.logger.log(
            log_level,
            f"Warning detected: {debug_event.message}",
            extra={'debug_context': warning_analysis['component']}
        )
    
    def _analyze_exception(self, exc_type, exc_value, stack_trace) -> Dict[str, Any]:
        """Analyze exception to provide intelligent diagnosis"""
        
        analysis = {
            'component': 'unknown',
            'context': {},
            'suggested_fixes': [],
            'error_pattern': 'generic'
        }
        
        error_message = str(exc_value).lower()
        exc_name = exc_type.__name__
        
        # Memory-related errors
        if exc_name == 'MemoryError' or 'memory' in error_message:
            analysis.update({
                'component': 'memory_management',
                'error_pattern': 'memory_error',
                'suggested_fixes': [
                    "Reduce batch size or data chunk size",
                    "Clear unused variables with del statement",
                    "Use data generators instead of loading all data",
                    "Enable garbage collection with gc.collect()",
                    "Consider using data streaming or pagination"
                ]
            })
        
        # CUDA/GPU errors
        elif 'cuda' in error_message or 'gpu' in error_message or exc_name == 'RuntimeError':
            if 'cuda' in error_message:
                analysis.update({
                    'component': 'gpu_computation',
                    'error_pattern': 'cuda_error',
                    'suggested_fixes': [
                        "Check CUDA memory with torch.cuda.memory_summary()",
                        "Clear GPU cache with torch.cuda.empty_cache()",
                        "Reduce model size or batch size",
                        "Move tensors to CPU if GPU memory is full",
                        "Check CUDA installation and compatibility"
                    ]
                })
        
        # Data-related errors
        elif exc_name in ['KeyError', 'IndexError', 'ValueError'] and any(word in error_message for word in ['shape', 'size', 'dimension']):
            analysis.update({
                'component': 'data_processing',
                'error_pattern': 'data_shape_error',
                'suggested_fixes': [
                    "Check input data shapes and dimensions",
                    "Verify data preprocessing steps",
                    "Add shape validation before processing",
                    "Check for missing or NaN values",
                    "Ensure consistent data types"
                ]
            })
        
        # File/IO errors
        elif exc_name in ['FileNotFoundError', 'PermissionError', 'IOError']:
            analysis.update({
                'component': 'file_io',
                'error_pattern': 'io_error',
                'suggested_fixes': [
                    "Verify file paths exist",
                    "Check file permissions",
                    "Ensure directories are created",
                    "Handle file locking issues",
                    "Add retry mechanism for file operations"
                ]
            })
        
        # Network/connection errors
        elif any(word in error_message for word in ['connection', 'timeout', 'network', 'socket']):
            analysis.update({
                'component': 'network_communication',
                'error_pattern': 'network_error',
                'suggested_fixes': [
                    "Check internet connection",
                    "Increase timeout values",
                    "Add retry mechanism with exponential backoff",
                    "Verify API endpoints and credentials",
                    "Handle rate limiting"
                ]
            })
        
        # Import/module errors
        elif exc_name in ['ImportError', 'ModuleNotFoundError']:
            analysis.update({
                'component': 'dependencies',
                'error_pattern': 'import_error',
                'suggested_fixes': [
                    f"Install missing package: pip install {error_message.split()[-1] if error_message.split() else 'package_name'}",
                    "Check virtual environment activation",
                    "Verify Python path and package installation",
                    "Update package versions",
                    "Check for circular imports"
                ]
            })
        
        # Add context from stack trace
        analysis['context']['stack_trace_analysis'] = self._analyze_stack_trace(stack_trace)
        
        return analysis
    
    def _analyze_warning(self, message, category, filename, lineno) -> Dict[str, Any]:
        """Analyze warnings for intelligent handling"""
        
        analysis = {
            'component': 'general',
            'context': {
                'filename': filename,
                'lineno': lineno,
                'category': category.__name__
            },
            'severity': 'medium',
            'suggested_fixes': []
        }
        
        message_str = str(message).lower()
        
        # Deprecation warnings
        if 'deprecat' in message_str:
            analysis.update({
                'component': 'deprecated_api',
                'severity': 'medium',
                'suggested_fixes': [
                    "Update to newer API version",
                    "Check library documentation for alternatives",
                    "Plan migration strategy for deprecated features"
                ]
            })
        
        # Performance warnings
        elif any(word in message_str for word in ['slow', 'performance', 'inefficient']):
            analysis.update({
                'component': 'performance',
                'severity': 'low',
                'suggested_fixes': [
                    "Profile code to identify bottlenecks",
                    "Consider vectorized operations",
                    "Optimize data structures",
                    "Use caching for repeated computations"
                ]
            })
        
        # Data conversion warnings
        elif any(word in message_str for word in ['conversion', 'dtype', 'casting']):
            analysis.update({
                'component': 'data_types',
                'severity': 'low',
                'suggested_fixes': [
                    "Explicitly specify data types",
                    "Add data type validation",
                    "Handle type conversions explicitly"
                ]
            })
        
        return analysis
    
    def _analyze_stack_trace(self, stack_trace: str) -> Dict[str, Any]:
        """Analyze stack trace for patterns and insights"""
        
        lines = stack_trace.split('\n')
        
        analysis = {
            'depth': len([line for line in lines if 'File "' in line]),
            'user_code_involved': False,
            'library_errors': [],
            'potential_infinite_recursion': False
        }
        
        # Check for user code involvement
        user_code_patterns = ['main.py', 'experiment', 'scripts/', 'src/']
        for line in lines:
            if any(pattern in line for pattern in user_code_patterns):
                analysis['user_code_involved'] = True
                break
        
        # Identify library errors
        library_patterns = ['site-packages', 'torch', 'sklearn', 'pandas', 'numpy']
        for line in lines:
            for pattern in library_patterns:
                if pattern in line:
                    analysis['library_errors'].append(pattern)
        
        # Check for potential infinite recursion
        if analysis['depth'] > 50:
            analysis['potential_infinite_recursion'] = True
        
        return analysis
    
    def detect_statistical_anomalies(self, metrics: Dict[str, float], metric_name: str = "experiment") -> List[AnomalyDetection]:
        """Detect statistical anomalies in experiment metrics"""
        
        if not self.enable_anomaly_detection:
            return []
        
        anomalies = []
        
        for metric_key, value in metrics.items():
            # Store metric for historical analysis
            self.historical_metrics[metric_key].append(value)
            
            # Skip if we don't have enough historical data
            if len(self.historical_metrics[metric_key]) < 10:
                continue
            
            # Get historical values
            historical_values = list(self.historical_metrics[metric_key])
            
            # Statistical anomaly detection using z-score
            if len(historical_values) >= 5:
                mean_val = np.mean(historical_values[:-1])  # Exclude current value
                std_val = np.std(historical_values[:-1])
                
                if std_val > 0:
                    z_score = abs((value - mean_val) / std_val)
                    
                    # Z-score threshold (3 standard deviations)
                    if z_score > 3:
                        anomalies.append(AnomalyDetection(
                            is_anomaly=True,
                            anomaly_score=z_score,
                            anomaly_type='statistical',
                            description=f"{metric_key} value {value:.4f} is {z_score:.2f} standard deviations from mean {mean_val:.4f}",
                            expected_range=(mean_val - 3*std_val, mean_val + 3*std_val),
                            actual_value=value,
                            confidence=min(0.99, (z_score - 3) / 10 + 0.5)
                        ))
            
            # Isolation Forest anomaly detection if sklearn available
            if SKLEARN_AVAILABLE and len(historical_values) >= 20:
                try:
                    # Reshape for sklearn
                    X = np.array(historical_values).reshape(-1, 1)
                    
                    # Fit and predict
                    self.isolation_forest.fit(X[:-1])  # Exclude current value
                    anomaly_score = self.isolation_forest.decision_function([[value]])[0]
                    is_outlier = self.isolation_forest.predict([[value]])[0] == -1
                    
                    if is_outlier:
                        anomalies.append(AnomalyDetection(
                            is_anomaly=True,
                            anomaly_score=-anomaly_score,  # Convert to positive
                            anomaly_type='isolation_forest',
                            description=f"{metric_key} identified as outlier by isolation forest",
                            expected_range=(min(historical_values), max(historical_values)),
                            actual_value=value,
                            confidence=min(0.95, abs(anomaly_score))
                        ))
                        
                except Exception as e:
                    self.logger.debug(f"Isolation forest anomaly detection failed: {e}")
        
        # Log anomalies
        for anomaly in anomalies:
            debug_event = DebugEvent(
                timestamp=datetime.now(),
                event_type='anomaly',
                severity='medium' if anomaly.confidence > 0.8 else 'low',
                component='statistical_analysis',
                message=anomaly.description,
                context={
                    'metric_name': metric_name,
                    'anomaly_score': anomaly.anomaly_score,
                    'confidence': anomaly.confidence,
                    'anomaly_type': anomaly.anomaly_type
                }
            )
            self.debug_events.append(debug_event)
            
            self.logger.warning(
                f"Statistical anomaly detected: {anomaly.description}",
                extra={'debug_context': 'anomaly_detection'}
            )
        
        return anomalies
    
    def detect_performance_bottlenecks(self, execution_time: float, 
                                     component: str = "unknown",
                                     memory_usage: float = None,
                                     cpu_usage: float = None,
                                     io_operations: int = None) -> List[str]:
        """Advanced performance bottleneck detection with resource monitoring"""
        
        bottlenecks = []
        
        # Get system resource info
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            current_cpu = process.cpu_percent()
            system_memory = psutil.virtual_memory().percent
            system_cpu = psutil.cpu_percent()
            disk_io = psutil.disk_io_counters()
            PSUTIL_AVAILABLE = True
        except ImportError:
            current_memory = memory_usage
            current_cpu = cpu_usage
            system_memory = None
            system_cpu = None
            disk_io = None
            PSUTIL_AVAILABLE = False
        
        # Store comprehensive performance data
        performance_entry = {
            'timestamp': datetime.now(),
            'execution_time': execution_time,
            'component': component,
            'memory_usage_mb': current_memory,
            'cpu_usage_percent': current_cpu,
            'system_memory_percent': system_memory,
            'system_cpu_percent': system_cpu,
            'io_operations': io_operations
        }
        
        if disk_io:
            performance_entry.update({
                'disk_read_mb': disk_io.read_bytes / 1024 / 1024,
                'disk_write_mb': disk_io.write_bytes / 1024 / 1024,
                'disk_read_count': disk_io.read_count,
                'disk_write_count': disk_io.write_count
            })
        
        self.profiling_data[component].append(performance_entry)
        
        # Analyze performance trends
        recent_entries = self.profiling_data[component][-10:]
        
        if len(recent_entries) >= 3:
            # Time-based analysis
            recent_times = [entry['execution_time'] for entry in recent_entries]
            avg_time = np.mean(recent_times[:-1])  # Exclude current
            current_slowdown = execution_time / avg_time if avg_time > 0 else 1
            
            # Memory-based analysis
            if current_memory:
                recent_memory = [entry.get('memory_usage_mb', 0) for entry in recent_entries if entry.get('memory_usage_mb')]
                if recent_memory:
                    avg_memory = np.mean(recent_memory[:-1])
                    memory_increase = (current_memory - avg_memory) / avg_memory if avg_memory > 0 else 0
                    
                    # Detect memory leaks
                    if memory_increase > 0.5:  # 50% increase
                        bottlenecks.append(f"{component} memory usage increased {memory_increase*100:.1f}% (possible memory leak)")
                    
                    # Detect high absolute memory usage
                    if current_memory > 1024:  # 1GB
                        bottlenecks.append(f"{component} using {current_memory:.1f}MB memory (high usage)")
            
            # CPU-based analysis
            if current_cpu and current_cpu > 80:
                bottlenecks.append(f"{component} using {current_cpu:.1f}% CPU (high usage)")
            
            # System resource analysis
            if PSUTIL_AVAILABLE:
                if system_memory and system_memory > 90:
                    bottlenecks.append(f"System memory usage at {system_memory:.1f}% (system bottleneck)")
                
                if system_cpu and system_cpu > 90:
                    bottlenecks.append(f"System CPU usage at {system_cpu:.1f}% (system bottleneck)")
            
            # Execution time analysis
            if current_slowdown > 2.0:
                bottlenecks.append(f"{component} is running {current_slowdown:.1f}x slower than average")
                
                # Advanced suggestions based on bottleneck type
                suggestions = self._generate_performance_suggestions(
                    execution_time, current_memory, current_cpu, system_memory, system_cpu
                )
                
                debug_event = DebugEvent(
                    timestamp=datetime.now(),
                    event_type='performance',
                    severity='high' if current_slowdown > 5.0 else 'medium',
                    component=component,
                    message=f"Performance degradation detected: {current_slowdown:.1f}x slowdown",
                    context={
                        'execution_time': execution_time,
                        'average_time': avg_time,
                        'slowdown_factor': current_slowdown,
                        'memory_usage_mb': current_memory,
                        'cpu_usage_percent': current_cpu,
                        'system_memory_percent': system_memory,
                        'system_cpu_percent': system_cpu
                    },
                    suggested_fixes=suggestions
                )
                self.debug_events.append(debug_event)
        
        # Absolute performance thresholds
        threshold_bottlenecks = self._check_performance_thresholds(
            execution_time, current_memory, current_cpu, component
        )
        bottlenecks.extend(threshold_bottlenecks)
        
        # Algorithm complexity analysis
        complexity_analysis = self._analyze_algorithmic_complexity(component, execution_time)
        if complexity_analysis:
            bottlenecks.append(complexity_analysis)
        
        return bottlenecks
    
    def _generate_performance_suggestions(self, execution_time: float, memory_mb: float, 
                                        cpu_percent: float, system_memory: float, 
                                        system_cpu: float) -> List[str]:
        """Generate intelligent performance optimization suggestions"""
        
        suggestions = []
        
        # Memory-based suggestions
        if memory_mb and memory_mb > 1024:  # > 1GB
            suggestions.extend([
                "Consider reducing data batch sizes",
                "Implement data generators instead of loading all data",
                "Use memory mapping for large datasets",
                "Clear intermediate variables with del statement",
                "Enable memory profiling to identify memory hotspots"
            ])
        
        # CPU-based suggestions  
        if cpu_percent and cpu_percent > 80:
            suggestions.extend([
                "Profile CPU usage to identify computational bottlenecks",
                "Consider vectorized operations (NumPy, Pandas)",
                "Implement parallel processing with multiprocessing",
                "Use compiled libraries (Numba, Cython) for hot loops",
                "Optimize algorithms for better time complexity"
            ])
        
        # System-level suggestions
        if system_memory and system_memory > 90:
            suggestions.extend([
                "Close unnecessary applications to free system memory",
                "Consider using cloud computing with more RAM",
                "Implement data streaming or chunking strategies"
            ])
        
        if system_cpu and system_cpu > 90:
            suggestions.extend([
                "Reduce concurrent processes",
                "Use task scheduling to avoid peak CPU times",
                "Consider distributed computing for CPU-intensive tasks"
            ])
        
        # Time-based suggestions
        if execution_time > 300:  # 5+ minutes
            suggestions.extend([
                "Implement checkpointing to save intermediate results",
                "Add progress monitoring and early stopping",
                "Break long-running tasks into smaller chunks",
                "Consider asynchronous processing"
            ])
        
        # Default performance suggestions
        if not suggestions:
            suggestions = [
                "Profile code to identify specific bottlenecks",
                "Check system resource usage (CPU, memory, I/O)",
                "Optimize algorithms or data structures", 
                "Consider parallel processing",
                "Clear caches or restart services if needed"
            ]
        
        return suggestions
    
    def _check_performance_thresholds(self, execution_time: float, memory_mb: float,
                                    cpu_percent: float, component: str) -> List[str]:
        """Check against absolute performance thresholds"""
        
        bottlenecks = []
        
        # Execution time thresholds
        if execution_time > 1800:  # 30 minutes
            bottlenecks.append(f"{component} took {execution_time/60:.1f} minutes (extremely long)")
        elif execution_time > 600:  # 10 minutes  
            bottlenecks.append(f"{component} took {execution_time/60:.1f} minutes (very long)")
        elif execution_time > 300:  # 5 minutes
            bottlenecks.append(f"{component} took {execution_time:.1f}s (unusually long)")
        
        # Memory thresholds
        if memory_mb:
            if memory_mb > 8192:  # 8GB
                bottlenecks.append(f"{component} using {memory_mb/1024:.1f}GB memory (extremely high)")
            elif memory_mb > 4096:  # 4GB
                bottlenecks.append(f"{component} using {memory_mb/1024:.1f}GB memory (very high)")
            elif memory_mb > 2048:  # 2GB
                bottlenecks.append(f"{component} using {memory_mb:.1f}MB memory (high)")
        
        # CPU thresholds
        if cpu_percent:
            if cpu_percent > 95:
                bottlenecks.append(f"{component} using {cpu_percent:.1f}% CPU (maxing out CPU)")
            elif cpu_percent > 80:
                bottlenecks.append(f"{component} using {cpu_percent:.1f}% CPU (high CPU usage)")
        
        return bottlenecks
    
    def _analyze_algorithmic_complexity(self, component: str, execution_time: float) -> Optional[str]:
        """Analyze potential algorithmic complexity issues"""
        
        # Track execution times for pattern analysis
        if component not in self.profiling_data or len(self.profiling_data[component]) < 5:
            return None
        
        recent_times = [entry['execution_time'] for entry in self.profiling_data[component][-5:]]
        
        # Check for exponential growth pattern
        if len(recent_times) >= 3:
            growth_ratios = [recent_times[i+1]/recent_times[i] for i in range(len(recent_times)-1) if recent_times[i] > 0]
            
            if growth_ratios and np.mean(growth_ratios) > 2.0:
                return f"{component} shows exponential time growth pattern (possible O(2^n) or O(n!) complexity)"
            
            # Check for quadratic growth
            elif growth_ratios and np.mean(growth_ratios) > 1.5:
                return f"{component} shows rapid time growth (possible O(n^2) or higher complexity)"
        
        return None
    
    def detect_data_quality_issues(self, data: Union[np.ndarray, pd.DataFrame], 
                                 data_name: str = "dataset",
                                 expected_schema: dict = None,
                                 quality_thresholds: dict = None) -> Dict[str, Any]:
        """Advanced data quality detection with comprehensive analysis"""
        
        # Default quality thresholds
        default_thresholds = {
            'missing_threshold': 20,      # % missing values threshold
            'outlier_threshold': 5,       # % outliers threshold  
            'duplicate_threshold': 1,     # % duplicates threshold
            'cardinality_threshold': 0.95, # High cardinality threshold
            'skewness_threshold': 2.0,    # Skewness threshold
            'correlation_threshold': 0.95  # High correlation threshold
        }
        
        thresholds = {**default_thresholds, **(quality_thresholds or {})}
        
        quality_report = {
            'data_name': data_name,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'recommendations': [],
            'severity_score': 0  # 0-100 scale
        }
        
        try:
            if isinstance(data, pd.DataFrame):
                quality_report.update(self._analyze_dataframe_quality(data, thresholds, expected_schema))
            elif isinstance(data, np.ndarray):
                quality_report.update(self._analyze_array_quality(data, thresholds))
            else:
                quality_report['issues'].append(f"Unsupported data type: {type(data)}")
                quality_report['severity_score'] = 50
        
        except Exception as e:
            quality_report['issues'].append(f"Error during data quality analysis: {str(e)}")
            quality_report['severity_score'] = 75
        
        # Calculate overall severity score
        if not quality_report['severity_score']:
            quality_report['severity_score'] = self._calculate_severity_score(quality_report)
        
        # Generate recommendations
        quality_report['recommendations'] = self._generate_data_quality_recommendations(quality_report)
        
        # Log data quality issues
        if quality_report['issues'] or quality_report['warnings']:
            severity = self._determine_severity(quality_report['severity_score'])
            
            debug_event = DebugEvent(
                timestamp=datetime.now(),
                event_type='data_quality',
                severity=severity,
                component='data_validation',
                message=f"Data quality analysis completed for {data_name}",
                context=quality_report,
                suggested_fixes=quality_report['recommendations']
            )
            self.debug_events.append(debug_event)
        
        return quality_report
    
    def _analyze_dataframe_quality(self, df: pd.DataFrame, thresholds: dict, 
                                 expected_schema: dict = None) -> Dict[str, Any]:
        """Comprehensive DataFrame quality analysis"""
        
        analysis = {
            'issues': [],
            'warnings': [],
            'statistics': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'dtypes': df.dtypes.to_dict()
            }
        }
        
        # Schema validation
        if expected_schema:
            schema_issues = self._validate_schema(df, expected_schema)
            analysis['issues'].extend(schema_issues)
        
        # Missing values analysis
        missing_analysis = self._analyze_missing_values(df, thresholds['missing_threshold'])
        analysis['issues'].extend(missing_analysis['issues'])
        analysis['warnings'].extend(missing_analysis['warnings'])
        analysis['statistics']['missing_values'] = missing_analysis['statistics']
        
        # Duplicate analysis
        duplicate_analysis = self._analyze_duplicates(df, thresholds['duplicate_threshold'])
        analysis['issues'].extend(duplicate_analysis['issues'])
        analysis['statistics']['duplicates'] = duplicate_analysis['statistics']
        
        # Data type consistency
        dtype_issues = self._analyze_data_types(df)
        analysis['issues'].extend(dtype_issues)
        
        # Statistical analysis for numeric columns
        numeric_analysis = self._analyze_numeric_columns(df, thresholds)
        analysis['issues'].extend(numeric_analysis['issues'])
        analysis['warnings'].extend(numeric_analysis['warnings'])
        analysis['statistics']['numeric_stats'] = numeric_analysis['statistics']
        
        # Categorical columns analysis
        categorical_analysis = self._analyze_categorical_columns(df, thresholds)
        analysis['issues'].extend(categorical_analysis['issues'])
        analysis['warnings'].extend(categorical_analysis['warnings'])
        analysis['statistics']['categorical_stats'] = categorical_analysis['statistics']
        
        # Correlation analysis
        correlation_analysis = self._analyze_correlations(df, thresholds['correlation_threshold'])
        analysis['warnings'].extend(correlation_analysis['warnings'])
        analysis['statistics']['correlations'] = correlation_analysis['statistics']
        
        # Data distribution analysis
        distribution_analysis = self._analyze_distributions(df, thresholds)
        analysis['warnings'].extend(distribution_analysis['warnings'])
        analysis['statistics']['distributions'] = distribution_analysis['statistics']
        
        return analysis
    
    def _analyze_array_quality(self, arr: np.ndarray, thresholds: dict) -> Dict[str, Any]:
        """Comprehensive NumPy array quality analysis"""
        
        analysis = {
            'issues': [],
            'warnings': [],
            'statistics': {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'memory_usage_mb': arr.nbytes / 1024 / 1024
            }
        }
        
        # Basic data integrity checks
        if arr.dtype.kind in 'fc':  # Float or complex
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            
            if nan_count > 0:
                nan_pct = nan_count / arr.size * 100
                if nan_pct > thresholds['missing_threshold']:
                    analysis['issues'].append(f"High NaN percentage: {nan_pct:.1f}%")
                else:
                    analysis['warnings'].append(f"{nan_count} NaN values found")
                    
            if inf_count > 0:
                inf_pct = inf_count / arr.size * 100
                analysis['issues'].append(f"{inf_count} infinite values ({inf_pct:.1f}%)")
                
            analysis['statistics']['nan_count'] = int(nan_count)
            analysis['statistics']['inf_count'] = int(inf_count)
        
        # Statistical analysis for numeric arrays
        if arr.dtype.kind in 'fc' and arr.size > 0:
            flat_data = arr.flatten()
            valid_data = flat_data[np.isfinite(flat_data)]
            
            if len(valid_data) > 0:
                stats = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'median': float(np.median(valid_data))
                }
                
                analysis['statistics']['numeric_stats'] = stats
                
                # Zero variance check
                if stats['std'] == 0:
                    analysis['issues'].append("Data has zero variance (all values identical)")
                
                # Extreme variance check
                elif stats['std'] > 1000 * abs(stats['mean']) and stats['mean'] != 0:
                    analysis['warnings'].append("Extremely high variance detected")
                
                # Range analysis
                data_range = stats['max'] - stats['min']
                if data_range == 0:
                    analysis['issues'].append("All values are identical")
                elif abs(stats['max']) > 1e10 or abs(stats['min']) > 1e10:
                    analysis['warnings'].append("Extremely large values detected")
        
        # Shape analysis
        if len(arr.shape) > 4:
            analysis['warnings'].append(f"High-dimensional data ({len(arr.shape)} dimensions)")
        
        # Memory usage analysis
        if analysis['statistics']['memory_usage_mb'] > 1024:  # 1GB
            analysis['warnings'].append(f"Large memory footprint: {analysis['statistics']['memory_usage_mb']:.1f}MB")
        
        return analysis
    
    def _analyze_missing_values(self, df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Analyze missing values pattern and severity"""
        
        analysis = {'issues': [], 'warnings': [], 'statistics': {}}
        
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        missing_pct_by_col = (missing_counts / len(df) * 100)
        
        # High missing value columns
        high_missing_cols = missing_pct_by_col[missing_pct_by_col > threshold].to_dict()
        if high_missing_cols:
            analysis['issues'].append(f"Columns with >{threshold}% missing: {high_missing_cols}")
        
        # Moderate missing value columns
        moderate_missing_cols = missing_pct_by_col[
            (missing_pct_by_col > 5) & (missing_pct_by_col <= threshold)
        ].to_dict()
        if moderate_missing_cols:
            analysis['warnings'].append(f"Columns with 5-{threshold}% missing: {moderate_missing_cols}")
        
        # Missing value patterns
        if total_missing > 0:
            # Check for systematic missing patterns
            missing_pattern = df.isnull().sum(axis=1)
            rows_with_missing = (missing_pattern > 0).sum()
            
            if rows_with_missing == len(df):
                analysis['issues'].append("Every row has at least one missing value")
            elif rows_with_missing > len(df) * 0.5:
                analysis['warnings'].append(f"{rows_with_missing} rows ({rows_with_missing/len(df)*100:.1f}%) have missing values")
        
        analysis['statistics'] = {
            'total_missing': int(total_missing),
            'missing_percentage': float(total_missing / (len(df) * len(df.columns)) * 100),
            'columns_with_missing': len(missing_counts[missing_counts > 0]),
            'missing_by_column': missing_pct_by_col.to_dict()
        }
        
        return analysis
    
    def _analyze_duplicates(self, df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Analyze duplicate rows and patterns"""
        
        analysis = {'issues': [], 'statistics': {}}
        
        # Full row duplicates
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        duplicate_pct = duplicate_count / len(df) * 100
        
        if duplicate_pct > threshold:
            analysis['issues'].append(f"{duplicate_count} duplicate rows ({duplicate_pct:.1f}%)")
        
        # Near-duplicates (same values except for one column)
        if len(df.columns) > 1:
            near_duplicates = 0
            for col in df.columns:
                temp_df = df.drop(columns=[col])
                near_dup_count = temp_df.duplicated().sum()
                near_duplicates = max(near_duplicates, near_dup_count)
            
            if near_duplicates > duplicate_count * 2:
                analysis['issues'].append(f"Potential near-duplicates detected: {near_duplicates}")
        
        analysis['statistics'] = {
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': float(duplicate_pct),
            'unique_rows': int(len(df) - duplicate_count)
        }
        
        return analysis
    
    def _analyze_numeric_columns(self, df: pd.DataFrame, thresholds: dict) -> Dict[str, Any]:
        """Analyze numeric columns for outliers and distributions"""
        
        analysis = {'issues': [], 'warnings': [], 'statistics': {}}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        col_stats = {}
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
                
            # Basic statistics
            stats = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis())
            }
            
            # Outlier detection using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_pct = len(outliers) / len(col_data) * 100
                
                stats['outlier_count'] = len(outliers)
                stats['outlier_percentage'] = outlier_pct
                
                if outlier_pct > thresholds['outlier_threshold']:
                    analysis['issues'].append(f"High outliers in '{col}': {outlier_pct:.1f}%")
            
            # Distribution analysis
            if abs(stats['skewness']) > thresholds['skewness_threshold']:
                analysis['warnings'].append(f"High skewness in '{col}': {stats['skewness']:.2f}")
            
            # Zero variance
            if stats['std'] == 0:
                analysis['issues'].append(f"Column '{col}' has zero variance")
            
            col_stats[col] = stats
        
        analysis['statistics'] = col_stats
        return analysis
    
    def _analyze_categorical_columns(self, df: pd.DataFrame, thresholds: dict) -> Dict[str, Any]:
        """Analyze categorical columns for cardinality and distribution"""
        
        analysis = {'issues': [], 'warnings': [], 'statistics': {}}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        col_stats = {}
        
        for col in categorical_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
                
            unique_count = col_data.nunique()
            total_count = len(col_data)
            cardinality_ratio = unique_count / total_count
            
            stats = {
                'unique_count': unique_count,
                'total_count': total_count,
                'cardinality_ratio': cardinality_ratio,
                'most_frequent': col_data.value_counts().head(3).to_dict()
            }
            
            # High cardinality check
            if cardinality_ratio > thresholds['cardinality_threshold']:
                analysis['warnings'].append(f"High cardinality in '{col}': {unique_count} unique values")
            
            # Single value check
            if unique_count == 1:
                analysis['issues'].append(f"Column '{col}' has only one unique value")
            
            # Check for potential ID columns (very high cardinality)
            if cardinality_ratio > 0.98 and unique_count > 100:
                analysis['warnings'].append(f"Column '{col}' may be an ID column (very high cardinality)")
            
            col_stats[col] = stats
        
        analysis['statistics'] = col_stats
        return analysis
    
    def _calculate_severity_score(self, quality_report: Dict) -> int:
        """Calculate overall data quality severity score (0-100)"""
        
        score = 0
        
        # Weight different issue types
        issue_weights = {
            'missing': 15,
            'duplicate': 10, 
            'type': 20,
            'outlier': 5,
            'variance': 15,
            'correlation': 5,
            'schema': 25,
            'distribution': 5
        }
        
        # Count issues by type (simplified heuristic)
        for issue in quality_report['issues']:
            issue_lower = issue.lower()
            if 'missing' in issue_lower:
                score += issue_weights['missing']
            elif 'duplicate' in issue_lower:
                score += issue_weights['duplicate']
            elif 'type' in issue_lower or 'dtype' in issue_lower:
                score += issue_weights['type']
            elif 'outlier' in issue_lower:
                score += issue_weights['outlier']
            elif 'variance' in issue_lower or 'identical' in issue_lower:
                score += issue_weights['variance']
            elif 'schema' in issue_lower or 'column' in issue_lower:
                score += issue_weights['schema']
            else:
                score += 10  # Default weight
        
        # Add penalty for warnings (lighter weight)
        score += len(quality_report['warnings']) * 3
        
        return min(100, score)  # Cap at 100
    
    def _determine_severity(self, score: int) -> str:
        """Determine severity level from score"""
        if score >= 75:
            return 'critical'
        elif score >= 50:
            return 'high'
        elif score >= 25:
            return 'medium'
        else:
            return 'low'
    
    def _generate_data_quality_recommendations(self, quality_report: Dict) -> List[str]:
        """Generate intelligent recommendations based on quality issues"""
        
        recommendations = []
        issues = quality_report['issues']
        warnings = quality_report['warnings']
        
        # Missing value recommendations
        if any('missing' in issue.lower() for issue in issues + warnings):
            recommendations.extend([
                "Implement missing value imputation (mean/median/mode)",
                "Consider removing columns with >50% missing values",
                "Investigate systematic missing value patterns",
                "Use advanced imputation methods (KNN, iterative)"
            ])
        
        # Duplicate recommendations
        if any('duplicate' in issue.lower() for issue in issues):
            recommendations.extend([
                "Remove duplicate rows after careful analysis",
                "Investigate root cause of data duplication",
                "Implement data deduplication pipeline"
            ])
        
        # Outlier recommendations
        if any('outlier' in issue.lower() for issue in issues + warnings):
            recommendations.extend([
                "Apply outlier detection and removal/capping",
                "Use robust statistical methods less sensitive to outliers",
                "Investigate if outliers represent valid extreme cases",
                "Consider log transformation for skewed data"
            ])
        
        # Type consistency recommendations
        if any('type' in issue.lower() or 'dtype' in issue.lower() for issue in issues):
            recommendations.extend([
                "Standardize data types across columns",
                "Implement data type validation in pipeline",
                "Handle mixed-type columns explicitly"
            ])
        
        # Variance recommendations
        if any('variance' in issue.lower() or 'identical' in issue.lower() for issue in issues):
            recommendations.extend([
                "Remove zero-variance columns",
                "Investigate data collection process",
                "Check for data processing errors"
            ])
        
        # Default recommendations if no specific issues
        if not recommendations:
            recommendations = [
                "Data quality looks good overall",
                "Consider regular quality monitoring",
                "Implement data validation tests",
                "Document data quality standards"
            ]
        
        return recommendations
    
    def _validate_schema(self, df: pd.DataFrame, expected_schema: dict) -> List[str]:
        """Validate DataFrame against expected schema"""
        
        issues = []
        
        # Check for missing columns
        expected_cols = set(expected_schema.get('columns', {}))
        actual_cols = set(df.columns)
        
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            issues.append(f"Missing expected columns: {list(missing_cols)}")
        
        extra_cols = actual_cols - expected_cols
        if extra_cols:
            issues.append(f"Unexpected columns found: {list(extra_cols)}")
        
        # Check data types
        for col, expected_type in expected_schema.get('columns', {}).items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type != actual_type:
                    issues.append(f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}")
        
        # Check row count constraints
        if 'min_rows' in expected_schema and len(df) < expected_schema['min_rows']:
            issues.append(f"Insufficient rows: {len(df)} < {expected_schema['min_rows']}")
        
        if 'max_rows' in expected_schema and len(df) > expected_schema['max_rows']:
            issues.append(f"Too many rows: {len(df)} > {expected_schema['max_rows']}")
        
        return issues
    
    def _analyze_data_types(self, df: pd.DataFrame) -> List[str]:
        """Analyze data type consistency issues"""
        
        issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                sample_data = df[col].dropna().head(100)  # Sample for performance
                unique_types = set(type(x).__name__ for x in sample_data)
                if len(unique_types) > 1:
                    issues.append(f"Mixed data types in column '{col}': {unique_types}")
                
                # Check if object column contains only numbers
                non_null_data = df[col].dropna()
                if len(non_null_data) > 0:
                    try:
                        pd.to_numeric(non_null_data.head(50))
                        issues.append(f"Column '{col}' contains numeric data but stored as object")
                    except (ValueError, TypeError):
                        pass
        
        return issues
    
    def _analyze_correlations(self, df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        
        analysis = {'warnings': [], 'statistics': {}}
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                
                # Find high correlations
                high_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > threshold:
                            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                            high_correlations.append((col1, col2, corr_value))
                
                if high_correlations:
                    analysis['warnings'].append(f"High correlations found: {high_correlations[:5]}")  # Limit output
                
                analysis['statistics'] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'high_correlations': high_correlations
                }
        
        except Exception as e:
            analysis['warnings'].append(f"Correlation analysis failed: {str(e)}")
        
        return analysis
    
    def _analyze_distributions(self, df: pd.DataFrame, thresholds: dict) -> Dict[str, Any]:
        """Analyze data distributions for normality and patterns"""
        
        analysis = {'warnings': [], 'statistics': {}}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            distribution_stats = {}
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) < 10:  # Need minimum data for distribution analysis
                    continue
                
                # Basic distribution statistics
                stats = {
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                }
                
                # Normality test (if scipy available)
                try:
                    from scipy import stats as scipy_stats
                    if len(col_data) <= 5000:  # Sample for large datasets
                        sample_data = col_data.sample(min(len(col_data), 5000))
                        shapiro_stat, shapiro_p = scipy_stats.shapiro(sample_data)
                        stats['normality_test'] = {
                            'statistic': float(shapiro_stat),
                            'p_value': float(shapiro_p),
                            'is_normal': shapiro_p > 0.05
                        }
                except ImportError:
                    pass
                
                # Distribution shape warnings
                if abs(stats['skewness']) > thresholds['skewness_threshold']:
                    analysis['warnings'].append(f"Non-normal distribution in '{col}' (skewness: {stats['skewness']:.2f})")
                
                distribution_stats[col] = stats
            
            analysis['statistics'] = distribution_stats
            
        except Exception as e:
            analysis['warnings'].append(f"Distribution analysis failed: {str(e)}")
        
        return analysis
    
    def _attempt_self_healing(self, debug_event: DebugEvent) -> bool:
        """Attempt to automatically heal common issues"""
        
        if not self.enable_self_healing:
            return False
        
        healing_key = f"{debug_event.component}_{debug_event.event_type}"
        
        # Prevent excessive healing attempts
        if self.healing_attempts[healing_key] >= 3:
            self.logger.warning(f"Maximum healing attempts reached for {healing_key}")
            return False
        
        self.healing_attempts[healing_key] += 1
        
        try:
            # Memory error healing
            if debug_event.component == 'memory_management':
                import gc
                gc.collect()
                
                # Try to clear GPU cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                self.logger.info("Applied memory healing: garbage collection and cache clearing")
                return True
            
            # File I/O healing
            elif debug_event.component == 'file_io':
                # Create missing directories
                if 'No such file or directory' in debug_event.message:
                    try:
                        # Extract file path from error message
                        import re
                        path_match = re.search(r"'([^']+)'", debug_event.message)
                        if path_match:
                            file_path = Path(path_match.group(1))
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            self.logger.info(f"Created missing directory: {file_path.parent}")
                            return True
                    except Exception:
                        pass
            
            # Network error healing
            elif debug_event.component == 'network_communication':
                # Implement retry with exponential backoff
                time.sleep(2 ** min(self.healing_attempts[healing_key], 4))  # Max 16 seconds
                self.logger.info(f"Applied network healing: retry delay {2 ** min(self.healing_attempts[healing_key], 4)}s")
                return True
        
        except Exception as heal_error:
            self.logger.error(f"Self-healing attempt failed: {heal_error}")
        
        return False
    
    def detect_code_logic_errors(self, code_context: Dict[str, Any], 
                               function_name: str = "unknown") -> List[str]:
        """Advanced code logic error detection"""
        
        logic_errors = []
        
        try:
            # Variable usage analysis
            variables = code_context.get('variables', {})
            function_args = code_context.get('function_args', {})
            return_value = code_context.get('return_value')
            
            # Detect uninitialized variable usage
            uninitialized_vars = []
            for var_name, var_info in variables.items():
                if var_info.get('initialized', True) == False:
                    uninitialized_vars.append(var_name)
            
            if uninitialized_vars:
                logic_errors.append(f"Potentially uninitialized variables: {uninitialized_vars}")
            
            # Detect infinite loops
            loop_counters = code_context.get('loop_counters', {})
            for loop_id, count in loop_counters.items():
                if count > 10000:  # Arbitrary threshold
                    logic_errors.append(f"Potential infinite loop detected in {loop_id} (>{count} iterations)")
            
            # Detect division by zero patterns
            if 'division_operations' in code_context:
                for div_op in code_context['division_operations']:
                    denominator = div_op.get('denominator')
                    if denominator is not None and abs(denominator) < 1e-10:
                        logic_errors.append(f"Division by zero or near-zero value: {denominator}")
            
            # Detect array/list index issues
            if 'array_accesses' in code_context:
                for access in code_context['array_accesses']:
                    array_size = access.get('array_size', 0)
                    index = access.get('index', 0)
                    if index >= array_size or index < -array_size:
                        logic_errors.append(f"Array index out of bounds: index {index} for size {array_size}")
            
            # Detect None/null pointer dereference
            if 'null_checks' in code_context:
                for check in code_context['null_checks']:
                    var_name = check.get('variable')
                    is_null = check.get('is_null', False)
                    if is_null:
                        logic_errors.append(f"Potential null pointer dereference: {var_name}")
            
            # Detect type mismatches
            if 'type_operations' in code_context:
                for op in code_context['type_operations']:
                    expected_type = op.get('expected_type')
                    actual_type = op.get('actual_type')
                    operation = op.get('operation')
                    if expected_type != actual_type:
                        logic_errors.append(f"Type mismatch in {operation}: expected {expected_type}, got {actual_type}")
            
            # Detect unreachable code
            if 'code_coverage' in code_context:
                uncovered_lines = code_context['code_coverage'].get('uncovered_lines', [])
                if uncovered_lines:
                    logic_errors.append(f"Potentially unreachable code at lines: {uncovered_lines}")
            
            # Function return value analysis
            if return_value is not None:
                # Check for inconsistent return types
                if 'previous_returns' in code_context:
                    prev_types = set(type(ret).__name__ for ret in code_context['previous_returns'])
                    current_type = type(return_value).__name__
                    if current_type not in prev_types and len(prev_types) > 0:
                        logic_errors.append(f"Inconsistent return type: {current_type} vs previous {prev_types}")
                
                # Check for missing return statements
                if code_context.get('expects_return', False) and return_value is None:
                    logic_errors.append("Function expected to return value but returns None")
            
            # Resource leak detection
            if 'resource_usage' in code_context:
                resources = code_context['resource_usage']
                unclosed_resources = []
                for resource_type, resource_list in resources.items():
                    for resource in resource_list:
                        if not resource.get('closed', False):
                            unclosed_resources.append(f"{resource_type}: {resource.get('name', 'unnamed')}")
                
                if unclosed_resources:
                    logic_errors.append(f"Potential resource leaks: {unclosed_resources}")
            
            # Algorithm complexity warnings
            complexity_analysis = self._analyze_algorithm_complexity(code_context)
            if complexity_analysis:
                logic_errors.extend(complexity_analysis)
            
            # Pattern-based error detection
            pattern_errors = self._detect_common_patterns(code_context)
            logic_errors.extend(pattern_errors)
        
        except Exception as e:
            logic_errors.append(f"Error during logic analysis: {str(e)}")
        
        # Log logic errors
        if logic_errors:
            debug_event = DebugEvent(
                timestamp=datetime.now(),
                event_type='logic_error',
                severity='high',
                component='code_analysis',
                message=f"Logic errors detected in {function_name}",
                context={
                    'function_name': function_name,
                    'errors': logic_errors,
                    'code_context': code_context
                },
                suggested_fixes=[
                    "Review variable initialization before usage",
                    "Add bounds checking for array accesses",
                    "Implement null/None checks before dereferencing",
                    "Ensure consistent return types across function",
                    "Add loop termination conditions",
                    "Close resources properly (files, connections, etc.)"
                ]
            )
            self.debug_events.append(debug_event)
        
        return logic_errors
    
    def _analyze_algorithm_complexity(self, code_context: Dict[str, Any]) -> List[str]:
        """Analyze algorithm complexity issues"""
        
        complexity_warnings = []
        
        # Nested loop detection
        if 'nested_loops' in code_context:
            max_nesting = code_context['nested_loops'].get('max_depth', 0)
            if max_nesting >= 3:
                complexity_warnings.append(f"High loop nesting detected (depth: {max_nesting}) - consider optimization")
            elif max_nesting == 2:
                loop_sizes = code_context['nested_loops'].get('estimated_sizes', [])
                if loop_sizes and len(loop_sizes) >= 2:
                    estimated_complexity = 1
                    for size in loop_sizes[:2]:
                        if isinstance(size, (int, float)) and size > 0:
                            estimated_complexity *= size
                    
                    if estimated_complexity > 1000000:  # O(n^2) with large n
                        complexity_warnings.append(f"Potential O(n^2) or worse complexity detected (estimated ops: {estimated_complexity})")
        
        # Recursive function analysis
        if 'recursion_info' in code_context:
            recursion_depth = code_context['recursion_info'].get('current_depth', 0)
            max_depth = code_context['recursion_info'].get('max_depth', 1000)
            
            if recursion_depth > max_depth * 0.8:
                complexity_warnings.append(f"Deep recursion detected ({recursion_depth} levels) - risk of stack overflow")
            
            # Check for missing base case
            has_base_case = code_context['recursion_info'].get('has_base_case', True)
            if not has_base_case:
                complexity_warnings.append("Recursive function may be missing base case - infinite recursion risk")
        
        # Large data structure operations
        if 'data_operations' in code_context:
            for op in code_context['data_operations']:
                operation_type = op.get('type')
                data_size = op.get('size', 0)
                
                # Inefficient operations on large datasets
                if data_size > 100000:  # 100k elements
                    if operation_type in ['sort', 'search_linear', 'nested_iteration']:
                        complexity_warnings.append(f"Potentially expensive {operation_type} on large dataset (size: {data_size})")
        
        return complexity_warnings
    
    def _detect_common_patterns(self, code_context: Dict[str, Any]) -> List[str]:
        """Detect common programming error patterns"""
        
        pattern_errors = []
        
        # Assignment vs comparison (common typo)
        if 'comparisons' in code_context:
            for comp in code_context['comparisons']:
                if comp.get('operator') == '=' and comp.get('context') == 'conditional':
                    pattern_errors.append(f"Potential assignment in conditional (did you mean '=='?)")
        
        # String concatenation in loops
        if 'string_operations' in code_context:
            for str_op in code_context['string_operations']:
                if str_op.get('type') == 'concatenation' and str_op.get('in_loop', False):
                    pattern_errors.append("String concatenation in loop - consider using join() or StringBuilder")
        
        # Exception handling anti-patterns
        if 'exception_handling' in code_context:
            eh = code_context['exception_handling']
            
            # Catching too broad exceptions
            if 'broad_exceptions' in eh:
                broad_catches = eh['broad_exceptions']
                if broad_catches:
                    pattern_errors.append(f"Catching overly broad exceptions: {broad_catches}")
            
            # Empty except blocks
            if eh.get('empty_handlers', 0) > 0:
                pattern_errors.append("Empty exception handler blocks found - consider logging or handling")
            
            # Exception swallowing
            if eh.get('swallowed_exceptions', 0) > 0:
                pattern_errors.append("Exceptions being silently ignored - consider proper handling")
        
        # Magic number usage
        if 'magic_numbers' in code_context:
            magic_nums = code_context['magic_numbers']
            if len(magic_nums) > 3:
                pattern_errors.append(f"Multiple magic numbers detected: {magic_nums} - consider using constants")
        
        # Mutable default arguments (Python-specific)
        if 'default_args' in code_context:
            for arg in code_context['default_args']:
                if arg.get('is_mutable', False):
                    pattern_errors.append(f"Mutable default argument '{arg.get('name')}' detected - potential bug source")
        
        # Floating point equality comparisons
        if 'float_comparisons' in code_context:
            direct_float_comparisons = code_context['float_comparisons'].get('direct_equality', 0)
            if direct_float_comparisons > 0:
                pattern_errors.append("Direct floating point equality comparison - consider using tolerance-based comparison")
        
        return pattern_errors
    
    def generate_comprehensive_error_report(self, report_name: str = None, 
                                           include_historical: bool = True,
                                           include_performance: bool = True,
                                           include_predictions: bool = True) -> Dict[str, Any]:
        """Generate comprehensive error and debugging report"""
        
        if report_name is None:
            report_name = f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Collect all debugging data
        report = {
            'report_metadata': {
                'report_name': report_name,
                'generation_timestamp': datetime.now().isoformat(),
                'debugger_version': '1.0.0',
                'total_events': len(self.debug_events),
                'reporting_period': {
                    'start': min(event.timestamp for event in self.debug_events).isoformat() if self.debug_events else None,
                    'end': max(event.timestamp for event in self.debug_events).isoformat() if self.debug_events else None
                }
            },
            'executive_summary': self._generate_executive_summary(),
            'error_analysis': self._analyze_all_errors(),
            'performance_analysis': self._analyze_performance_trends() if include_performance else None,
            'anomaly_analysis': self._analyze_anomaly_patterns(),
            'recommendations': self._generate_strategic_recommendations(),
            'detailed_events': [self._serialize_debug_event(event) for event in self.debug_events[-50:]]  # Last 50 events
        }
        
        if include_historical:
            report['historical_trends'] = self._analyze_historical_trends()
        
        if include_predictions:
            report['predictive_analysis'] = self._generate_predictive_analysis()
        
        # Add system health assessment
        report['system_health'] = self._assess_system_health()
        
        # Generate actionable insights
        report['actionable_insights'] = self._generate_actionable_insights(report)
        
        # Save comprehensive report
        self._save_comprehensive_report(report, report_name)
        
        # Generate summary visualizations
        if include_performance:
            self._generate_report_visualizations(report, report_name)
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate high-level executive summary"""
        
        if not self.debug_events:
            return {'status': 'no_events', 'message': 'No debugging events recorded'}
        
        # Categorize events by severity and type
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        type_counts = {}
        component_counts = {}
        
        for event in self.debug_events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
            component_counts[event.component] = component_counts.get(event.component, 0) + 1
        
        # Calculate health score
        health_score = self._calculate_health_score(severity_counts)
        
        # Identify top issues
        top_error_patterns = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'overall_health_score': health_score,
            'health_status': self._get_health_status(health_score),
            'total_issues': len(self.debug_events),
            'severity_breakdown': severity_counts,
            'event_type_breakdown': type_counts,
            'most_affected_components': sorted(component_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'top_error_patterns': top_error_patterns,
            'healing_success_rate': self._calculate_healing_success_rate(),
            'key_findings': self._generate_key_findings()
        }
    
    def _analyze_all_errors(self) -> Dict[str, Any]:
        """Comprehensive error analysis"""
        
        error_events = [event for event in self.debug_events if event.event_type == 'error']
        
        if not error_events:
            return {'status': 'no_errors', 'message': 'No errors recorded'}
        
        # Temporal analysis
        timestamps = [event.timestamp for event in error_events]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        
        # Error clustering
        error_clusters = self._cluster_similar_errors(error_events)
        
        # Root cause analysis
        root_causes = self._identify_root_causes(error_events)
        
        # Error correlation analysis
        correlations = self._analyze_error_correlations(error_events)
        
        return {
            'total_errors': len(error_events),
            'temporal_analysis': {
                'first_error': timestamps[0].isoformat() if timestamps else None,
                'last_error': timestamps[-1].isoformat() if timestamps else None,
                'average_time_between_errors': np.mean(time_diffs) if time_diffs else 0,
                'error_frequency_trend': self._calculate_frequency_trend(timestamps)
            },
            'error_clusters': error_clusters,
            'root_cause_analysis': root_causes,
            'error_correlations': correlations,
            'resolution_analysis': self._analyze_error_resolutions(error_events)
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from profiling data"""
        
        if not self.profiling_data:
            return {'status': 'no_performance_data'}
        
        performance_summary = {}
        
        for component, data_points in self.profiling_data.items():
            if not data_points:
                continue
                
            execution_times = [dp.get('execution_time', 0) for dp in data_points]
            memory_usage = [dp.get('memory_usage_mb', 0) for dp in data_points if dp.get('memory_usage_mb')]
            
            # Trend analysis
            time_trend = np.polyfit(range(len(execution_times)), execution_times, 1)[0] if len(execution_times) > 1 else 0
            
            performance_summary[component] = {
                'total_executions': len(execution_times),
                'avg_execution_time': np.mean(execution_times),
                'max_execution_time': np.max(execution_times),
                'performance_trend': 'improving' if time_trend < -0.001 else 'degrading' if time_trend > 0.001 else 'stable',
                'trend_slope': float(time_trend),
                'memory_analysis': {
                    'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                    'max_memory_mb': np.max(memory_usage) if memory_usage else 0,
                    'memory_trend': self._analyze_memory_trend(memory_usage)
                } if memory_usage else None,
                'bottleneck_score': self._calculate_bottleneck_score(execution_times)
            }
        
        return {
            'component_performance': performance_summary,
            'overall_performance_health': self._assess_overall_performance(performance_summary),
            'performance_recommendations': self._generate_performance_recommendations(performance_summary)
        }
    
    def _analyze_anomaly_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in detected anomalies"""
        
        anomaly_events = [event for event in self.debug_events if event.event_type == 'anomaly']
        
        if not anomaly_events:
            return {'status': 'no_anomalies'}
        
        # Group by anomaly type
        anomaly_types = {}
        for event in anomaly_events:
            anomaly_type = event.context.get('anomaly_type', 'unknown')
            if anomaly_type not in anomaly_types:
                anomaly_types[anomaly_type] = []
            anomaly_types[anomaly_type].append(event)
        
        # Analyze each type
        type_analysis = {}
        for anom_type, events in anomaly_types.items():
            scores = [event.context.get('anomaly_score', 0) for event in events]
            confidences = [event.context.get('confidence', 0) for event in events]
            
            type_analysis[anom_type] = {
                'count': len(events),
                'avg_score': np.mean(scores) if scores else 0,
                'max_score': np.max(scores) if scores else 0,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'trend': self._analyze_anomaly_trend(events),
                'severity_distribution': {sev: sum(1 for e in events if e.severity == sev) for sev in ['low', 'medium', 'high', 'critical']}
            }
        
        return {
            'total_anomalies': len(anomaly_events),
            'anomaly_types': type_analysis,
            'anomaly_frequency': self._calculate_anomaly_frequency(anomaly_events),
            'anomaly_predictions': self._predict_future_anomalies(anomaly_events)
        }
    
    def _generate_strategic_recommendations(self) -> List[Dict[str, Any]]:
        """Generate strategic recommendations based on all analysis"""
        
        recommendations = []
        
        # Based on error patterns
        top_errors = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        for error_pattern, count in top_errors:
            recommendations.append({
                'type': 'error_prevention',
                'priority': 'high' if count > 5 else 'medium',
                'title': f"Address recurring {error_pattern} errors",
                'description': f"This error pattern has occurred {count} times",
                'action_items': self._get_error_specific_actions(error_pattern),
                'expected_impact': 'high'
            })
        
        # Based on performance
        if self.profiling_data:
            slow_components = [(comp, np.mean([dp.get('execution_time', 0) for dp in data])) 
                             for comp, data in self.profiling_data.items() if data]
            slow_components.sort(key=lambda x: x[1], reverse=True)
            
            for comp, avg_time in slow_components[:2]:
                if avg_time > 5.0:  # Slow components
                    recommendations.append({
                        'type': 'performance_optimization',
                        'priority': 'medium',
                        'title': f"Optimize performance of {comp}",
                        'description': f"Average execution time: {avg_time:.2f}s",
                        'action_items': [
                            "Profile component for bottlenecks",
                            "Consider algorithmic optimizations",
                            "Implement caching if applicable",
                            "Parallelize operations where possible"
                        ],
                        'expected_impact': 'medium'
                    })
        
        # Based on healing attempts
        failed_healing = {k: v for k, v in self.healing_attempts.items() if v >= 3}
        if failed_healing:
            recommendations.append({
                'type': 'system_reliability',
                'priority': 'high',
                'title': "Improve auto-healing mechanisms",
                'description': f"{len(failed_healing)} components reached max healing attempts",
                'action_items': [
                    "Review and enhance healing strategies",
                    "Add more specific error handling",
                    "Implement manual intervention triggers",
                    "Add monitoring for healing effectiveness"
                ],
                'expected_impact': 'high'
            })
        
        # Data quality recommendations
        data_quality_events = [e for e in self.debug_events if e.event_type == 'data_quality']
        if len(data_quality_events) > 5:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'medium',
                'title': "Implement comprehensive data validation",
                'description': f"{len(data_quality_events)} data quality issues detected",
                'action_items': [
                    "Create data validation pipeline",
                    "Implement automated data cleaning",
                    "Add data quality monitoring",
                    "Establish data quality standards"
                ],
                'expected_impact': 'medium'
            })
        
        return sorted(recommendations, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
    
    def _generate_actionable_insights(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific actionable insights from the report"""
        
        insights = []
        
        # From executive summary
        exec_summary = report.get('executive_summary', {})
        health_score = exec_summary.get('overall_health_score', 50)
        
        if health_score < 30:
            insights.append({
                'category': 'urgent_action',
                'insight': "System health is critical - immediate attention required",
                'evidence': f"Health score: {health_score}/100",
                'action': "Implement emergency debugging session and fix critical issues",
                'timeline': 'immediate'
            })
        
        # From error analysis
        error_analysis = report.get('error_analysis', {})
        if isinstance(error_analysis, dict) and 'total_errors' in error_analysis:
            error_frequency = error_analysis.get('temporal_analysis', {}).get('average_time_between_errors', 0)
            if error_frequency < 300:  # Less than 5 minutes
                insights.append({
                    'category': 'stability',
                    'insight': "High error frequency indicates system instability",
                    'evidence': f"Average time between errors: {error_frequency:.1f}s",
                    'action': "Implement circuit breaker patterns and increase error handling",
                    'timeline': 'short_term'
                })
        
        # From performance analysis
        perf_analysis = report.get('performance_analysis')
        if perf_analysis and isinstance(perf_analysis, dict):
            overall_health = perf_analysis.get('overall_performance_health', 'unknown')
            if overall_health == 'poor':
                insights.append({
                    'category': 'performance',
                    'insight': "Performance degradation detected across multiple components",
                    'evidence': "Overall performance health rated as 'poor'",
                    'action': "Conduct comprehensive performance audit and optimization",
                    'timeline': 'medium_term'
                })
        
        # From anomaly analysis
        anomaly_analysis = report.get('anomaly_analysis', {})
        if isinstance(anomaly_analysis, dict) and anomaly_analysis.get('total_anomalies', 0) > 10:
            insights.append({
                'category': 'monitoring',
                'insight': "High anomaly rate suggests need for better baseline establishment",
                'evidence': f"Total anomalies: {anomaly_analysis['total_anomalies']}",
                'action': "Refine anomaly detection thresholds and establish better baselines",
                'timeline': 'short_term'
            })
        
        return insights
    
    def _save_comprehensive_report(self, report: Dict[str, Any], report_name: str):
        """Save comprehensive report to multiple formats"""
        
        # JSON format (detailed)
        json_file = self.output_dir / f"{report_name}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # HTML format (readable)
        html_file = self.output_dir / f"{report_name}.html"
        self._generate_html_report(report, html_file)
        
        # CSV format (for data analysis)
        csv_file = self.output_dir / f"{report_name}_events.csv"
        self._generate_csv_report(report, csv_file)
        
        self.logger.info(f"Comprehensive report saved: {json_file}, {html_file}, {csv_file}")
    
    def _generate_immediate_error_report(self, debug_event: DebugEvent):
        """Generate immediate error report for critical issues"""
        
        report = {
            'timestamp': debug_event.timestamp.isoformat(),
            'error_summary': {
                'type': debug_event.event_type,
                'severity': debug_event.severity,
                'component': debug_event.component,
                'message': debug_event.message
            },
            'context': debug_event.context,
            'suggested_fixes': debug_event.suggested_fixes,
            'auto_healing_applied': debug_event.auto_healing_applied,
            'stack_trace': debug_event.stack_trace,
            'immediate_actions': self._generate_immediate_actions(debug_event),
            'related_events': self._find_related_events(debug_event)
        }
        
        # Save immediate report
        report_file = self.output_dir / f"critical_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.critical(f"Critical error report generated: {report_file}")
        
        return report

# Decorator for automatic debugging
def debug_monitor(component: str = None, profile: bool = True):
    """Decorator to automatically monitor function execution for debugging"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create debugger instance
            if not hasattr(wrapper, '_debugger'):
                wrapper._debugger = IntelligentDebugger()
            
            debugger = wrapper._debugger
            component_name = component or f"{func.__module__}.{func.__name__}"
            
            # Start profiling
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Check for performance issues
                if profile:
                    bottlenecks = debugger.detect_performance_bottlenecks(execution_time, component_name)
                
                # If result contains metrics, check for anomalies
                if isinstance(result, dict) and any(key in result for key in ['accuracy', 'loss', 'score']):
                    anomalies = debugger.detect_statistical_anomalies(result, component_name)
                
                return result
                
            except Exception as e:
                # Exception is automatically handled by global handler
                raise
        
        return wrapper
    return decorator

# Context manager for debugging sessions
class DebugSession:
    """Context manager for debugging sessions"""
    
    def __init__(self, session_name: str, debugger: IntelligentDebugger = None):
        self.session_name = session_name
        self.debugger = debugger or IntelligentDebugger()
        self.start_time = None
        self.session_data = {}
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.debugger.logger.info(f"Starting debug session: {self.session_name}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        duration = datetime.now() - self.start_time
        
        if exc_type is not None:
            self.debugger.logger.error(f"Debug session '{self.session_name}' ended with exception after {duration}")
        else:
            self.debugger.logger.info(f"Debug session '{self.session_name}' completed successfully in {duration}")
        
        # Generate session report
        self.generate_session_report()
    
    def log_metric(self, name: str, value: float):
        """Log a metric for this debugging session"""
        self.session_data[name] = value
        
        # Check for anomalies
        anomalies = self.debugger.detect_statistical_anomalies({name: value}, self.session_name)
    
    def check_data_quality(self, data, name: str = "data"):
        """Check data quality during session"""
        issues = self.debugger.detect_data_quality_issues(data, f"{self.session_name}_{name}")
        return issues
    
    def generate_session_report(self):
        """Generate a comprehensive session report"""
        session_events = [event for event in self.debugger.debug_events 
                         if event.timestamp >= self.start_time]
        
        report = {
            'session_name': self.session_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'session_data': self.session_data,
            'events_count': len(session_events),
            'events_by_type': {},
            'events_by_severity': {}
        }
        
        # Categorize events
        for event in session_events:
            report['events_by_type'][event.event_type] = report['events_by_type'].get(event.event_type, 0) + 1
            report['events_by_severity'][event.severity] = report['events_by_severity'].get(event.severity, 0) + 1
        
        # Save session report
        report_file = self.debugger.output_dir / f"session_{self.session_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

if __name__ == "__main__":
    # Example usage
    debugger = IntelligentDebugger()
    
    # Test anomaly detection
    test_metrics = {'accuracy': 0.95, 'loss': 0.02}
    anomalies = debugger.detect_statistical_anomalies(test_metrics)
    
    print(f"Detected {len(anomalies)} anomalies")
    
    # Test with debug session
    with DebugSession("test_session", debugger) as session:
        session.log_metric("test_accuracy", 0.85)
        
        # Simulate some data quality issues
        import pandas as pd
        test_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 1000],  # Has missing and outlier
            'feature2': [1, 1, 1, 1, 1]  # No variance
        })
        
        issues = session.check_data_quality(test_data, "test_dataset")
        print(f"Found {len(issues)} data quality issues")