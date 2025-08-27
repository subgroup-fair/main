"""
Step-by-Step Execution Tracer for Research Debugging
Provides detailed execution tracing with variable state inspection and performance profiling
"""

import sys
import inspect
import time
import traceback
import threading
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import logging
import copy
import gc
import psutil
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.debugging.intelligent_debugger import IntelligentDebugger

@dataclass 
class ExecutionStep:
    """Single execution step record"""
    step_id: str
    timestamp: datetime
    function_name: str
    module_name: str
    line_number: int
    code_line: str
    event_type: str  # 'call', 'line', 'return', 'exception'
    local_variables: Dict[str, Any]
    global_variables: Dict[str, Any]
    stack_depth: int
    memory_usage: float
    cpu_percent: float
    execution_time: float
    parent_step_id: Optional[str] = None
    error_info: Optional[Dict[str, Any]] = None

@dataclass
class FunctionTrace:
    """Complete trace of a function execution"""
    function_name: str
    module_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: List[ExecutionStep] = None
    total_duration: float = 0.0
    max_memory: float = 0.0
    avg_cpu: float = 0.0
    return_value: Any = None
    exception_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []

class ExecutionTracer:
    """
    Advanced step-by-step execution tracer with variable state inspection
    """
    
    def __init__(self,
                 max_steps: int = 10000,
                 trace_variables: bool = True,
                 trace_memory: bool = True,
                 trace_performance: bool = True,
                 filter_functions: List[str] = None,
                 output_dir: str = "execution_traces"):
        
        self.max_steps = max_steps
        self.trace_variables = trace_variables
        self.trace_memory = trace_memory
        self.trace_performance = trace_performance
        self.filter_functions = filter_functions or []
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # State tracking
        self.execution_steps: List[ExecutionStep] = []
        self.function_traces: Dict[str, FunctionTrace] = {}
        self.call_stack: List[str] = []
        self.step_counter = 0
        self.start_time = None
        self.is_tracing = False
        
        # Performance monitoring
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Thread safety
        self.trace_lock = threading.Lock()
        
        self.logger = logging.getLogger("execution_tracer")
        
        self.logger.info("ExecutionTracer initialized")
    
    def start_tracing(self, target_function: Optional[Callable] = None):
        """Start execution tracing"""
        
        with self.trace_lock:
            if self.is_tracing:
                self.logger.warning("Tracing already active")
                return
            
            self.is_tracing = True
            self.start_time = datetime.now()
            self.execution_steps.clear()
            self.function_traces.clear()
            self.call_stack.clear()
            self.step_counter = 0
            
            # Install trace function
            if target_function:
                # Trace specific function
                self._trace_function(target_function)
            else:
                # Trace everything
                sys.settrace(self._trace_calls)
                threading.settrace(self._trace_calls)  # Also trace threads
            
            self.logger.info("Execution tracing started")
    
    def stop_tracing(self) -> Dict[str, Any]:
        """Stop execution tracing and return summary"""
        
        with self.trace_lock:
            if not self.is_tracing:
                return {}
            
            self.is_tracing = False
            
            # Remove trace function
            sys.settrace(None)
            threading.settrace(None)
            
            # Generate summary
            summary = self._generate_trace_summary()
            
            # Save trace data
            self._save_trace_data(summary)
            
            self.logger.info(f"Execution tracing stopped. Captured {len(self.execution_steps)} steps")
            
            return summary
    
    def _trace_calls(self, frame, event, arg):
        """Main trace function called by Python interpreter"""
        
        if not self.is_tracing:
            return None
        
        try:
            # Filter out unwanted functions/modules
            if self._should_skip_frame(frame):
                return None
            
            # Create execution step
            step = self._create_execution_step(frame, event, arg)
            
            if step:
                with self.trace_lock:
                    self.execution_steps.append(step)
                    
                    # Maintain maximum steps limit
                    if len(self.execution_steps) > self.max_steps:
                        self.execution_steps = self.execution_steps[-self.max_steps:]
                    
                    # Update function traces
                    self._update_function_trace(step, event, arg)
                    
                    self.step_counter += 1
        
        except Exception as e:
            self.logger.error(f"Error in trace function: {e}")
        
        return self._trace_calls  # Continue tracing
    
    def _should_skip_frame(self, frame) -> bool:
        """Determine if frame should be skipped"""
        
        filename = frame.f_code.co_filename
        function_name = frame.f_code.co_name
        
        # Skip system/library files
        if any(skip_path in filename for skip_path in [
            'site-packages', '/usr/lib/', 'importlib', 'pkgutil',
            '<frozen', 'threading.py', 'queue.py', 'logging'
        ]):
            return True
        
        # Skip if function in filter list
        if self.filter_functions and function_name not in self.filter_functions:
            return True
        
        # Skip tracer internal functions
        if 'tracer' in filename.lower() or 'debugging' in filename.lower():
            return True
        
        return False
    
    def _create_execution_step(self, frame, event: str, arg) -> Optional[ExecutionStep]:
        """Create an execution step from frame information"""
        
        try:
            # Get basic frame information
            filename = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            line_number = frame.f_lineno
            module_name = self._get_module_name(filename)
            
            # Get code line
            code_line = self._get_code_line(filename, line_number)
            
            # Generate step ID
            step_id = f"{self.step_counter:06d}_{function_name}_{line_number}"
            
            # Get parent step ID
            parent_step_id = self.call_stack[-1] if self.call_stack else None
            
            # Get variable states
            local_vars = {}
            global_vars = {}
            
            if self.trace_variables:
                local_vars = self._capture_variables(frame.f_locals)
                global_vars = self._capture_variables(frame.f_globals, max_items=10)
            
            # Get performance metrics
            memory_usage = 0.0
            cpu_percent = 0.0
            
            if self.trace_performance:
                try:
                    memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
                    cpu_percent = self.process.cpu_percent()
                except:
                    pass
            
            # Handle exceptions
            error_info = None
            if event == 'exception' and arg:
                error_info = {
                    'exception_type': type(arg[1]).__name__,
                    'exception_message': str(arg[1]),
                    'exception_traceback': traceback.format_exception(*arg)
                }
            
            # Calculate execution time (rough estimate based on step interval)
            execution_time = 0.001  # Default 1ms per step
            
            step = ExecutionStep(
                step_id=step_id,
                timestamp=datetime.now(),
                function_name=function_name,
                module_name=module_name,
                line_number=line_number,
                code_line=code_line,
                event_type=event,
                local_variables=local_vars,
                global_variables=global_vars,
                stack_depth=len(self.call_stack),
                memory_usage=memory_usage,
                cpu_percent=cpu_percent,
                execution_time=execution_time,
                parent_step_id=parent_step_id,
                error_info=error_info
            )
            
            return step
        
        except Exception as e:
            self.logger.debug(f"Error creating execution step: {e}")
            return None
    
    def _get_module_name(self, filename: str) -> str:
        """Extract module name from filename"""
        
        try:
            path = Path(filename)
            if 'site-packages' in str(path):
                parts = str(path).split('site-packages')[-1].split('/')
                return '.'.join(parts[1:]).replace('.py', '')
            else:
                return path.stem
        except:
            return 'unknown'
    
    def _get_code_line(self, filename: str, line_number: int) -> str:
        """Get the actual code line from file"""
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if 0 <= line_number - 1 < len(lines):
                    return lines[line_number - 1].strip()
        except:
            pass
        
        return "# Could not read source line"
    
    def _capture_variables(self, var_dict: Dict[str, Any], max_items: int = 50) -> Dict[str, Any]:
        """Capture and serialize variables safely"""
        
        captured = {}
        count = 0
        
        for name, value in var_dict.items():
            if count >= max_items:
                captured['...'] = f"({len(var_dict) - max_items} more variables)"
                break
            
            # Skip special variables and large objects
            if name.startswith('__') and name.endswith('__'):
                continue
            
            try:
                # Serialize value safely
                serialized_value = self._safe_serialize(value)
                captured[name] = serialized_value
                count += 1
            except Exception as e:
                captured[name] = f"<Error serializing: {str(e)[:100]}>"
        
        return captured
    
    def _safe_serialize(self, value: Any, max_depth: int = 3) -> Any:
        """Advanced variable serialization with state inspection"""
        
        if max_depth <= 0:
            return f"<Max depth reached: {type(value).__name__}>"
        
        try:
            # Handle common types with enhanced inspection
            if value is None or isinstance(value, (bool, int, float, str)):
                result = {'value': value, 'type': type(value).__name__}
                
                # Add metadata for strings
                if isinstance(value, str):
                    if len(value) > 200:
                        result['value'] = value[:200] + f"... (truncated)"
                        result['full_length'] = len(value)
                    result['encoding_issues'] = self._check_string_encoding(value)
                
                # Add metadata for numbers
                elif isinstance(value, (int, float)):
                    result['is_finite'] = True
                    if isinstance(value, float):
                        result['is_finite'] = np.isfinite(value)
                        result['is_nan'] = np.isnan(value)
                        result['is_inf'] = np.isinf(value)
                
                return result
            
            elif isinstance(value, (list, tuple)):
                result = {
                    'type': type(value).__name__,
                    'length': len(value),
                    'memory_size_bytes': sys.getsizeof(value)
                }
                
                if len(value) > 0:
                    # Sample first few elements
                    sample_size = min(5, len(value))
                    result['sample'] = [self._safe_serialize(v, max_depth-1) for v in value[:sample_size]]
                    
                    # Type analysis
                    element_types = [type(v).__name__ for v in value[:20]]  # Check first 20
                    result['element_types'] = list(set(element_types))
                    result['is_homogeneous'] = len(set(element_types)) == 1
                    
                    # Statistical analysis for numeric data
                    if all(isinstance(v, (int, float)) for v in value[:10]):
                        numeric_stats = self._analyze_numeric_array(value)
                        result['numeric_stats'] = numeric_stats
                
                if len(value) > 10:
                    result['truncated'] = True
                    result['hidden_items'] = len(value) - 5
                
                return result
            
            elif isinstance(value, dict):
                result = {
                    'type': 'dict',
                    'length': len(value),
                    'memory_size_bytes': sys.getsizeof(value)
                }
                
                if value:
                    # Sample first few items
                    items = list(value.items())[:5]
                    result['sample'] = {k: self._safe_serialize(v, max_depth-1) for k, v in items}
                    
                    # Key analysis
                    key_types = [type(k).__name__ for k in value.keys()]
                    result['key_types'] = list(set(key_types))
                    
                    # Value analysis
                    value_types = [type(v).__name__ for v in list(value.values())[:20]]
                    result['value_types'] = list(set(value_types))
                
                if len(value) > 5:
                    result['truncated'] = True
                    result['hidden_items'] = len(value) - 5
                
                return result
            
            elif isinstance(value, np.ndarray):
                result = {
                    'type': 'numpy.ndarray',
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'memory_size_bytes': value.nbytes,
                    'is_c_contiguous': value.flags.c_contiguous,
                    'is_fortran_contiguous': value.flags.f_contiguous
                }
                
                if value.size > 0:
                    # Statistical analysis
                    if value.dtype.kind in 'fc':  # Numeric types
                        flat_data = value.flatten()
                        valid_mask = np.isfinite(flat_data) if value.dtype.kind == 'f' else np.ones(len(flat_data), dtype=bool)
                        valid_data = flat_data[valid_mask]
                        
                        if len(valid_data) > 0:
                            result['stats'] = {
                                'min': float(np.min(valid_data)),
                                'max': float(np.max(valid_data)),
                                'mean': float(np.mean(valid_data)),
                                'std': float(np.std(valid_data)),
                                'nan_count': int(np.isnan(flat_data).sum()) if value.dtype.kind == 'f' else 0,
                                'inf_count': int(np.isinf(flat_data).sum()) if value.dtype.kind == 'f' else 0
                            }
                    
                    # Sample data
                    sample_size = min(10, value.size)
                    result['sample'] = value.flat[:sample_size].tolist()
                
                return result
            
            elif hasattr(value, 'shape') and hasattr(value, 'dtype'):  # Pandas/other array-like
                result = {
                    'type': type(value).__name__,
                    'module': getattr(type(value), '__module__', 'unknown'),
                    'shape': getattr(value, 'shape', 'unknown'),
                    'dtype': str(getattr(value, 'dtype', 'unknown'))
                }
                
                try:
                    result['memory_usage_bytes'] = value.memory_usage(deep=True).sum() if hasattr(value, 'memory_usage') else sys.getsizeof(value)
                except:
                    result['memory_usage_bytes'] = sys.getsizeof(value)
                
                # For DataFrames
                if hasattr(value, 'columns') and hasattr(value, 'dtypes'):
                    result['columns'] = list(value.columns)[:10]  # First 10 columns
                    result['dtypes'] = {col: str(dtype) for col, dtype in value.dtypes.head(10).items()}
                    result['null_counts'] = {col: int(count) for col, count in value.isnull().sum().head(10).items()}
                
                return result
            
            elif hasattr(value, '__dict__'):
                # Custom objects with enhanced inspection
                result = {
                    'type': type(value).__name__,
                    'module': getattr(type(value), '__module__', 'unknown'),
                    'memory_size_bytes': sys.getsizeof(value)
                }
                
                # Inspect attributes
                attributes = {}
                attr_count = 0
                for k, v in value.__dict__.items():
                    if attr_count >= 5:  # Limit attributes
                        break
                    if not k.startswith('_'):
                        attributes[k] = self._safe_serialize(v, max_depth-1)
                        attr_count += 1
                
                result['attributes'] = attributes
                
                # Check for common patterns
                if hasattr(value, '__len__'):
                    result['length'] = len(value)
                
                if hasattr(value, '__iter__'):
                    result['is_iterable'] = True
                
                # Method analysis
                methods = [m for m in dir(value) if not m.startswith('_') and callable(getattr(value, m, None))]
                result['public_methods'] = methods[:10]  # First 10 methods
                
                return result
            
            else:
                # Fallback with enhanced analysis
                result = {
                    'type': type(value).__name__,
                    'module': getattr(type(value), '__module__', 'unknown'),
                    'memory_size_bytes': sys.getsizeof(value)
                }
                
                # Try to get string representation
                try:
                    str_repr = str(value)
                    result['str_repr'] = str_repr[:100] + ('...' if len(str_repr) > 100 else '')
                except:
                    result['str_repr'] = '<Cannot convert to string>'
                
                # Check common attributes
                if hasattr(value, '__len__'):
                    result['length'] = len(value)
                
                if hasattr(value, '__iter__'):
                    result['is_iterable'] = True
                
                return result
        
        except Exception as e:
            return {
                'type': 'SerializationError',
                'error': str(e)[:100],
                'original_type': type(value).__name__
            }
    
    def _check_string_encoding(self, s: str) -> Dict[str, Any]:
        """Check string for encoding issues"""
        
        issues = {}
        
        try:
            # Check for common encoding issues
            s.encode('utf-8')
            issues['utf8_compatible'] = True
        except UnicodeEncodeError:
            issues['utf8_compatible'] = False
            issues['encoding_errors'] = True
        
        # Check for unusual characters
        if any(ord(c) > 127 for c in s):
            issues['contains_non_ascii'] = True
            non_ascii_count = sum(1 for c in s if ord(c) > 127)
            issues['non_ascii_ratio'] = non_ascii_count / len(s)
        
        return issues
    
    def _analyze_numeric_array(self, arr: list) -> Dict[str, Any]:
        """Analyze numeric array/list for statistical properties"""
        
        try:
            numeric_values = [v for v in arr if isinstance(v, (int, float))]
            if not numeric_values:
                return {'error': 'No numeric values found'}
            
            np_arr = np.array(numeric_values)
            
            return {
                'count': len(numeric_values),
                'min': float(np.min(np_arr)),
                'max': float(np.max(np_arr)),
                'mean': float(np.mean(np_arr)),
                'std': float(np.std(np_arr)),
                'median': float(np.median(np_arr)),
                'has_negatives': bool(np.any(np_arr < 0)),
                'has_zeros': bool(np.any(np_arr == 0)),
                'range': float(np.max(np_arr) - np.min(np_arr))
            }
        
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def inspect_variable_changes(self, var_name: str, 
                               start_step: int = 0, 
                               end_step: int = None) -> Dict[str, Any]:
        """Advanced inspection of how a variable changes over execution"""
        
        if end_step is None:
            end_step = len(self.execution_steps)
        
        changes = []
        previous_value = None
        
        for i, step in enumerate(self.execution_steps[start_step:end_step], start_step):
            # Check both local and global variables
            current_value = None
            var_location = None
            
            if var_name in step.local_variables:
                current_value = step.local_variables[var_name]
                var_location = 'local'
            elif var_name in step.global_variables:
                current_value = step.global_variables[var_name]
                var_location = 'global'
            
            if current_value is not None:
                # Detect changes
                if previous_value is None or not self._values_equal(previous_value, current_value):
                    change_info = {
                        'step_id': step.step_id,
                        'step_index': i,
                        'timestamp': step.timestamp.isoformat(),
                        'function': step.function_name,
                        'line': step.line_number,
                        'code': step.code_line,
                        'location': var_location,
                        'previous_value': previous_value,
                        'new_value': current_value,
                        'change_type': self._classify_change(previous_value, current_value)
                    }
                    changes.append(change_info)
                    previous_value = current_value
        
        # Analysis summary
        analysis = {
            'variable_name': var_name,
            'total_changes': len(changes),
            'changes': changes,
            'change_pattern_analysis': self._analyze_change_patterns(changes),
            'value_type_changes': self._analyze_type_changes(changes)
        }
        
        return analysis
    
    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Compare two values for equality, handling complex types"""
        
        try:
            if type(val1) != type(val2):
                return False
            
            if isinstance(val1, dict) and isinstance(val2, dict):
                return val1.get('value') == val2.get('value')
            
            if isinstance(val1, (list, tuple)):
                if len(val1) != len(val2):
                    return False
                return all(self._values_equal(a, b) for a, b in zip(val1, val2))
            
            return val1 == val2
        
        except Exception:
            return False
    
    def _classify_change(self, old_val: Any, new_val: Any) -> str:
        """Classify the type of change between two values"""
        
        if old_val is None:
            return 'initialization'
        
        if new_val is None:
            return 'set_to_none'
        
        old_type = type(old_val.get('value', old_val) if isinstance(old_val, dict) else old_val)
        new_type = type(new_val.get('value', new_val) if isinstance(new_val, dict) else new_val)
        
        if old_type != new_type:
            return f'type_change_{old_type.__name__}_to_{new_type.__name__}'
        
        # For dictionaries (our serialized format)
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            old_value = old_val.get('value')
            new_value = new_val.get('value')
            
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value > old_value:
                    return 'numeric_increase'
                elif new_value < old_value:
                    return 'numeric_decrease'
                else:
                    return 'numeric_same'
            
            elif isinstance(old_value, str) and isinstance(new_value, str):
                if len(new_value) > len(old_value):
                    return 'string_length_increase'
                elif len(new_value) < len(old_value):
                    return 'string_length_decrease'
                else:
                    return 'string_content_change'
            
            elif isinstance(old_val, dict) and 'length' in old_val and 'length' in new_val:
                old_len = old_val['length']
                new_len = new_val['length']
                if new_len > old_len:
                    return 'collection_size_increase'
                elif new_len < old_len:
                    return 'collection_size_decrease'
                else:
                    return 'collection_content_change'
        
        return 'value_change'
    
    def _analyze_change_patterns(self, changes: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in variable changes"""
        
        if not changes:
            return {}
        
        change_types = [change['change_type'] for change in changes]
        type_counts = {}
        for change_type in change_types:
            type_counts[change_type] = type_counts.get(change_type, 0) + 1
        
        # Time analysis
        if len(changes) >= 2:
            timestamps = [datetime.fromisoformat(change['timestamp']) for change in changes]
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            avg_time_between_changes = sum(time_diffs) / len(time_diffs)
        else:
            avg_time_between_changes = 0
        
        return {
            'change_type_distribution': type_counts,
            'most_common_change': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
            'average_time_between_changes_seconds': avg_time_between_changes,
            'functions_with_changes': list(set(change['function'] for change in changes))
        }
    
    def _analyze_type_changes(self, changes: List[Dict]) -> Dict[str, Any]:
        """Analyze type changes in variable evolution"""
        
        type_sequence = []
        for change in changes:
            new_val = change['new_value']
            if isinstance(new_val, dict) and 'type' in new_val:
                type_sequence.append(new_val['type'])
            else:
                type_sequence.append(type(new_val).__name__)
        
        type_changes = []
        for i in range(1, len(type_sequence)):
            if type_sequence[i] != type_sequence[i-1]:
                type_changes.append(f"{type_sequence[i-1]} -> {type_sequence[i]}")
        
        return {
            'type_sequence': type_sequence,
            'type_changes': type_changes,
            'type_stability': len(set(type_sequence)) == 1  # True if type never changes
        }
    
    def _update_function_trace(self, step: ExecutionStep, event: str, arg):
        """Update function trace information"""
        
        function_key = f"{step.module_name}.{step.function_name}"
        
        if event == 'call':
            # Start new function trace
            if function_key not in self.function_traces:
                self.function_traces[function_key] = FunctionTrace(
                    function_name=step.function_name,
                    module_name=step.module_name,
                    start_time=step.timestamp
                )
            
            # Add to call stack
            self.call_stack.append(step.step_id)
        
        elif event == 'return':
            # End function trace
            if function_key in self.function_traces:
                trace = self.function_traces[function_key]
                trace.end_time = step.timestamp
                trace.total_duration = (trace.end_time - trace.start_time).total_seconds()
                trace.return_value = self._safe_serialize(arg) if arg else None
            
            # Remove from call stack
            if self.call_stack and self.call_stack[-1].endswith(f"{step.function_name}_{step.line_number}"):
                self.call_stack.pop()
        
        elif event == 'exception':
            # Record exception in function trace
            if function_key in self.function_traces:
                trace = self.function_traces[function_key]
                trace.exception_info = step.error_info
        
        # Add step to function trace
        if function_key in self.function_traces:
            self.function_traces[function_key].steps.append(step)
            
            # Update performance metrics
            trace = self.function_traces[function_key]
            if step.memory_usage > trace.max_memory:
                trace.max_memory = step.memory_usage
            
            # Update average CPU (simple running average)
            if len(trace.steps) > 0:
                trace.avg_cpu = (trace.avg_cpu * (len(trace.steps) - 1) + step.cpu_percent) / len(trace.steps)
    
    def _generate_trace_summary(self) -> Dict[str, Any]:
        """Generate comprehensive trace summary"""
        
        if not self.start_time:
            return {}
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            'trace_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration': total_duration,
                'total_steps': len(self.execution_steps),
                'total_functions': len(self.function_traces)
            },
            'performance_summary': self._analyze_performance(),
            'function_analysis': self._analyze_functions(),
            'variable_analysis': self._analyze_variables(),
            'error_analysis': self._analyze_errors(),
            'execution_flow': self._analyze_execution_flow()
        }
        
        return summary
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance characteristics from trace"""
        
        if not self.execution_steps:
            return {}
        
        memory_values = [step.memory_usage for step in self.execution_steps if step.memory_usage > 0]
        cpu_values = [step.cpu_percent for step in self.execution_steps if step.cpu_percent > 0]
        
        analysis = {
            'memory_analysis': {
                'peak_memory_mb': max(memory_values) if memory_values else 0,
                'avg_memory_mb': np.mean(memory_values) if memory_values else 0,
                'memory_growth_mb': (max(memory_values) - min(memory_values)) if len(memory_values) > 1 else 0
            },
            'cpu_analysis': {
                'peak_cpu_percent': max(cpu_values) if cpu_values else 0,
                'avg_cpu_percent': np.mean(cpu_values) if cpu_values else 0
            },
            'slowest_functions': self._find_slowest_functions(),
            'memory_intensive_functions': self._find_memory_intensive_functions()
        }
        
        return analysis
    
    def _analyze_functions(self) -> Dict[str, Any]:
        """Analyze function execution patterns"""
        
        if not self.function_traces:
            return {}
        
        function_stats = {}
        
        for func_key, trace in self.function_traces.items():
            stats = {
                'total_duration': trace.total_duration,
                'max_memory': trace.max_memory,
                'avg_cpu': trace.avg_cpu,
                'step_count': len(trace.steps),
                'had_exception': trace.exception_info is not None,
                'max_stack_depth': max([step.stack_depth for step in trace.steps]) if trace.steps else 0
            }
            
            function_stats[func_key] = stats
        
        # Find top functions by various metrics
        analysis = {
            'total_functions': len(function_stats),
            'functions_with_errors': len([f for f in self.function_traces.values() if f.exception_info]),
            'top_by_duration': sorted(function_stats.items(), 
                                    key=lambda x: x[1]['total_duration'], reverse=True)[:5],
            'top_by_memory': sorted(function_stats.items(), 
                                  key=lambda x: x[1]['max_memory'], reverse=True)[:5],
            'top_by_steps': sorted(function_stats.items(), 
                                 key=lambda x: x[1]['step_count'], reverse=True)[:5]
        }
        
        return analysis
    
    def _analyze_variables(self) -> Dict[str, Any]:
        """Analyze variable usage patterns"""
        
        variable_patterns = {}
        variable_changes = {}
        
        for step in self.execution_steps:
            # Track variable appearances
            for var_name in step.local_variables.keys():
                if var_name not in variable_patterns:
                    variable_patterns[var_name] = {
                        'first_seen': step.timestamp,
                        'last_seen': step.timestamp,
                        'appearances': 0,
                        'functions': set()
                    }
                
                pattern = variable_patterns[var_name]
                pattern['last_seen'] = step.timestamp
                pattern['appearances'] += 1
                pattern['functions'].add(step.function_name)
            
            # Track variable value changes (simplified)
            for var_name, var_value in step.local_variables.items():
                if var_name not in variable_changes:
                    variable_changes[var_name] = []
                
                # Only track if different from last value
                if (not variable_changes[var_name] or 
                    variable_changes[var_name][-1]['value'] != var_value):
                    
                    variable_changes[var_name].append({
                        'timestamp': step.timestamp,
                        'value': var_value,
                        'function': step.function_name,
                        'line': step.line_number
                    })
        
        # Convert sets to lists for JSON serialization
        for pattern in variable_patterns.values():
            pattern['functions'] = list(pattern['functions'])
        
        analysis = {
            'total_variables_tracked': len(variable_patterns),
            'most_used_variables': sorted(variable_patterns.items(), 
                                        key=lambda x: x[1]['appearances'], reverse=True)[:10],
            'variables_with_changes': len([v for v in variable_changes.values() if len(v) > 1]),
            'variable_change_patterns': {k: v for k, v in variable_changes.items() if len(v) > 1}
        }
        
        return analysis
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze errors and exceptions from trace"""
        
        error_steps = [step for step in self.execution_steps if step.error_info]
        
        if not error_steps:
            return {'total_errors': 0}
        
        error_types = {}
        error_functions = {}
        
        for step in error_steps:
            error_type = step.error_info['exception_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            func_key = f"{step.module_name}.{step.function_name}"
            error_functions[func_key] = error_functions.get(func_key, 0) + 1
        
        analysis = {
            'total_errors': len(error_steps),
            'error_types': error_types,
            'functions_with_errors': error_functions,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
            'error_prone_function': max(error_functions.items(), key=lambda x: x[1])[0] if error_functions else None
        }
        
        return analysis
    
    def _analyze_execution_flow(self) -> Dict[str, Any]:
        """Analyze execution flow patterns"""
        
        if not self.execution_steps:
            return {}
        
        # Analyze call patterns
        call_steps = [step for step in self.execution_steps if step.event_type == 'call']
        return_steps = [step for step in self.execution_steps if step.event_type == 'return']
        
        # Calculate average function duration
        function_durations = {}
        for trace in self.function_traces.values():
            if trace.total_duration > 0:
                function_durations[f"{trace.module_name}.{trace.function_name}"] = trace.total_duration
        
        analysis = {
            'total_function_calls': len(call_steps),
            'total_function_returns': len(return_steps),
            'max_stack_depth': max([step.stack_depth for step in self.execution_steps]) if self.execution_steps else 0,
            'avg_function_duration': np.mean(list(function_durations.values())) if function_durations else 0,
            'execution_hotspots': self._find_execution_hotspots()
        }
        
        return analysis
    
    def _find_slowest_functions(self) -> List[Dict[str, Any]]:
        """Find functions that took the most time"""
        
        slowest = []
        
        for func_key, trace in self.function_traces.items():
            if trace.total_duration > 0:
                slowest.append({
                    'function': func_key,
                    'duration': trace.total_duration,
                    'step_count': len(trace.steps)
                })
        
        return sorted(slowest, key=lambda x: x['duration'], reverse=True)[:10]
    
    def _find_memory_intensive_functions(self) -> List[Dict[str, Any]]:
        """Find functions that used the most memory"""
        
        memory_intensive = []
        
        for func_key, trace in self.function_traces.items():
            if trace.max_memory > 0:
                memory_intensive.append({
                    'function': func_key,
                    'max_memory': trace.max_memory,
                    'avg_cpu': trace.avg_cpu
                })
        
        return sorted(memory_intensive, key=lambda x: x['max_memory'], reverse=True)[:10]
    
    def _find_execution_hotspots(self) -> List[Dict[str, Any]]:
        """Find code locations that execute frequently"""
        
        hotspots = {}
        
        for step in self.execution_steps:
            location = f"{step.module_name}:{step.function_name}:{step.line_number}"
            if location not in hotspots:
                hotspots[location] = {
                    'count': 0,
                    'total_time': 0.0,
                    'code_line': step.code_line
                }
            
            hotspots[location]['count'] += 1
            hotspots[location]['total_time'] += step.execution_time
        
        # Return top hotspots by execution count
        return sorted([
            {'location': loc, **data} 
            for loc, data in hotspots.items()
        ], key=lambda x: x['count'], reverse=True)[:10]
    
    def _save_trace_data(self, summary: Dict[str, Any]):
        """Save trace data to files"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary_file = self.output_dir / f"trace_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed steps (if not too large)
        if len(self.execution_steps) < 1000:
            steps_file = self.output_dir / f"execution_steps_{timestamp}.json"
            steps_data = [asdict(step) for step in self.execution_steps]
            
            with open(steps_file, 'w') as f:
                json.dump(steps_data, f, indent=2, default=str)
        
        # Save function traces
        traces_file = self.output_dir / f"function_traces_{timestamp}.json"
        traces_data = {}
        
        for func_key, trace in self.function_traces.items():
            # Don't include all steps in detailed save (too large)
            trace_copy = FunctionTrace(
                function_name=trace.function_name,
                module_name=trace.module_name,
                start_time=trace.start_time,
                end_time=trace.end_time,
                total_duration=trace.total_duration,
                max_memory=trace.max_memory,
                avg_cpu=trace.avg_cpu,
                return_value=trace.return_value,
                exception_info=trace.exception_info,
                steps=[]  # Don't include steps in summary
            )
            
            traces_data[func_key] = asdict(trace_copy)
        
        with open(traces_file, 'w') as f:
            json.dump(traces_data, f, indent=2, default=str)
        
        self.logger.info(f"Trace data saved to {self.output_dir}")
    
    def _trace_function(self, target_function: Callable):
        """Trace a specific function using a decorator approach"""
        
        original_function = target_function
        
        @functools.wraps(original_function)
        def traced_wrapper(*args, **kwargs):
            # Start function-specific tracing
            start_time = time.time()
            
            try:
                # Execute with sys.settrace for this function
                sys.settrace(self._trace_calls)
                result = original_function(*args, **kwargs)
                return result
            finally:
                sys.settrace(None)
                end_time = time.time()
                
                # Log function completion
                duration = end_time - start_time
                self.logger.debug(f"Traced function {target_function.__name__} completed in {duration:.3f}s")
        
        # Replace the original function
        if hasattr(target_function, '__name__'):
            globals()[target_function.__name__] = traced_wrapper
        
        return traced_wrapper

# Decorator for automatic function tracing
def trace_execution(tracer: ExecutionTracer = None, 
                   save_trace: bool = True,
                   max_steps: int = 1000):
    """Decorator to automatically trace function execution"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create tracer if not provided
            if tracer is None:
                function_tracer = ExecutionTracer(max_steps=max_steps)
            else:
                function_tracer = tracer
            
            # Start tracing
            function_tracer.start_tracing()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Stop tracing and save results
                summary = function_tracer.stop_tracing()
                
                if save_trace:
                    print(f"Execution trace completed for {func.__name__}")
                    print(f"Total steps: {summary.get('trace_info', {}).get('total_steps', 0)}")
                    print(f"Total duration: {summary.get('trace_info', {}).get('total_duration', 0):.3f}s")
        
        return wrapper
    return decorator

# Context manager for tracing code blocks
class TraceExecution:
    """Context manager for tracing code blocks"""
    
    def __init__(self, tracer: ExecutionTracer = None, description: str = "code_block"):
        self.tracer = tracer or ExecutionTracer()
        self.description = description
        self.summary = None
    
    def __enter__(self):
        self.tracer.start_tracing()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.summary = self.tracer.stop_tracing()
        
        print(f"\n=== Execution Trace Summary for {self.description} ===")
        if self.summary:
            trace_info = self.summary.get('trace_info', {})
            print(f"Duration: {trace_info.get('total_duration', 0):.3f} seconds")
            print(f"Steps traced: {trace_info.get('total_steps', 0)}")
            print(f"Functions traced: {trace_info.get('total_functions', 0)}")
            
            # Show performance highlights
            perf = self.summary.get('performance_summary', {})
            if perf:
                memory = perf.get('memory_analysis', {})
                print(f"Peak memory: {memory.get('peak_memory_mb', 0):.1f} MB")
                print(f"CPU usage: {perf.get('cpu_analysis', {}).get('avg_cpu_percent', 0):.1f}%")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get the execution summary"""
        return self.summary or {}

if __name__ == "__main__":
    # Example usage
    
    def example_function(n: int) -> int:
        """Example function to trace"""
        total = 0
        for i in range(n):
            total += i * i
            if i % 100 == 0:
                temp_list = [x for x in range(10)]  # Create some variables
        return total
    
    def another_function(data: list) -> float:
        """Another example function"""
        result = sum(data) / len(data) if data else 0
        return result * 1.5
    
    # Example 1: Use decorator
    @trace_execution(max_steps=500)
    def traced_computation():
        result1 = example_function(1000)
        data = [1, 2, 3, 4, 5]
        result2 = another_function(data)
        return result1, result2
    
    print("Running traced computation...")
    results = traced_computation()
    
    # Example 2: Use context manager
    print("\nUsing context manager...")
    with TraceExecution(description="manual_trace") as trace:
        x = example_function(500)
        y = another_function([1, 2, 3])
        z = x + y
    
    summary = trace.get_summary()
    if summary:
        print(f"Context manager trace captured {len(summary.get('function_analysis', {}).get('top_by_duration', []))} function calls")
    
    # Example 3: Manual tracer usage
    print("\nManual tracer usage...")
    manual_tracer = ExecutionTracer(max_steps=200)
    manual_tracer.start_tracing()
    
    try:
        result = example_function(300)
        print(f"Manual trace result: {result}")
    finally:
        summary = manual_tracer.stop_tracing()
        print(f"Manual trace completed with {summary.get('trace_info', {}).get('total_steps', 0)} steps")