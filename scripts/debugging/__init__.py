"""
Intelligent Debugging System for Research Experiments

A comprehensive debugging and self-healing system that provides:
- Automatic error detection and anomaly detection
- Interactive data exploration and visualization
- Step-by-step execution tracing with variable inspection
- Self-healing capabilities with automatic recovery
- Performance profiling and bottleneck identification
- Intelligent suggestions and recommendations

Quick Start:
    from scripts.debugging import IntelligentDebugSession
    
    # Create a debugging session
    with IntelligentDebugSession("my_experiment") as debug:
        # Your experiment code here
        debug.add_data("training_data", your_data)
        debug.log_metric("accuracy", 0.85)
        
        # The system automatically monitors for issues
        # and provides suggestions for improvements

Components:
- IntelligentDebugger: Core error detection and diagnosis
- InteractiveDataExplorer: Web-based data exploration
- ExecutionTracer: Step-by-step execution tracking
- SelfHealingSystem: Automatic issue resolution

Advanced Usage:
    from scripts.debugging import create_comprehensive_debug_system
    
    # Create full debugging system with all components
    debug_system = create_comprehensive_debug_system()
    debug_system.start_monitoring()
    debug_system.launch_explorer()
"""

from .intelligent_debugger import (
    IntelligentDebugger,
    DebugEvent,
    AnomalyDetection,
    debug_monitor,
    DebugSession
)

from .interactive_explorer import (
    InteractiveDataExplorer,
    create_interactive_explorer
)

from .execution_tracer import (
    ExecutionTracer,
    ExecutionStep,
    FunctionTrace,
    trace_execution,
    TraceExecution
)

from .self_healing_system import (
    SelfHealingSystem,
    HealingAction,
    ParameterSuggestion,
    MethodRecommendation,
    auto_heal
)

from .comprehensive_debug_system import (
    ComprehensiveDebugSystem,
    IntelligentDebugSession,
    create_comprehensive_debug_system,
    debug_experiment
)

__version__ = "1.0.0"
__author__ = "Intelligent Debugging System"

# Main components for easy import
__all__ = [
    # Core components
    "IntelligentDebugger",
    "InteractiveDataExplorer", 
    "ExecutionTracer",
    "SelfHealingSystem",
    
    # Integrated system
    "ComprehensiveDebugSystem",
    "IntelligentDebugSession",
    "create_comprehensive_debug_system",
    
    # Decorators and context managers
    "debug_monitor",
    "trace_execution", 
    "auto_heal",
    "debug_experiment",
    "DebugSession",
    "TraceExecution",
    
    # Data classes
    "DebugEvent",
    "AnomalyDetection", 
    "ExecutionStep",
    "FunctionTrace",
    "HealingAction",
    "ParameterSuggestion",
    "MethodRecommendation",
    
    # Utility functions
    "create_interactive_explorer"
]