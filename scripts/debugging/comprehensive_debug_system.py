"""
Comprehensive Debugging System Integration
Combines all debugging components into a unified, easy-to-use system
"""

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import logging
import functools
from contextlib import contextmanager

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.debugging.intelligent_debugger import IntelligentDebugger, DebugSession
from scripts.debugging.interactive_explorer import InteractiveDataExplorer, create_interactive_explorer
from scripts.debugging.execution_tracer import ExecutionTracer, TraceExecution
from scripts.debugging.self_healing_system import SelfHealingSystem, auto_heal

class ComprehensiveDebugSystem:
    """
    Master debugging system that integrates all components for seamless debugging experience
    """
    
    def __init__(self,
                 output_dir: str = "comprehensive_debug_output",
                 enable_auto_healing: bool = True,
                 enable_execution_tracing: bool = True,
                 enable_web_explorer: bool = True,
                 explorer_port: int = 5560,
                 auto_start: bool = False):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.enable_auto_healing = enable_auto_healing
        self.enable_execution_tracing = enable_execution_tracing
        self.enable_web_explorer = enable_web_explorer
        self.explorer_port = explorer_port
        
        # Initialize core components
        self.debugger = IntelligentDebugger(
            output_dir=str(self.output_dir / "intelligent_debug"),
            enable_anomaly_detection=True,
            enable_self_healing=enable_auto_healing
        )
        
        self.healing_system = SelfHealingSystem(
            debugger=self.debugger,
            output_dir=str(self.output_dir / "self_healing")
        ) if enable_auto_healing else None
        
        self.tracer = ExecutionTracer(
            output_dir=str(self.output_dir / "execution_traces")
        ) if enable_execution_tracing else None
        
        self.explorer = InteractiveDataExplorer(
            debugger=self.debugger,
            port=explorer_port
        ) if enable_web_explorer else None
        
        # State tracking
        self.active_sessions: Dict[str, 'IntelligentDebugSession'] = {}
        self.system_monitoring_active = False
        self.explorer_thread = None
        
        self.logger = logging.getLogger("comprehensive_debug_system")
        
        if auto_start:
            self.start_monitoring()
            if enable_web_explorer:
                self.launch_explorer()
        
        self.logger.info("ComprehensiveDebugSystem initialized with all components")
    
    def start_monitoring(self):
        """Start system-wide monitoring"""
        
        if self.system_monitoring_active:
            return
        
        self.system_monitoring_active = True
        
        # Install global error handlers
        self.debugger._install_global_handlers()
        
        self.logger.info("System-wide monitoring started")
    
    def stop_monitoring(self):
        """Stop system-wide monitoring"""
        
        if not self.system_monitoring_active:
            return
        
        self.system_monitoring_active = False
        
        # Stop all active sessions
        for session in self.active_sessions.values():
            session.end()
        
        # Stop tracer if running
        if self.tracer and self.tracer.is_tracing:
            self.tracer.stop_tracing()
        
        self.logger.info("System-wide monitoring stopped")
    
    def launch_explorer(self, open_browser: bool = True):
        """Launch the interactive web explorer"""
        
        if not self.enable_web_explorer or not self.explorer:
            self.logger.warning("Web explorer not enabled")
            return
        
        if self.explorer_thread and self.explorer_thread.is_alive():
            self.logger.info("Web explorer already running")
            return
        
        # Run explorer in separate thread
        self.explorer_thread = threading.Thread(
            target=lambda: self.explorer.run(open_browser=open_browser),
            daemon=True
        )
        self.explorer_thread.start()
        
        # Give it time to start
        time.sleep(2)
        
        self.logger.info(f"Web explorer launched at http://localhost:{self.explorer_port}")
    
    def create_debug_session(self, session_name: str) -> 'IntelligentDebugSession':
        """Create a new debugging session"""
        
        session = IntelligentDebugSession(
            session_name=session_name,
            debug_system=self
        )
        
        self.active_sessions[session_name] = session
        
        return session
    
    def get_session(self, session_name: str) -> Optional['IntelligentDebugSession']:
        """Get an existing debugging session"""
        
        return self.active_sessions.get(session_name)
    
    def heal_issue(self, error_info: Union[Exception, str], 
                   context: Dict[str, Any] = None) -> bool:
        """Attempt to heal an issue using the self-healing system"""
        
        if not self.healing_system:
            return False
        
        healing_action = self.healing_system.attempt_healing(error_info, context)
        
        self.logger.info(f"Healing attempt: {healing_action.action_description} - {'Success' if healing_action.success else 'Failed'}")
        
        return healing_action.success
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'system_info': {
                'monitoring_active': self.system_monitoring_active,
                'output_directory': str(self.output_dir),
                'active_sessions': list(self.active_sessions.keys()),
                'components_enabled': {
                    'auto_healing': self.enable_auto_healing,
                    'execution_tracing': self.enable_execution_tracing,
                    'web_explorer': self.enable_web_explorer
                }
            },
            'debugger_stats': {
                'total_events': len(self.debugger.debug_events),
                'recent_events': len([e for e in self.debugger.debug_events 
                                    if (datetime.now() - e.timestamp).total_seconds() < 3600])
            }
        }
        
        if self.healing_system:
            status['healing_stats'] = self.healing_system.get_healing_statistics()
        
        if self.explorer:
            status['explorer_info'] = {
                'port': self.explorer_port,
                'datasets_loaded': len(self.explorer.datasets),
                'variables_tracked': len(self.explorer.variables)
            }
        
        return status
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive debugging report"""
        
        report = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'system_status': self.get_system_status()
            },
            'debug_events_analysis': self._analyze_debug_events(),
            'performance_analysis': self._analyze_performance(),
            'recommendations': self._generate_recommendations()
        }
        
        # Add session reports
        if self.active_sessions:
            report['session_reports'] = {}
            for session_name, session in self.active_sessions.items():
                report['session_reports'][session_name] = session.generate_session_report()
        
        # Save report
        report_file = self.output_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report saved to {report_file}")
        
        return report
    
    def _analyze_debug_events(self) -> Dict[str, Any]:
        """Analyze debug events across the system"""
        
        events = self.debugger.debug_events
        
        if not events:
            return {'message': 'No debug events to analyze'}
        
        # Categorize events
        events_by_type = {}
        events_by_severity = {}
        recent_events = []
        
        cutoff_time = datetime.now()
        one_hour_ago = cutoff_time.timestamp() - 3600
        
        for event in events:
            # By type
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
            
            # By severity
            events_by_severity[event.severity] = events_by_severity.get(event.severity, 0) + 1
            
            # Recent events
            if event.timestamp.timestamp() > one_hour_ago:
                recent_events.append(event)
        
        return {
            'total_events': len(events),
            'events_by_type': events_by_type,
            'events_by_severity': events_by_severity,
            'recent_events_count': len(recent_events),
            'critical_events': len([e for e in events if e.severity == 'critical']),
            'most_common_type': max(events_by_type.items(), key=lambda x: x[1])[0] if events_by_type else None
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance patterns"""
        
        analysis = {
            'execution_traces': {},
            'healing_performance': {},
            'system_health': {}
        }
        
        # Execution traces analysis
        if self.tracer and self.tracer.execution_steps:
            steps = self.tracer.execution_steps
            analysis['execution_traces'] = {
                'total_steps': len(steps),
                'avg_memory_usage': sum(s.memory_usage for s in steps) / len(steps) if steps else 0,
                'peak_memory_usage': max(s.memory_usage for s in steps) if steps else 0,
                'functions_traced': len(self.tracer.function_traces)
            }
        
        # Healing performance analysis
        if self.healing_system:
            healing_stats = self.healing_system.get_healing_statistics()
            analysis['healing_performance'] = {
                'success_rate': healing_stats.get('success_rate', 0),
                'total_attempts': healing_stats.get('total_attempts', 0),
                'most_successful_type': max(healing_stats.get('success_by_type', {}).items(), 
                                          key=lambda x: x[1])[0] if healing_stats.get('success_by_type') else None
            }
        
        return analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        
        recommendations = []
        
        # Analyze debug events
        events = self.debugger.debug_events
        if events:
            error_count = len([e for e in events if e.event_type == 'error'])
            if error_count > 10:
                recommendations.append(
                    f"High error count ({error_count}) detected. Consider implementing more robust error handling."
                )
            
            anomaly_count = len([e for e in events if e.event_type == 'anomaly'])
            if anomaly_count > 5:
                recommendations.append(
                    f"Multiple anomalies detected ({anomaly_count}). Review data quality and processing pipelines."
                )
        
        # Analyze healing system
        if self.healing_system:
            stats = self.healing_system.get_healing_statistics()
            success_rate = stats.get('success_rate', 0)
            
            if success_rate < 0.5:
                recommendations.append(
                    f"Low healing success rate ({success_rate:.1%}). Review and improve healing strategies."
                )
            elif success_rate > 0.8:
                recommendations.append(
                    "High healing success rate! Consider expanding self-healing to more error types."
                )
        
        # Performance recommendations
        if self.tracer and len(self.tracer.execution_steps) > 1000:
            recommendations.append(
                "High execution trace volume. Consider filtering trace targets for better performance."
            )
        
        # Explorer recommendations
        if self.explorer and len(self.explorer.datasets) == 0:
            recommendations.append(
                "No datasets loaded in explorer. Add datasets for better debugging insights."
            )
        
        return recommendations

class IntelligentDebugSession:
    """
    High-level debugging session that provides easy-to-use interface for experiment debugging
    """
    
    def __init__(self, session_name: str, debug_system: ComprehensiveDebugSystem):
        self.session_name = session_name
        self.debug_system = debug_system
        self.start_time = datetime.now()
        self.end_time = None
        
        # Session-specific data
        self.session_data: Dict[str, Any] = {}
        self.logged_metrics: Dict[str, List[float]] = {}
        self.session_context: Dict[str, Any] = {}
        
        # Initialize session with debugger
        self.debug_session = DebugSession(session_name, debug_system.debugger)
        
        self.logger = logging.getLogger(f"debug_session_{session_name}")
        
        self.logger.info(f"Debug session '{session_name}' started")
    
    def __enter__(self):
        self.debug_session.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end()
        self.debug_session.__exit__(exc_type, exc_value, traceback)
    
    def add_data(self, name: str, data: Any, metadata: Dict[str, Any] = None):
        """Add data for monitoring and exploration"""
        
        self.session_data[name] = data
        
        # Add to explorer if available
        if self.debug_system.explorer:
            if hasattr(data, 'shape'):  # DataFrame or array
                self.debug_system.explorer.add_dataset(f"{self.session_name}_{name}", data, metadata)
            else:
                self.debug_system.explorer.add_variable(f"{self.session_name}_{name}", data, metadata)
        
        # Check data quality
        if hasattr(data, 'isnull'):  # pandas DataFrame
            issues = self.debug_system.debugger.detect_data_quality_issues(data, name)
            if issues:
                self.logger.warning(f"Data quality issues in {name}: {issues}")
    
    def log_metric(self, name: str, value: float):
        """Log a metric for anomaly detection"""
        
        if name not in self.logged_metrics:
            self.logged_metrics[name] = []
        
        self.logged_metrics[name].append(value)
        
        # Check for anomalies
        anomalies = self.debug_system.debugger.detect_statistical_anomalies(
            {name: value}, f"{self.session_name}_metrics"
        )
        
        if anomalies:
            self.logger.warning(f"Anomaly detected in metric {name}: {anomalies[0].description}")
        
        # Log to debug session
        if hasattr(self.debug_session, 'log_metric'):
            self.debug_session.log_metric(name, value)
    
    def set_context(self, **kwargs):
        """Set context variables for the session"""
        
        self.session_context.update(kwargs)
        
        # Add variables to explorer
        if self.debug_system.explorer:
            for key, value in kwargs.items():
                self.debug_system.explorer.add_variable(f"{self.session_name}_ctx_{key}", value)
    
    def trace_function(self, func: Callable) -> Callable:
        """Add execution tracing to a function"""
        
        if not self.debug_system.tracer:
            return func
        
        @functools.wraps(func)
        def traced_wrapper(*args, **kwargs):
            # Start tracing for this function
            with TraceExecution(self.debug_system.tracer, f"{self.session_name}_{func.__name__}"):
                return func(*args, **kwargs)
        
        return traced_wrapper
    
    def auto_heal_function(self, func: Callable, max_attempts: int = 3) -> Callable:
        """Add automatic healing to a function"""
        
        if not self.debug_system.healing_system:
            return func
        
        return auto_heal(self.debug_system.healing_system, max_attempts)(func)
    
    def heal_issue(self, error_info: Union[Exception, str]) -> bool:
        """Attempt to heal an issue in this session"""
        
        context = {
            'session_name': self.session_name,
            'session_context': self.session_context,
            'session_data': self.session_data
        }
        
        return self.debug_system.heal_issue(error_info, context)
    
    def get_session_insights(self) -> Dict[str, Any]:
        """Get insights and recommendations for this session"""
        
        insights = {
            'session_info': {
                'name': self.session_name,
                'duration': (datetime.now() - self.start_time).total_seconds(),
                'data_items': len(self.session_data),
                'metrics_logged': len(self.logged_metrics),
                'context_variables': len(self.session_context)
            },
            'data_analysis': {},
            'metric_analysis': {},
            'recommendations': []
        }
        
        # Analyze logged metrics
        for metric_name, values in self.logged_metrics.items():
            if len(values) >= 2:
                insights['metric_analysis'][metric_name] = {
                    'count': len(values),
                    'latest_value': values[-1],
                    'trend': 'increasing' if values[-1] > values[0] else 'decreasing',
                    'variance': float(np.var(values)) if len(values) > 1 else 0,
                    'anomalies_detected': len([
                        e for e in self.debug_system.debugger.debug_events 
                        if e.component == f"{self.session_name}_metrics" and metric_name in e.message
                    ])
                }
        
        # Generate recommendations
        if not self.session_data:
            insights['recommendations'].append("No data loaded. Consider adding datasets for better debugging insights.")
        
        if not self.logged_metrics:
            insights['recommendations'].append("No metrics logged. Track key performance metrics for anomaly detection.")
        
        # Check for common issues
        errors_in_session = [
            e for e in self.debug_system.debugger.debug_events
            if self.session_name in e.context.get('session_name', '')
        ]
        
        if len(errors_in_session) > 3:
            insights['recommendations'].append(
                f"Multiple errors detected in this session ({len(errors_in_session)}). Consider using auto-healing."
            )
        
        return insights
    
    def generate_session_report(self) -> Dict[str, Any]:
        """Generate a comprehensive session report"""
        
        report = {
            'session_info': {
                'name': self.session_name,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration': (self.end_time or datetime.now() - self.start_time).total_seconds()
            },
            'session_insights': self.get_session_insights(),
            'debug_events': [
                e for e in self.debug_system.debugger.debug_events
                if e.timestamp >= self.start_time and (not self.end_time or e.timestamp <= self.end_time)
            ]
        }
        
        return report
    
    def end(self):
        """End the debugging session"""
        
        if self.end_time:
            return  # Already ended
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        self.logger.info(f"Debug session '{self.session_name}' ended after {duration:.1f} seconds")
        
        # Remove from active sessions
        if self.session_name in self.debug_system.active_sessions:
            del self.debug_system.active_sessions[self.session_name]

# Convenience functions
def create_comprehensive_debug_system(
    output_dir: str = "comprehensive_debug_output",
    enable_all: bool = True,
    auto_start: bool = True,
    explorer_port: int = 5560
) -> ComprehensiveDebugSystem:
    """Create a comprehensive debugging system with all features enabled"""
    
    return ComprehensiveDebugSystem(
        output_dir=output_dir,
        enable_auto_healing=enable_all,
        enable_execution_tracing=enable_all,
        enable_web_explorer=enable_all,
        explorer_port=explorer_port,
        auto_start=auto_start
    )

@contextmanager
def debug_experiment(experiment_name: str, 
                    debug_system: ComprehensiveDebugSystem = None,
                    auto_launch_explorer: bool = True):
    """Context manager for debugging experiments"""
    
    if debug_system is None:
        debug_system = create_comprehensive_debug_system(auto_start=True)
    
    if auto_launch_explorer and debug_system.enable_web_explorer:
        debug_system.launch_explorer()
    
    session = debug_system.create_debug_session(experiment_name)
    
    try:
        with session:
            yield session
    finally:
        # Generate final report
        insights = session.get_session_insights()
        print(f"\n=== Debug Session Summary: {experiment_name} ===")
        print(f"Duration: {insights['session_info']['duration']:.1f} seconds")
        print(f"Data items: {insights['session_info']['data_items']}")
        print(f"Metrics logged: {insights['session_info']['metrics_logged']}")
        
        if insights['recommendations']:
            print("\nRecommendations:")
            for rec in insights['recommendations']:
                print(f"  - {rec}")

if __name__ == "__main__":
    # Example usage of the comprehensive debugging system
    
    import numpy as np
    import pandas as pd
    
    print("=== Comprehensive Debugging System Demo ===")
    
    # Example 1: Basic usage with context manager
    print("\n1. Basic debugging session:")
    
    with debug_experiment("demo_experiment") as debug:
        # Add some test data
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        debug.add_data("training_data", test_data)
        debug.set_context(learning_rate=0.01, batch_size=32)
        
        # Log some metrics
        for i in range(10):
            accuracy = 0.7 + 0.03 * i + np.random.normal(0, 0.02)
            debug.log_metric("accuracy", accuracy)
            
            loss = 0.5 - 0.02 * i + np.random.normal(0, 0.05)
            debug.log_metric("loss", loss)
    
    # Example 2: Using individual components
    print("\n2. Advanced usage with individual components:")
    
    debug_system = create_comprehensive_debug_system(auto_start=False)
    debug_system.start_monitoring()
    
    # Create a function with automatic healing
    @auto_heal(debug_system.healing_system)
    def potentially_failing_function(x):
        if x < 0.3:  # Simulate random failures
            raise ValueError("Random failure for demonstration")
        return x * 2
    
    # Test the auto-healing
    for i in range(5):
        try:
            value = np.random.random()
            result = potentially_failing_function(value)
            print(f"  Function call {i+1}: {value:.2f} -> {result:.2f}")
        except Exception as e:
            print(f"  Function call {i+1}: Failed with {e}")
    
    # Show system status
    print("\n3. System Status:")
    status = debug_system.get_system_status()
    print(f"  Monitoring active: {status['system_info']['monitoring_active']}")
    print(f"  Debug events: {status['debugger_stats']['total_events']}")
    
    if 'healing_stats' in status:
        healing_stats = status['healing_stats']
        print(f"  Healing attempts: {healing_stats['total_attempts']}")
        print(f"  Healing success rate: {healing_stats['success_rate']:.1%}")
    
    print("\n=== Demo completed! ===")
    print("Check the comprehensive_debug_output directory for detailed logs and reports.")
    
    # Optionally launch web explorer
    user_input = input("\nLaunch web explorer? (y/N): ")
    if user_input.lower() == 'y':
        debug_system.launch_explorer()
        print("Web explorer launched! Check your browser.")
        input("Press Enter to exit...")