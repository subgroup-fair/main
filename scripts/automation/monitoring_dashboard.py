"""
Real-time Monitoring Dashboard for Experiment Execution
Provides web-based dashboard for monitoring experiment progress, metrics, and resource usage
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import asdict
import webbrowser

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import plotly.graph_objs as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.automation.resource_monitor import ResourceMonitor
from scripts.automation.experiment_executor import ExperimentExecutor

class ExperimentDashboard:
    """
    Real-time dashboard for monitoring experiment execution
    """
    
    def __init__(self, 
                 executor: Optional[ExperimentExecutor] = None,
                 resource_monitor: Optional[ResourceMonitor] = None,
                 port: int = 5555,
                 debug: bool = False):
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the dashboard. Install with: pip install flask")
        
        self.executor = executor
        self.resource_monitor = resource_monitor
        self.port = port
        self.debug = debug
        
        # Dashboard state
        self.experiment_history: List[Dict] = []
        self.current_experiments: Dict[str, Dict] = {}
        self.performance_metrics: Dict[str, List] = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'experiment_progress': []
        }
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder=self._create_template_dir(),
                        static_folder=self._create_static_dir())
        
        self.logger = logging.getLogger("experiment_dashboard")
        
        # Background data collection
        self.data_collection_active = False
        self.data_collection_thread = None
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info(f"ExperimentDashboard initialized on port {port}")
    
    def _create_template_dir(self) -> str:
        """Create templates directory and HTML files"""
        
        template_dir = Path(__file__).parent / "dashboard_templates"
        template_dir.mkdir(exist_ok=True)
        
        # Create main dashboard template
        dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card { margin-bottom: 20px; }
        .status-running { color: #28a745; }
        .status-completed { color: #6c757d; }
        .status-failed { color: #dc3545; }
        .chart-container { height: 400px; margin-bottom: 30px; }
        .refresh-indicator { position: fixed; top: 10px; right: 10px; }
        .experiment-card { border-left: 4px solid #007bff; }
        .progress-bar-container { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <div class="col-12">
                <h1 class="mt-3 mb-4">
                    <i class="fas fa-chart-line"></i> Experiment Monitoring Dashboard
                    <span class="refresh-indicator">
                        <span id="refresh-status" class="badge bg-success">Live</span>
                        <small id="last-update" class="text-muted"></small>
                    </span>
                </h1>
            </div>
        </div>
        
        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Active Experiments</h5>
                        <h3 id="active-count" class="text-primary">-</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Completed</h5>
                        <h3 id="completed-count" class="text-success">-</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Failed</h5>
                        <h3 id="failed-count" class="text-danger">-</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Success Rate</h5>
                        <h3 id="success-rate" class="text-info">-</h3>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Resource Usage Charts -->
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>System Resource Usage</h5>
                    </div>
                    <div class="card-body">
                        <div id="resource-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Experiment Progress</h5>
                    </div>
                    <div class="card-body">
                        <div id="progress-chart" class="chart-container"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Current Experiments -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Current Experiments</h5>
                    </div>
                    <div class="card-body">
                        <div id="current-experiments">
                            <p class="text-muted">No active experiments</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Results -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Results</h5>
                    </div>
                    <div class="card-body">
                        <div id="recent-results">
                            <p class="text-muted">No recent results</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Dashboard JavaScript
        let refreshInterval;
        
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateSummaryCards(data);
                    updateCharts(data);
                    updateCurrentExperiments(data);
                    updateRecentResults(data);
                    
                    $('#last-update').text('Last updated: ' + new Date().toLocaleTimeString());
                    $('#refresh-status').removeClass('bg-danger').addClass('bg-success').text('Live');
                })
                .catch(error => {
                    console.error('Error updating dashboard:', error);
                    $('#refresh-status').removeClass('bg-success').addClass('bg-danger').text('Error');
                });
        }
        
        function updateSummaryCards(data) {
            $('#active-count').text(data.summary.active_experiments);
            $('#completed-count').text(data.summary.completed_experiments);
            $('#failed-count').text(data.summary.failed_experiments);
            $('#success-rate').text(data.summary.success_rate + '%');
        }
        
        function updateCharts(data) {
            // Resource usage chart
            if (data.resource_data && data.resource_data.length > 0) {
                const timestamps = data.resource_data.map(d => d.timestamp);
                const cpuData = data.resource_data.map(d => d.cpu_percent);
                const memoryData = data.resource_data.map(d => d.memory_percent);
                
                const resourceTraces = [
                    {
                        x: timestamps,
                        y: cpuData,
                        name: 'CPU %',
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#ff7f0e' }
                    },
                    {
                        x: timestamps,
                        y: memoryData,
                        name: 'Memory %',
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#2ca02c' }
                    }
                ];
                
                const resourceLayout = {
                    title: '',
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Usage %', range: [0, 100] },
                    margin: { t: 30, r: 30, b: 50, l: 50 }
                };
                
                Plotly.newPlot('resource-chart', resourceTraces, resourceLayout);
            }
            
            // Progress chart
            if (data.progress_data && data.progress_data.length > 0) {
                const progressTrace = {
                    x: data.progress_data.map(d => d.experiment_name),
                    y: data.progress_data.map(d => d.progress_percent),
                    type: 'bar',
                    marker: { color: '#1f77b4' }
                };
                
                const progressLayout = {
                    title: '',
                    xaxis: { title: 'Experiments' },
                    yaxis: { title: 'Progress %', range: [0, 100] },
                    margin: { t: 30, r: 30, b: 80, l: 50 }
                };
                
                Plotly.newPlot('progress-chart', [progressTrace], progressLayout);
            }
        }
        
        function updateCurrentExperiments(data) {
            const container = $('#current-experiments');
            
            if (data.current_experiments && data.current_experiments.length > 0) {
                let html = '';
                data.current_experiments.forEach(exp => {
                    const statusClass = `status-${exp.status}`;
                    const progressWidth = exp.progress_percent || 0;
                    
                    html += `
                        <div class="experiment-card card mb-3">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div>
                                        <h6 class="card-title">${exp.job_id}</h6>
                                        <p class="card-text">${exp.experiment_type}</p>
                                        <span class="badge ${statusClass}">${exp.status}</span>
                                    </div>
                                    <div class="text-end">
                                        <small class="text-muted">Started: ${exp.start_time}</small>
                                        <div class="progress-bar-container">
                                            <div class="progress">
                                                <div class="progress-bar" style="width: ${progressWidth}%">${progressWidth}%</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                container.html(html);
            } else {
                container.html('<p class="text-muted">No active experiments</p>');
            }
        }
        
        function updateRecentResults(data) {
            const container = $('#recent-results');
            
            if (data.recent_results && data.recent_results.length > 0) {
                let html = '<div class="table-responsive">';
                html += '<table class="table table-striped">';
                html += '<thead><tr><th>Job ID</th><th>Status</th><th>Duration</th><th>Accuracy</th><th>Fairness</th></tr></thead><tbody>';
                
                data.recent_results.forEach(result => {
                    const duration = result.duration ? `${result.duration.toFixed(1)}s` : 'N/A';
                    const accuracy = result.accuracy ? `${(result.accuracy * 100).toFixed(1)}%` : 'N/A';
                    const fairness = result.fairness ? result.fairness.toFixed(3) : 'N/A';
                    
                    html += `
                        <tr>
                            <td>${result.job_id}</td>
                            <td><span class="badge status-${result.status}">${result.status}</span></td>
                            <td>${duration}</td>
                            <td>${accuracy}</td>
                            <td>${fairness}</td>
                        </tr>
                    `;
                });
                
                html += '</tbody></table></div>';
                container.html(html);
            } else {
                container.html('<p class="text-muted">No recent results</p>');
            }
        }
        
        // Auto-refresh every 5 seconds
        $(document).ready(function() {
            updateDashboard();
            refreshInterval = setInterval(updateDashboard, 5000);
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        });
    </script>
</body>
</html>'''
        
        with open(template_dir / "dashboard.html", 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        return str(template_dir)
    
    def _create_static_dir(self) -> str:
        """Create static files directory"""
        
        static_dir = Path(__file__).parent / "dashboard_static"
        static_dir.mkdir(exist_ok=True)
        
        return str(static_dir)
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify(self._get_dashboard_data())
        
        @self.app.route('/api/experiments/<job_id>')
        def api_experiment_details(job_id):
            return jsonify(self._get_experiment_details(job_id))
        
        @self.app.route('/api/resource_data')
        def api_resource_data():
            return jsonify(self._get_resource_data())
        
        @self.app.route('/api/export_data')
        def api_export_data():
            return jsonify(self._export_dashboard_data())
    
    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._get_experiment_summary(),
            'resource_data': self._get_recent_resource_data(),
            'current_experiments': self._get_current_experiments_data(),
            'recent_results': self._get_recent_results_data(),
            'progress_data': self._get_progress_data()
        }
        
        return data
    
    def _get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment execution summary"""
        
        if not self.executor:
            return {
                'active_experiments': 0,
                'completed_experiments': 0,
                'failed_experiments': 0,
                'success_rate': 0
            }
        
        status = self.executor.get_status()
        
        total_jobs = status['jobs_completed'] + status['jobs_failed']
        success_rate = (status['jobs_completed'] / total_jobs * 100) if total_jobs > 0 else 0
        
        return {
            'active_experiments': status['jobs_running'],
            'completed_experiments': status['jobs_completed'],
            'failed_experiments': status['jobs_failed'],
            'success_rate': round(success_rate, 1),
            'queued_experiments': status['jobs_in_queue']
        }
    
    def _get_recent_resource_data(self) -> List[Dict[str, Any]]:
        """Get recent resource usage data for charts"""
        
        if not self.resource_monitor or not self.resource_monitor.is_monitoring():
            return []
        
        # Get last 50 data points (about 5 minutes at 1-second intervals)
        recent_data = self.resource_monitor.get_usage_history(minutes=5)
        
        return [
            {
                'timestamp': snapshot.timestamp.strftime('%H:%M:%S'),
                'cpu_percent': snapshot.cpu_percent,
                'memory_percent': snapshot.memory_percent,
                'memory_gb': snapshot.memory_gb,
                'disk_usage_percent': snapshot.disk_usage_percent
            }
            for snapshot in recent_data[-50:]  # Last 50 points
        ]
    
    def _get_current_experiments_data(self) -> List[Dict[str, Any]]:
        """Get data for currently running experiments"""
        
        if not self.executor:
            return []
        
        current_experiments = []
        
        # Get running experiments from executor
        for job_id, job_info in self.executor.running_jobs.items():
            experiment_data = {
                'job_id': job_id,
                'experiment_type': job_info['job'].experiment_type.value,
                'status': 'running',
                'start_time': job_info['start_time'].strftime('%H:%M:%S'),
                'progress_percent': self._estimate_progress(job_info),
                'estimated_completion': self._estimate_completion_time(job_info)
            }
            
            current_experiments.append(experiment_data)
        
        return current_experiments
    
    def _get_recent_results_data(self) -> List[Dict[str, Any]]:
        """Get recent experiment results"""
        
        if not self.executor:
            return []
        
        recent_results = []
        
        # Get last 10 completed experiments
        all_results = self.executor.completed_jobs + self.executor.failed_jobs
        sorted_results = sorted(all_results, key=lambda x: x.end_time or x.start_time, reverse=True)
        
        for result in sorted_results[:10]:
            duration = None
            if result.end_time:
                duration = (result.end_time - result.start_time).total_seconds()
            
            # Extract accuracy and fairness if available
            accuracy = None
            fairness = None
            
            if result.results and 'results' in result.results:
                # Try to extract metrics from nested structure
                metrics = self._extract_summary_metrics(result.results)
                accuracy = metrics.get('accuracy')
                fairness = metrics.get('fairness')
            
            result_data = {
                'job_id': result.job_id,
                'status': result.status,
                'duration': duration,
                'accuracy': accuracy,
                'fairness': fairness,
                'end_time': result.end_time.strftime('%H:%M:%S') if result.end_time else 'N/A'
            }
            
            recent_results.append(result_data)
        
        return recent_results
    
    def _get_progress_data(self) -> List[Dict[str, Any]]:
        """Get progress data for progress chart"""
        
        if not self.executor:
            return []
        
        progress_data = []
        
        # Add progress for running experiments
        for job_id, job_info in self.executor.running_jobs.items():
            progress_data.append({
                'experiment_name': job_id[:20] + '...' if len(job_id) > 20 else job_id,
                'progress_percent': self._estimate_progress(job_info)
            })
        
        return progress_data
    
    def _estimate_progress(self, job_info: Dict[str, Any]) -> float:
        """Estimate experiment progress based on elapsed time"""
        
        if 'start_time' not in job_info:
            return 0.0
        
        elapsed = datetime.now() - job_info['start_time']
        elapsed_minutes = elapsed.total_seconds() / 60
        
        # Estimate based on timeout (assuming linear progress)
        timeout_minutes = job_info['job'].timeout_minutes
        
        progress = min(95.0, (elapsed_minutes / timeout_minutes) * 100)
        
        return round(progress, 1)
    
    def _estimate_completion_time(self, job_info: Dict[str, Any]) -> str:
        """Estimate completion time for running experiment"""
        
        if 'start_time' not in job_info:
            return 'Unknown'
        
        elapsed = datetime.now() - job_info['start_time']
        elapsed_minutes = elapsed.total_seconds() / 60
        
        # Simple linear estimation based on timeout
        timeout_minutes = job_info['job'].timeout_minutes
        
        if elapsed_minutes < timeout_minutes:
            remaining_minutes = timeout_minutes - elapsed_minutes
            completion_time = datetime.now() + timedelta(minutes=remaining_minutes)
            return completion_time.strftime('%H:%M:%S')
        
        return 'Overdue'
    
    def _extract_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract summary metrics from experiment results"""
        
        metrics = {}
        
        try:
            if 'datasets' in results:
                all_accuracies = []
                all_fairness = []
                
                for dataset_name, dataset_results in results['datasets'].items():
                    if 'methods' in dataset_results:
                        for method_name, method_results in dataset_results['methods'].items():
                            if 'lambda_sweep' in method_results:
                                for sweep_point in method_results['lambda_sweep']:
                                    if 'accuracy_mean' in sweep_point:
                                        all_accuracies.append(sweep_point['accuracy_mean'])
                                    if 'sup_ipm_mean' in sweep_point:
                                        all_fairness.append(sweep_point['sup_ipm_mean'])
                
                if all_accuracies:
                    metrics['accuracy'] = sum(all_accuracies) / len(all_accuracies)
                if all_fairness:
                    metrics['fairness'] = sum(all_fairness) / len(all_fairness)
        
        except Exception as e:
            self.logger.debug(f"Error extracting metrics: {e}")
        
        return metrics
    
    def _get_experiment_details(self, job_id: str) -> Dict[str, Any]:
        """Get detailed information for specific experiment"""
        
        # Search in running jobs
        if self.executor and job_id in self.executor.running_jobs:
            job_info = self.executor.running_jobs[job_id]
            return {
                'job_id': job_id,
                'status': 'running',
                'start_time': job_info['start_time'].isoformat(),
                'experiment_type': job_info['job'].experiment_type.value,
                'parameters': asdict(job_info['job'].params)
            }
        
        # Search in completed jobs
        if self.executor:
            for result in self.executor.completed_jobs + self.executor.failed_jobs:
                if result.job_id == job_id:
                    return asdict(result)
        
        return {'error': 'Experiment not found'}
    
    def _get_resource_data(self) -> Dict[str, Any]:
        """Get detailed resource usage data"""
        
        if not self.resource_monitor:
            return {'error': 'Resource monitor not available'}
        
        summary = self.resource_monitor.get_summary(minutes=60)  # Last hour
        current_usage = self.resource_monitor.get_current_usage()
        
        return {
            'summary': summary,
            'current': asdict(current_usage) if current_usage else None,
            'alerts': self.resource_monitor.get_resource_alerts()
        }
    
    def _export_dashboard_data(self) -> Dict[str, Any]:
        """Export all dashboard data for analysis"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'dashboard_data': self._get_dashboard_data(),
            'resource_summary': self._get_resource_data(),
            'experiment_history': self.experiment_history
        }
        
        return export_data
    
    def start_data_collection(self):
        """Start background data collection"""
        
        if self.data_collection_active:
            return
        
        self.data_collection_active = True
        self.data_collection_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self.data_collection_thread.start()
        
        self.logger.info("Dashboard data collection started")
    
    def stop_data_collection(self):
        """Stop background data collection"""
        
        self.data_collection_active = False
        
        if self.data_collection_thread:
            self.data_collection_thread.join(timeout=5)
        
        self.logger.info("Dashboard data collection stopped")
    
    def _data_collection_loop(self):
        """Background loop for collecting performance metrics"""
        
        while self.data_collection_active:
            try:
                current_time = datetime.now()
                
                # Collect resource metrics
                if self.resource_monitor:
                    current_usage = self.resource_monitor.get_current_usage()
                    
                    if current_usage:
                        self.performance_metrics['timestamps'].append(current_time)
                        self.performance_metrics['cpu_usage'].append(current_usage.cpu_percent)
                        self.performance_metrics['memory_usage'].append(current_usage.memory_percent)
                        
                        # GPU usage if available
                        gpu_usage = 0
                        if current_usage.gpu_usage:
                            gpu_usage = sum(gpu['load'] for gpu in current_usage.gpu_usage) / len(current_usage.gpu_usage)
                        self.performance_metrics['gpu_usage'].append(gpu_usage)
                
                # Collect experiment progress
                if self.executor:
                    experiment_count = len(self.executor.running_jobs)
                    self.performance_metrics['experiment_progress'].append(experiment_count)
                
                # Maintain history size
                max_history = 1000
                for key in self.performance_metrics:
                    if len(self.performance_metrics[key]) > max_history:
                        self.performance_metrics[key] = self.performance_metrics[key][-max_history:]
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in data collection loop: {e}")
                time.sleep(5)
    
    def run(self, 
            host: str = '127.0.0.1',
            open_browser: bool = True,
            auto_start_monitoring: bool = True):
        """Start the dashboard server"""
        
        if auto_start_monitoring:
            self.start_data_collection()
        
        if open_browser:
            # Open browser after a short delay
            threading.Timer(1.0, lambda: webbrowser.open(f'http://{host}:{self.port}')).start()
        
        self.logger.info(f"Starting dashboard server on http://{host}:{self.port}")
        
        try:
            self.app.run(host=host, port=self.port, debug=self.debug, use_reloader=False)
        finally:
            self.stop_data_collection()

# Convenience function
def create_experiment_dashboard(executor: Optional[ExperimentExecutor] = None,
                              resource_monitor: Optional[ResourceMonitor] = None,
                              port: int = 5555) -> ExperimentDashboard:
    """Create and configure experiment dashboard"""
    
    # Create resource monitor if not provided
    if resource_monitor is None and executor is not None:
        resource_monitor = ResourceMonitor()
        resource_monitor.start_monitoring()
    
    dashboard = ExperimentDashboard(
        executor=executor,
        resource_monitor=resource_monitor,
        port=port
    )
    
    return dashboard

def run_standalone_dashboard(port: int = 5555):
    """Run standalone dashboard without executor (for testing)"""
    
    print(f"Starting standalone experiment dashboard on port {port}")
    print("This dashboard will show simulated data for testing purposes")
    
    dashboard = ExperimentDashboard(port=port)
    dashboard.run(open_browser=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Monitoring Dashboard")
    parser.add_argument("--port", type=int, default=5555, help="Port to run dashboard on")
    parser.add_argument("--standalone", action="store_true", help="Run standalone without executor")
    
    args = parser.parse_args()
    
    if args.standalone:
        run_standalone_dashboard(args.port)
    else:
        print("To use with experiment executor:")
        print("from monitoring_dashboard import create_experiment_dashboard")
        print("dashboard = create_experiment_dashboard(executor)")
        print("dashboard.run()")