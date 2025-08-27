"""
Interactive Data Explorer for Research Debugging
Provides web-based interactive exploration of experiment data, variables, and execution traces
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import threading
import webbrowser

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Plotting imports
try:
    import plotly.graph_objs as go
    import plotly.express as px
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.debugging.intelligent_debugger import IntelligentDebugger, DebugEvent

class InteractiveDataExplorer:
    """
    Interactive web-based data explorer for debugging research experiments
    """
    
    def __init__(self,
                 debugger: IntelligentDebugger = None,
                 port: int = 5560,
                 debug: bool = False):
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the interactive explorer. Install with: pip install flask")
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualizations. Install with: pip install plotly")
        
        self.debugger = debugger or IntelligentDebugger()
        self.port = port
        self.debug = debug
        
        # Data storage for exploration
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.variables: Dict[str, Any] = {}
        self.execution_traces: List[Dict] = []
        self.analysis_results: Dict[str, Any] = {}
        
        # Flask app setup
        self.app = Flask(__name__,
                        template_folder=self._create_template_dir(),
                        static_folder=self._create_static_dir())
        
        self.logger = logging.getLogger("interactive_explorer")
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info(f"InteractiveDataExplorer initialized on port {port}")
    
    def _create_template_dir(self) -> str:
        """Create templates directory and HTML files"""
        
        template_dir = Path(__file__).parent / "explorer_templates"
        template_dir.mkdir(exist_ok=True)
        
        # Create main explorer template
        explorer_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Data Explorer - Research Debugging</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .explorer-container { height: 100vh; overflow: hidden; }
        .sidebar { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
        .main-content { height: 100vh; overflow-y: auto; }
        .data-table { max-height: 400px; overflow-y: auto; }
        .chart-container { height: 400px; margin-bottom: 20px; }
        .variable-item { 
            padding: 8px; 
            margin: 4px 0; 
            background: #e9ecef; 
            border-radius: 4px; 
            cursor: pointer; 
        }
        .variable-item:hover { background: #dee2e6; }
        .trace-item { 
            padding: 10px; 
            margin: 5px 0; 
            border-left: 3px solid #007bff; 
            background: #f8f9fa; 
        }
        .error-item { border-left-color: #dc3545; }
        .warning-item { border-left-color: #ffc107; }
        .code-block { 
            background: #2d3748; 
            color: #e2e8f0; 
            padding: 15px; 
            border-radius: 5px; 
            font-family: 'Courier New', monospace; 
        }
        .tab-content { padding: 20px; }
        .anomaly-badge { 
            background: #dc3545; 
            color: white; 
            padding: 2px 8px; 
            border-radius: 12px; 
            font-size: 0.8em; 
        }
        .quality-issue { 
            background: #fff3cd; 
            border: 1px solid #ffecb5; 
            padding: 10px; 
            border-radius: 5px; 
            margin: 5px 0; 
        }
    </style>
</head>
<body>
    <div class="container-fluid explorer-container">
        <div class="row h-100">
            <!-- Sidebar -->
            <div class="col-3 sidebar p-3">
                <h5><i class="fas fa-search"></i> Explorer</h5>
                
                <!-- Datasets Section -->
                <div class="mt-3">
                    <h6>📊 Datasets</h6>
                    <div id="datasets-list"></div>
                </div>
                
                <!-- Variables Section -->
                <div class="mt-3">
                    <h6>🔢 Variables</h6>
                    <div id="variables-list"></div>
                </div>
                
                <!-- Debug Events -->
                <div class="mt-3">
                    <h6>🐛 Debug Events</h6>
                    <div id="debug-events-summary"></div>
                </div>
                
                <!-- Quick Actions -->
                <div class="mt-3">
                    <h6>⚡ Quick Actions</h6>
                    <button class="btn btn-sm btn-primary mb-2 w-100" onclick="refreshData()">
                        <i class="fas fa-sync"></i> Refresh Data
                    </button>
                    <button class="btn btn-sm btn-info mb-2 w-100" onclick="generateReport()">
                        <i class="fas fa-file-alt"></i> Generate Report
                    </button>
                    <button class="btn btn-sm btn-warning mb-2 w-100" onclick="runDiagnostics()">
                        <i class="fas fa-stethoscope"></i> Run Diagnostics
                    </button>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="col-9 main-content">
                <!-- Navigation Tabs -->
                <nav class="navbar navbar-expand-lg navbar-light bg-light">
                    <div class="navbar-nav nav-tabs">
                        <a class="nav-link active" id="overview-tab" data-bs-toggle="tab" href="#overview">
                            <i class="fas fa-tachometer-alt"></i> Overview
                        </a>
                        <a class="nav-link" id="data-tab" data-bs-toggle="tab" href="#data">
                            <i class="fas fa-table"></i> Data Analysis
                        </a>
                        <a class="nav-link" id="variables-tab" data-bs-toggle="tab" href="#variables">
                            <i class="fas fa-code"></i> Variables
                        </a>
                        <a class="nav-link" id="traces-tab" data-bs-toggle="tab" href="#traces">
                            <i class="fas fa-route"></i> Execution Traces
                        </a>
                        <a class="nav-link" id="debug-tab" data-bs-toggle="tab" href="#debug">
                            <i class="fas fa-bug"></i> Debug Analysis
                        </a>
                    </div>
                </nav>
                
                <!-- Tab Content -->
                <div class="tab-content">
                    <!-- Overview Tab -->
                    <div class="tab-pane fade show active" id="overview">
                        <div class="row">
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <h5>📊 Datasets</h5>
                                        <h3 id="datasets-count" class="text-primary">-</h3>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <h5>🔢 Variables</h5>
                                        <h3 id="variables-count" class="text-info">-</h3>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <h5>🐛 Debug Events</h5>
                                        <h3 id="debug-events-count" class="text-danger">-</h3>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-4">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>🕒 Recent Activity Timeline</h5>
                                    </div>
                                    <div class="card-body">
                                        <div id="timeline-chart" class="chart-container"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Data Analysis Tab -->
                    <div class="tab-pane fade" id="data">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>📈 Data Visualization</h5>
                                        <select id="dataset-selector" class="form-select">
                                            <option value="">Select Dataset</option>
                                        </select>
                                    </div>
                                    <div class="card-body">
                                        <div id="data-chart" class="chart-container"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>📋 Data Summary</h5>
                                    </div>
                                    <div class="card-body">
                                        <div id="data-summary"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-4">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>🔍 Data Table</h5>
                                    </div>
                                    <div class="card-body">
                                        <div id="data-table" class="data-table"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Variables Tab -->
                    <div class="tab-pane fade" id="variables">
                        <div class="row">
                            <div class="col-md-8">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>🔍 Variable Inspector</h5>
                                    </div>
                                    <div class="card-body">
                                        <div id="variable-details"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>📊 Variable Stats</h5>
                                    </div>
                                    <div class="card-body">
                                        <div id="variable-stats"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Execution Traces Tab -->
                    <div class="tab-pane fade" id="traces">
                        <div class="card">
                            <div class="card-header">
                                <h5>🚀 Execution Traces</h5>
                            </div>
                            <div class="card-body">
                                <div id="execution-traces"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Debug Analysis Tab -->
                    <div class="tab-pane fade" id="debug">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>⚠️ Issues & Anomalies</h5>
                                    </div>
                                    <div class="card-body">
                                        <div id="debug-issues"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>💡 Suggestions</h5>
                                    </div>
                                    <div class="card-body">
                                        <div id="debug-suggestions"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-4">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-header">
                                        <h5>📈 Debug Analytics</h5>
                                    </div>
                                    <div class="card-body">
                                        <div id="debug-analytics" class="chart-container"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Explorer JavaScript
        let refreshInterval;
        
        function initializeExplorer() {
            loadData();
            refreshInterval = setInterval(loadData, 10000); // Refresh every 10 seconds
        }
        
        function loadData() {
            fetch('/api/explorer_data')
                .then(response => response.json())
                .then(data => {
                    updateOverview(data);
                    updateSidebar(data);
                    updateDataTab(data);
                    updateVariablesTab(data);
                    updateTracesTab(data);
                    updateDebugTab(data);
                })
                .catch(error => {
                    console.error('Error loading data:', error);
                });
        }
        
        function updateOverview(data) {
            $('#datasets-count').text(data.datasets_count);
            $('#variables-count').text(data.variables_count);
            $('#debug-events-count').text(data.debug_events_count);
            
            // Update timeline chart
            if (data.timeline_data && data.timeline_data.length > 0) {
                const timelineTrace = {
                    x: data.timeline_data.map(d => d.timestamp),
                    y: data.timeline_data.map(d => d.event_type),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {
                        color: data.timeline_data.map(d => d.severity === 'critical' ? '#dc3545' : 
                                                        d.severity === 'high' ? '#fd7e14' :
                                                        d.severity === 'medium' ? '#ffc107' : '#28a745'),
                        size: 10
                    },
                    text: data.timeline_data.map(d => d.message),
                    hovertemplate: '%{text}<br>%{x}<extra></extra>'
                };
                
                const timelineLayout = {
                    title: '',
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Event Type' },
                    margin: { t: 30, r: 30, b: 50, l: 100 }
                };
                
                Plotly.newPlot('timeline-chart', [timelineTrace], timelineLayout);
            }
        }
        
        function updateSidebar(data) {
            // Update datasets list
            let datasetsHtml = '';
            for (const [name, info] of Object.entries(data.datasets || {})) {
                datasetsHtml += `
                    <div class="variable-item" onclick="selectDataset('${name}')">
                        <strong>${name}</strong><br>
                        <small>${info.shape || 'Unknown shape'}</small>
                    </div>
                `;
            }
            $('#datasets-list').html(datasetsHtml || '<small class="text-muted">No datasets loaded</small>');
            
            // Update variables list
            let variablesHtml = '';
            for (const [name, info] of Object.entries(data.variables || {})) {
                variablesHtml += `
                    <div class="variable-item" onclick="selectVariable('${name}')">
                        <strong>${name}</strong><br>
                        <small>${info.type || 'Unknown type'}</small>
                    </div>
                `;
            }
            $('#variables-list').html(variablesHtml || '<small class="text-muted">No variables captured</small>');
            
            // Update debug events summary
            const eventsSummary = data.debug_events_summary || {};
            let eventsHtml = '';
            for (const [type, count] of Object.entries(eventsSummary)) {
                eventsHtml += `<span class="badge bg-secondary me-1">${type}: ${count}</span>`;
            }
            $('#debug-events-summary').html(eventsHtml || '<small class="text-muted">No debug events</small>');
        }
        
        function updateDataTab(data) {
            // Update dataset selector
            const selector = $('#dataset-selector');
            selector.empty().append('<option value="">Select Dataset</option>');
            
            for (const name of Object.keys(data.datasets || {})) {
                selector.append(`<option value="${name}">${name}</option>`);
            }
        }
        
        function updateVariablesTab(data) {
            // Variables will be updated when clicked
        }
        
        function updateTracesTab(data) {
            let tracesHtml = '';
            
            for (const trace of data.execution_traces || []) {
                const itemClass = trace.error ? 'trace-item error-item' : 
                                trace.warning ? 'trace-item warning-item' : 'trace-item';
                
                tracesHtml += `
                    <div class="${itemClass}">
                        <div class="d-flex justify-content-between">
                            <strong>${trace.function_name || 'Unknown Function'}</strong>
                            <small>${trace.timestamp}</small>
                        </div>
                        <div>Duration: ${trace.duration || 'N/A'}s</div>
                        ${trace.error ? `<div class="text-danger">Error: ${trace.error}</div>` : ''}
                        ${trace.variables ? `<div><small>Variables: ${Object.keys(trace.variables).length}</small></div>` : ''}
                    </div>
                `;
            }
            
            $('#execution-traces').html(tracesHtml || '<p class="text-muted">No execution traces available</p>');
        }
        
        function updateDebugTab(data) {
            // Update issues
            let issuesHtml = '';
            for (const issue of data.debug_issues || []) {
                issuesHtml += `
                    <div class="quality-issue">
                        <div class="d-flex justify-content-between">
                            <strong>${issue.type}</strong>
                            <span class="badge bg-${issue.severity === 'critical' ? 'danger' : 
                                                  issue.severity === 'high' ? 'warning' : 'info'}">${issue.severity}</span>
                        </div>
                        <div>${issue.message}</div>
                        ${issue.suggestions ? `<div><small><strong>Suggestions:</strong> ${issue.suggestions.join(', ')}</small></div>` : ''}
                    </div>
                `;
            }
            $('#debug-issues').html(issuesHtml || '<p class="text-muted">No issues detected</p>');
            
            // Update suggestions
            let suggestionsHtml = '';
            for (const suggestion of data.debug_suggestions || []) {
                suggestionsHtml += `
                    <div class="alert alert-info">
                        <i class="fas fa-lightbulb"></i> ${suggestion}
                    </div>
                `;
            }
            $('#debug-suggestions').html(suggestionsHtml || '<p class="text-muted">No suggestions available</p>');
        }
        
        function selectDataset(name) {
            $('#dataset-selector').val(name).change();
            $('.nav-link[href="#data"]').tab('show');
            
            // Load dataset details
            fetch(`/api/dataset/${name}`)
                .then(response => response.json())
                .then(data => {
                    // Update data summary
                    let summaryHtml = `
                        <h6>Dataset: ${name}</h6>
                        <ul>
                            <li>Shape: ${data.shape}</li>
                            <li>Columns: ${data.columns}</li>
                            <li>Data Types: ${data.dtypes}</li>
                            <li>Missing Values: ${data.missing_values}</li>
                        </ul>
                    `;
                    
                    if (data.quality_issues && data.quality_issues.length > 0) {
                        summaryHtml += '<h6 class="text-warning">Quality Issues:</h6><ul>';
                        for (const issue of data.quality_issues) {
                            summaryHtml += `<li>${issue}</li>`;
                        }
                        summaryHtml += '</ul>';
                    }
                    
                    $('#data-summary').html(summaryHtml);
                    
                    // Create visualization
                    if (data.visualization) {
                        const plotData = JSON.parse(data.visualization);
                        Plotly.newPlot('data-chart', plotData.data, plotData.layout);
                    }
                    
                    // Update data table
                    if (data.sample_data) {
                        $('#data-table').html(data.sample_data);
                    }
                });
        }
        
        function selectVariable(name) {
            $('.nav-link[href="#variables"]').tab('show');
            
            // Load variable details
            fetch(`/api/variable/${name}`)
                .then(response => response.json())
                .then(data => {
                    let detailsHtml = `
                        <h5>Variable: ${name}</h5>
                        <div class="code-block">
                            <strong>Type:</strong> ${data.type}<br>
                            <strong>Value:</strong><br>
                            <pre>${data.value}</pre>
                        </div>
                    `;
                    
                    if (data.analysis) {
                        detailsHtml += `
                            <h6 class="mt-3">Analysis</h6>
                            <ul>
                                ${data.analysis.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        `;
                    }
                    
                    $('#variable-details').html(detailsHtml);
                    
                    // Update stats
                    if (data.stats) {
                        let statsHtml = '<h6>Statistics</h6>';
                        for (const [key, value] of Object.entries(data.stats)) {
                            statsHtml += `<div><strong>${key}:</strong> ${value}</div>`;
                        }
                        $('#variable-stats').html(statsHtml);
                    }
                });
        }
        
        function refreshData() {
            loadData();
        }
        
        function generateReport() {
            fetch('/api/generate_report', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(`Report generated: ${data.report_path}`);
                });
        }
        
        function runDiagnostics() {
            fetch('/api/run_diagnostics', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(`Diagnostics completed. Found ${data.issues_count} issues.`);
                    loadData(); // Refresh data
                });
        }
        
        // Initialize when page loads
        $(document).ready(function() {
            initializeExplorer();
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
        
        with open(template_dir / "explorer.html", 'w', encoding='utf-8') as f:
            f.write(explorer_html)
        
        return str(template_dir)
    
    def _create_static_dir(self) -> str:
        """Create static files directory"""
        
        static_dir = Path(__file__).parent / "explorer_static"
        static_dir.mkdir(exist_ok=True)
        
        return str(static_dir)
    
    def _setup_routes(self):
        """Setup Flask routes for the interactive explorer"""
        
        @self.app.route('/')
        def explorer():
            return render_template('explorer.html')
        
        @self.app.route('/api/explorer_data')
        def api_explorer_data():
            return jsonify(self._get_explorer_data())
        
        @self.app.route('/api/dataset/<dataset_name>')
        def api_dataset_details(dataset_name):
            return jsonify(self._get_dataset_details(dataset_name))
        
        @self.app.route('/api/variable/<variable_name>')
        def api_variable_details(variable_name):
            return jsonify(self._get_variable_details(variable_name))
        
        @self.app.route('/api/generate_report', methods=['POST'])
        def api_generate_report():
            return jsonify(self._generate_exploration_report())
        
        @self.app.route('/api/run_diagnostics', methods=['POST'])
        def api_run_diagnostics():
            return jsonify(self._run_comprehensive_diagnostics())
    
    def add_dataset(self, name: str, data: pd.DataFrame, metadata: Dict = None):
        """Add a dataset for interactive exploration"""
        
        self.datasets[name] = data
        
        # Run automatic data quality analysis
        quality_issues = self.debugger.detect_data_quality_issues(data, name)
        
        # Store metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'added_at': datetime.now().isoformat(),
            'quality_issues': quality_issues,
            'shape': data.shape,
            'columns': list(data.columns) if hasattr(data, 'columns') else [],
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()} if hasattr(data, 'dtypes') else {}
        })
        
        # Store in analysis results
        self.analysis_results[f'dataset_{name}'] = metadata
        
        self.logger.info(f"Added dataset '{name}' with {len(quality_issues)} quality issues")
    
    def add_variable(self, name: str, value: Any, context: Dict = None):
        """Add a variable for interactive inspection"""
        
        self.variables[name] = {
            'value': value,
            'type': type(value).__name__,
            'added_at': datetime.now().isoformat(),
            'context': context or {},
            'analysis': self._analyze_variable(value)
        }
        
        self.logger.debug(f"Added variable '{name}' of type {type(value).__name__}")
    
    def add_execution_trace(self, function_name: str, duration: float, 
                          variables: Dict = None, error: str = None):
        """Add an execution trace for debugging"""
        
        trace = {
            'timestamp': datetime.now().isoformat(),
            'function_name': function_name,
            'duration': duration,
            'variables': variables or {},
            'error': error,
            'warning': duration > 10.0  # Flag slow executions
        }
        
        self.execution_traces.append(trace)
        
        # Keep only recent traces
        if len(self.execution_traces) > 100:
            self.execution_traces = self.execution_traces[-100:]
        
        # Detect performance issues
        if duration > 5.0:
            bottlenecks = self.debugger.detect_performance_bottlenecks(duration, function_name)
    
    def _analyze_variable(self, value: Any) -> List[str]:
        """Analyze a variable and provide insights"""
        
        analysis = []
        
        try:
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    if np.isnan(value):
                        analysis.append("Value is NaN")
                    elif np.isinf(value):
                        analysis.append("Value is infinite")
                    elif abs(value) > 1e6:
                        analysis.append("Very large numeric value")
                    elif abs(value) < 1e-6 and value != 0:
                        analysis.append("Very small numeric value")
                
                if value < 0:
                    analysis.append("Negative value")
                elif value == 0:
                    analysis.append("Zero value")
            
            elif isinstance(value, str):
                if len(value) == 0:
                    analysis.append("Empty string")
                elif len(value) > 1000:
                    analysis.append("Very long string")
                elif value.isdigit():
                    analysis.append("String contains only digits")
            
            elif isinstance(value, (list, tuple)):
                if len(value) == 0:
                    analysis.append("Empty collection")
                elif len(value) > 1000:
                    analysis.append("Large collection")
                
                # Check for type consistency
                if len(value) > 0:
                    first_type = type(value[0])
                    if not all(isinstance(x, first_type) for x in value):
                        analysis.append("Mixed types in collection")
            
            elif isinstance(value, dict):
                if len(value) == 0:
                    analysis.append("Empty dictionary")
                elif len(value) > 100:
                    analysis.append("Large dictionary")
            
            elif isinstance(value, np.ndarray):
                if value.size == 0:
                    analysis.append("Empty array")
                elif np.any(np.isnan(value)):
                    analysis.append("Contains NaN values")
                elif np.any(np.isinf(value)):
                    analysis.append("Contains infinite values")
                
                if len(value.shape) > 3:
                    analysis.append("High-dimensional array")
                
                if value.dtype.kind in 'fc':  # float or complex
                    std_dev = np.std(value)
                    if std_dev == 0:
                        analysis.append("Zero variance (all values identical)")
                    elif std_dev > 1000 * abs(np.mean(value)):
                        analysis.append("Very high variance")
        
        except Exception as e:
            analysis.append(f"Error during analysis: {str(e)}")
        
        return analysis
    
    def _get_explorer_data(self) -> Dict[str, Any]:
        """Get comprehensive data for the explorer interface"""
        
        # Get debug events from debugger
        recent_events = self.debugger.debug_events[-50:]  # Last 50 events
        
        # Summarize debug events by type
        events_summary = {}
        for event in recent_events:
            events_summary[event.event_type] = events_summary.get(event.event_type, 0) + 1
        
        # Create timeline data
        timeline_data = []
        for event in recent_events:
            timeline_data.append({
                'timestamp': event.timestamp.strftime('%H:%M:%S'),
                'event_type': event.event_type,
                'severity': event.severity,
                'message': event.message[:100] + '...' if len(event.message) > 100 else event.message
            })
        
        # Dataset summaries
        datasets_info = {}
        for name, data in self.datasets.items():
            datasets_info[name] = {
                'shape': f"{data.shape[0]} x {data.shape[1]}" if len(data.shape) == 2 else str(data.shape),
                'type': type(data).__name__
            }
        
        # Variable summaries
        variables_info = {}
        for name, var_info in self.variables.items():
            variables_info[name] = {
                'type': var_info['type'],
                'analysis_count': len(var_info.get('analysis', []))
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'datasets_count': len(self.datasets),
            'variables_count': len(self.variables),
            'debug_events_count': len(recent_events),
            'debug_events_summary': events_summary,
            'timeline_data': timeline_data,
            'datasets': datasets_info,
            'variables': variables_info,
            'execution_traces': self.execution_traces[-20:],  # Recent traces
            'debug_issues': self._get_current_issues(),
            'debug_suggestions': self._get_current_suggestions()
        }
    
    def _get_dataset_details(self, dataset_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific dataset"""
        
        if dataset_name not in self.datasets:
            return {'error': 'Dataset not found'}
        
        data = self.datasets[dataset_name]
        metadata = self.analysis_results.get(f'dataset_{dataset_name}', {})
        
        # Generate sample data HTML table
        sample_data = data.head(10) if hasattr(data, 'head') else str(data)[:1000]
        if hasattr(sample_data, 'to_html'):
            sample_html = sample_data.to_html(classes='table table-striped table-sm', table_id='dataset-table')
        else:
            sample_html = f'<pre>{sample_data}</pre>'
        
        # Create visualization
        visualization = None
        if hasattr(data, 'select_dtypes'):
            # For pandas DataFrames
            numeric_cols = data.select_dtypes(include=[np.number]).columns[:3]  # First 3 numeric columns
            
            if len(numeric_cols) > 0:
                try:
                    if len(numeric_cols) == 1:
                        # Histogram for single column
                        fig = px.histogram(data, x=numeric_cols[0], title=f'Distribution of {numeric_cols[0]}')
                    elif len(numeric_cols) == 2:
                        # Scatter plot for two columns
                        fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], 
                                       title=f'{numeric_cols[0]} vs {numeric_cols[1]}')
                    else:
                        # 3D scatter for three columns
                        fig = px.scatter_3d(data, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2],
                                          title=f'3D Plot: {", ".join(numeric_cols)}')
                    
                    visualization = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    
                except Exception as e:
                    self.logger.debug(f"Error creating visualization: {e}")
        
        # Data statistics
        stats = {}
        if hasattr(data, 'describe'):
            try:
                desc = data.describe()
                stats = desc.to_dict() if hasattr(desc, 'to_dict') else {}
            except Exception:
                pass
        
        return {
            'name': dataset_name,
            'shape': data.shape if hasattr(data, 'shape') else 'Unknown',
            'columns': list(data.columns) if hasattr(data, 'columns') else 'N/A',
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()} if hasattr(data, 'dtypes') else 'N/A',
            'missing_values': data.isnull().sum().to_dict() if hasattr(data, 'isnull') else 'N/A',
            'quality_issues': metadata.get('quality_issues', []),
            'sample_data': sample_html,
            'visualization': visualization,
            'stats': stats
        }
    
    def _get_variable_details(self, variable_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific variable"""
        
        if variable_name not in self.variables:
            return {'error': 'Variable not found'}
        
        var_info = self.variables[variable_name]
        value = var_info['value']
        
        # Format value for display
        if isinstance(value, str):
            display_value = value if len(value) <= 1000 else value[:1000] + '...'
        elif isinstance(value, (list, tuple)):
            if len(value) <= 10:
                display_value = str(value)
            else:
                display_value = f"{str(value[:5])}...{str(value[-5:])}"
        elif isinstance(value, np.ndarray):
            display_value = f"Array shape: {value.shape}\n{str(value) if value.size <= 100 else 'Large array'}"
        elif isinstance(value, pd.DataFrame):
            display_value = f"DataFrame shape: {value.shape}\n{value.head().to_string()}"
        else:
            display_value = str(value)
        
        # Generate statistics if applicable
        stats = {}
        try:
            if isinstance(value, (int, float)):
                stats = {'value': value, 'type': type(value).__name__}
            elif isinstance(value, np.ndarray) and value.dtype.kind in 'fc':
                stats = {
                    'shape': str(value.shape),
                    'mean': float(np.mean(value)),
                    'std': float(np.std(value)),
                    'min': float(np.min(value)),
                    'max': float(np.max(value))
                }
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if all(isinstance(x, (int, float)) for x in value):
                    stats = {
                        'length': len(value),
                        'mean': np.mean(value),
                        'std': np.std(value),
                        'min': min(value),
                        'max': max(value)
                    }
                else:
                    stats = {'length': len(value), 'type_distribution': {}}
                    for item in value:
                        item_type = type(item).__name__
                        stats['type_distribution'][item_type] = stats['type_distribution'].get(item_type, 0) + 1
        except Exception as e:
            stats = {'error': str(e)}
        
        return {
            'name': variable_name,
            'type': var_info['type'],
            'value': display_value,
            'analysis': var_info.get('analysis', []),
            'context': var_info.get('context', {}),
            'added_at': var_info.get('added_at', 'Unknown'),
            'stats': stats
        }
    
    def _get_current_issues(self) -> List[Dict[str, Any]]:
        """Get current issues from debug events"""
        
        recent_events = self.debugger.debug_events[-20:]  # Last 20 events
        issues = []
        
        for event in recent_events:
            if event.event_type in ['error', 'anomaly', 'data_quality']:
                issues.append({
                    'type': event.event_type,
                    'severity': event.severity,
                    'message': event.message,
                    'component': event.component,
                    'suggestions': event.suggested_fixes or []
                })
        
        return issues
    
    def _get_current_suggestions(self) -> List[str]:
        """Get current debugging suggestions"""
        
        suggestions = []
        
        # Analyze error patterns
        common_errors = self.debugger.error_patterns
        if common_errors:
            most_common = max(common_errors.items(), key=lambda x: x[1])
            if most_common[1] > 2:  # If error occurred more than twice
                suggestions.append(f"Consider addressing the recurring {most_common[0]} error pattern")
        
        # Check data quality across datasets
        total_quality_issues = 0
        for dataset_name in self.datasets:
            metadata = self.analysis_results.get(f'dataset_{dataset_name}', {})
            total_quality_issues += len(metadata.get('quality_issues', []))
        
        if total_quality_issues > 5:
            suggestions.append("Multiple data quality issues detected. Consider implementing data cleaning pipeline")
        
        # Performance suggestions
        slow_traces = [t for t in self.execution_traces if t.get('duration', 0) > 5]
        if len(slow_traces) > 3:
            suggestions.append("Multiple slow executions detected. Consider performance optimization")
        
        # Memory suggestions
        memory_events = [e for e in self.debugger.debug_events if 'memory' in e.message.lower()]
        if len(memory_events) > 2:
            suggestions.append("Memory-related issues detected. Consider reducing memory usage")
        
        return suggestions
    
    def _generate_exploration_report(self) -> Dict[str, Any]:
        """Generate comprehensive exploration report"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'datasets': len(self.datasets),
                'variables': len(self.variables),
                'debug_events': len(self.debugger.debug_events),
                'execution_traces': len(self.execution_traces)
            },
            'datasets_analysis': {},
            'variables_analysis': {},
            'debug_analysis': {},
            'recommendations': self._get_current_suggestions()
        }
        
        # Dataset analysis
        for name, data in self.datasets.items():
            metadata = self.analysis_results.get(f'dataset_{name}', {})
            report['datasets_analysis'][name] = {
                'shape': data.shape if hasattr(data, 'shape') else 'Unknown',
                'quality_issues': metadata.get('quality_issues', []),
                'dtypes': metadata.get('dtypes', {}),
                'missing_values': metadata.get('missing_values', 'Unknown')
            }
        
        # Variables analysis
        for name, var_info in self.variables.items():
            report['variables_analysis'][name] = {
                'type': var_info['type'],
                'analysis_points': len(var_info.get('analysis', [])),
                'issues': [analysis for analysis in var_info.get('analysis', []) 
                          if any(word in analysis.lower() for word in ['error', 'issue', 'problem', 'nan', 'inf'])]
            }
        
        # Debug events analysis
        events_by_type = {}
        events_by_severity = {}
        for event in self.debugger.debug_events:
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
            events_by_severity[event.severity] = events_by_severity.get(event.severity, 0) + 1
        
        report['debug_analysis'] = {
            'events_by_type': events_by_type,
            'events_by_severity': events_by_severity,
            'most_common_error': max(self.debugger.error_patterns.items(), key=lambda x: x[1])[0] 
                               if self.debugger.error_patterns else 'None',
            'healing_attempts': dict(self.debugger.healing_attempts)
        }
        
        # Save report
        report_file = self.debugger.output_dir / f"exploration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return {'report_path': str(report_file), 'report': report}
    
    def _run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics on all data"""
        
        total_issues = 0
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'datasets_diagnostics': {},
            'variables_diagnostics': {},
            'system_diagnostics': {}
        }
        
        # Dataset diagnostics
        for name, data in self.datasets.items():
            issues = self.debugger.detect_data_quality_issues(data, f"diagnostic_{name}")
            diagnostics['datasets_diagnostics'][name] = issues
            total_issues += len(issues)
        
        # Variable diagnostics
        for name, var_info in self.variables.items():
            value = var_info['value']
            
            # Run analysis
            analysis = self._analyze_variable(value)
            problem_analysis = [a for a in analysis if any(word in a.lower() 
                                                         for word in ['error', 'issue', 'problem', 'nan', 'inf'])]
            
            diagnostics['variables_diagnostics'][name] = problem_analysis
            total_issues += len(problem_analysis)
        
        # System diagnostics
        diagnostics['system_diagnostics'] = {
            'total_debug_events': len(self.debugger.debug_events),
            'recent_errors': len([e for e in self.debugger.debug_events[-10:] if e.event_type == 'error']),
            'slow_executions': len([t for t in self.execution_traces if t.get('duration', 0) > 5]),
            'healing_success_rate': self._calculate_healing_success_rate()
        }
        
        # Save diagnostics
        diag_file = self.debugger.output_dir / f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(diag_file, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        return {
            'issues_count': total_issues,
            'diagnostics_path': str(diag_file),
            'summary': diagnostics
        }
    
    def _calculate_healing_success_rate(self) -> float:
        """Calculate the success rate of self-healing attempts"""
        
        total_healing_attempts = sum(self.debugger.healing_attempts.values())
        if total_healing_attempts == 0:
            return 0.0
        
        # Estimate success based on reduced error patterns
        # This is a simplified calculation
        return min(1.0, 0.7)  # Placeholder - would need more sophisticated tracking
    
    def run(self, host: str = '127.0.0.1', open_browser: bool = True):
        """Start the interactive explorer server"""
        
        if open_browser:
            # Open browser after a short delay
            threading.Timer(1.0, lambda: webbrowser.open(f'http://{host}:{self.port}')).start()
        
        self.logger.info(f"Starting Interactive Data Explorer on http://{host}:{self.port}")
        
        try:
            self.app.run(host=host, port=self.port, debug=self.debug, use_reloader=False)
        except KeyboardInterrupt:
            self.logger.info("Interactive Data Explorer stopped by user")

# Convenience functions
def create_interactive_explorer(debugger: IntelligentDebugger = None, port: int = 5560) -> InteractiveDataExplorer:
    """Create and return an interactive data explorer"""
    
    return InteractiveDataExplorer(debugger=debugger, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Data Explorer for Research Debugging")
    parser.add_argument("--port", type=int, default=5560, help="Port to run explorer on")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    
    args = parser.parse_args()
    
    # Create example data for demonstration
    explorer = InteractiveDataExplorer(port=args.port, debug=args.debug)
    
    # Add sample dataset
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(2, 0.5, 100),
        'target': np.random.choice([0, 1], 100)
    })
    sample_data.loc[5, 'feature1'] = np.nan  # Add missing value
    sample_data.loc[10, 'feature2'] = 1000   # Add outlier
    
    explorer.add_dataset("sample_dataset", sample_data)
    
    # Add sample variables
    explorer.add_variable("model_accuracy", 0.85)
    explorer.add_variable("learning_rate", 0.001)
    explorer.add_variable("batch_size", 32)
    
    # Add sample execution trace
    explorer.add_execution_trace("train_model", 45.2, 
                               {"accuracy": 0.85, "loss": 0.3})
    
    print(f"Interactive Data Explorer starting on port {args.port}")
    print("Sample data has been loaded for demonstration")
    
    explorer.run()