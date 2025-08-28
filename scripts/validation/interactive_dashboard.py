"""
Interactive Dashboard Generator
Advanced interactive dashboards for experiment validation results
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Interactive dashboard libraries
try:
    import dash
    from dash import dcc, html, Input, Output, callback
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

@dataclass
class InteractiveDashboard:
    """Interactive dashboard configuration"""
    dashboard_id: str
    experiment_name: str
    dashboard_url: Optional[str]
    components: List[str]
    update_frequency: str
    filters: Dict[str, Any]
    custom_widgets: List[Dict[str, Any]] = None


class InteractiveDashboardGenerator:
    """Generate interactive dashboards for validation results"""
    
    def __init__(self):
        self.logger = logging.getLogger("interactive_dashboard")
        self.app = None
        self.dashboard_port = 8050
    
    def create_dashboard(self, validation_results: Dict[str, Any],
                        experiment_name: str = "Unknown Experiment") -> InteractiveDashboard:
        """Create interactive dashboard for validation results"""
        
        if not DASH_AVAILABLE:
            self.logger.error("Dash not available - cannot create interactive dashboard")
            return None
        
        dashboard_id = f"dashboard_{experiment_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Create dashboard layout
        self.app.layout = self._create_dashboard_layout(validation_results, experiment_name)
        
        # Setup callbacks
        self._setup_callbacks(validation_results)
        
        dashboard_url = f"http://localhost:{self.dashboard_port}"
        
        return InteractiveDashboard(
            dashboard_id=dashboard_id,
            experiment_name=experiment_name,
            dashboard_url=dashboard_url,
            components=self._get_dashboard_components(validation_results),
            update_frequency="real_time",
            filters=self._generate_filter_options(validation_results)
        )
    
    def _create_dashboard_layout(self, validation_results: Dict[str, Any], 
                                experiment_name: str) -> html.Div:
        """Create the main dashboard layout"""
        
        return html.Div([
            # Header
            dbc.NavbarSimple(
                brand=f"Validation Dashboard - {experiment_name}",
                brand_href="#",
                color="primary",
                dark=True,
                className="mb-4"
            ),
            
            # Main content
            dbc.Container([
                # Control panel
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Dashboard Controls", className="card-title"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Time Range:"),
                                        dcc.Dropdown(
                                            id='time-range-dropdown',
                                            options=[
                                                {'label': 'Last Day', 'value': 'last_day'},
                                                {'label': 'Last Week', 'value': 'last_week'},
                                                {'label': 'Last Month', 'value': 'last_month'},
                                                {'label': 'All Time', 'value': 'all_time'}
                                            ],
                                            value='last_week'
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Validation Type:"),
                                        dcc.Dropdown(
                                            id='validation-type-dropdown',
                                            options=[
                                                {'label': 'All', 'value': 'all'},
                                                {'label': 'Statistical', 'value': 'statistical'},
                                                {'label': 'Code Quality', 'value': 'code_quality'},
                                                {'label': 'Scientific Rigor', 'value': 'rigor'}
                                            ],
                                            value='all'
                                        )
                                    ], width=6)
                                ])
                            ])
                        ])
                    ], width=12)
                ], className="mb-4"),
                
                # Summary cards
                dbc.Row([
                    dbc.Col([
                        self._create_summary_card("Overall Score", 
                                                 validation_results.get("overall_score", 0),
                                                 "primary",
                                                 "overall-score-card")
                    ], width=3),
                    dbc.Col([
                        self._create_summary_card("Statistical Validity", 
                                                 validation_results.get("statistical_score", 0),
                                                 "info",
                                                 "statistical-score-card")
                    ], width=3),
                    dbc.Col([
                        self._create_summary_card("Code Quality", 
                                                 validation_results.get("code_quality_score", 0),
                                                 "success",
                                                 "code-quality-score-card")
                    ], width=3),
                    dbc.Col([
                        self._create_summary_card("Scientific Rigor", 
                                                 validation_results.get("rigor_score", 0),
                                                 "warning",
                                                 "rigor-score-card")
                    ], width=3)
                ], className="mb-4"),
                
                # Main charts row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Validation Overview", className="card-title"),
                                dcc.Graph(id="validation-overview-chart")
                            ])
                        ])
                    ], width=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Score Distribution", className="card-title"),
                                dcc.Graph(id="score-distribution-chart")
                            ])
                        ])
                    ], width=4)
                ], className="mb-4"),
                
                # Time series and trends
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Validation Trends Over Time", className="card-title"),
                                dcc.Graph(id="time-series-chart")
                            ])
                        ])
                    ], width=12)
                ], className="mb-4"),
                
                # Detailed analysis tabs
                dbc.Card([
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="Statistical Analysis", tab_id="statistical"),
                            dbc.Tab(label="Code Quality", tab_id="code_quality"),
                            dbc.Tab(label="Scientific Rigor", tab_id="rigor"),
                            dbc.Tab(label="Bias Detection", tab_id="bias"),
                            dbc.Tab(label="Issues & Recommendations", tab_id="recommendations")
                        ], id="analysis-tabs", active_tab="statistical"),
                        
                        html.Div(id="tab-content", className="mt-4")
                    ])
                ]),
                
                # Real-time updates
                dcc.Interval(
                    id='interval-component',
                    interval=30*1000,  # Update every 30 seconds
                    n_intervals=0
                ),
                
                # Store component for data
                dcc.Store(id='validation-data-store', data=validation_results)
            ])
        ])
    
    def _create_summary_card(self, title: str, value: float, color: str, card_id: str) -> dbc.Card:
        """Create summary card component"""
        
        # Determine trend icon and color
        trend_icon = "📈" if value >= 75 else "📊" if value >= 50 else "📉"
        
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H2(f"{value:.1f}", className=f"text-{color}", style={"margin": "0"}),
                    html.P(title, className="card-text", style={"margin": "0", "fontSize": "14px"}),
                    html.Span(trend_icon, style={"fontSize": "20px"})
                ], style={"textAlign": "center"})
            ])
        ], color=color, outline=True, id=card_id)
    
    def _setup_callbacks(self, validation_results: Dict[str, Any]):
        """Setup dashboard callbacks for interactivity"""
        
        @self.app.callback(
            [Output('validation-overview-chart', 'figure'),
             Output('score-distribution-chart', 'figure'),
             Output('time-series-chart', 'figure')],
            [Input('time-range-dropdown', 'value'),
             Input('validation-type-dropdown', 'value'),
             Input('analysis-tabs', 'active_tab')]
        )
        def update_main_charts(time_range, validation_type, active_tab):
            overview_fig = self._create_overview_chart(validation_results, validation_type)
            distribution_fig = self._create_distribution_chart(validation_results)
            time_series_fig = self._create_time_series_chart(validation_results, time_range)
            
            return overview_fig, distribution_fig, time_series_fig
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('analysis-tabs', 'active_tab'),
             Input('validation-data-store', 'data')]
        )
        def render_tab_content(active_tab, validation_data):
            return self._render_tab_content(validation_data, active_tab)
        
        # Real-time updates callback
        @self.app.callback(
            [Output('overall-score-card', 'children'),
             Output('statistical-score-card', 'children'),
             Output('code-quality-score-card', 'children'),
             Output('rigor-score-card', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_summary_cards(n):
            # In real implementation, this would fetch updated data
            return (
                self._create_summary_card("Overall Score", validation_results.get("overall_score", 0), "primary", "overall-score-card").children,
                self._create_summary_card("Statistical Validity", validation_results.get("statistical_score", 0), "info", "statistical-score-card").children,
                self._create_summary_card("Code Quality", validation_results.get("code_quality_score", 0), "success", "code-quality-score-card").children,
                self._create_summary_card("Scientific Rigor", validation_results.get("rigor_score", 0), "warning", "rigor-score-card").children
            )
    
    def _create_overview_chart(self, validation_results: Dict[str, Any], validation_type: str) -> go.Figure:
        """Create overview chart based on validation type"""
        
        if validation_type == "statistical":
            return self._create_statistical_overview(validation_results)
        elif validation_type == "code_quality":
            return self._create_code_quality_overview(validation_results)
        elif validation_type == "rigor":
            return self._create_rigor_overview(validation_results)
        else:
            return self._create_general_overview(validation_results)
    
    def _create_general_overview(self, validation_results: Dict[str, Any]) -> go.Figure:
        """Create general overview chart"""
        
        categories = ['Statistical', 'Code Quality', 'Scientific Rigor', 'Overall']
        scores = [
            validation_results.get('statistical_score', 70),
            validation_results.get('code_quality_score', 75),
            validation_results.get('rigor_score', 65),
            validation_results.get('overall_score', 70)
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                marker_color=colors,
                text=[f"{s:.1f}" for s in scores],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Validation Scores Overview",
            xaxis_title="Category",
            yaxis_title="Score",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def _create_statistical_overview(self, validation_results: Dict[str, Any]) -> go.Figure:
        """Create statistical analysis overview chart"""
        
        categories = ['Consistency', 'Significance', 'Effect Size', 'Power']
        scores = [
            validation_results.get('consistency_score', 70),
            validation_results.get('significance_score', 65),
            validation_results.get('effect_size_score', 75),
            validation_results.get('power_score', 80)
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            line_color='rgb(1,90,120)',
            fillcolor='rgba(1,90,120,0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            title="Statistical Analysis Scores",
            showlegend=False
        )
        
        return fig
    
    def _create_distribution_chart(self, validation_results: Dict[str, Any]) -> go.Figure:
        """Create score distribution chart"""
        
        scores = [
            validation_results.get('statistical_score', 70),
            validation_results.get('code_quality_score', 75),
            validation_results.get('rigor_score', 65),
            validation_results.get('overall_score', 70)
        ]
        
        fig = go.Figure(data=[
            go.Histogram(
                x=scores,
                nbinsx=10,
                marker_color='rgba(1,90,120,0.7)',
                hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Score Distribution",
            xaxis_title="Score",
            yaxis_title="Frequency",
            template='plotly_white'
        )
        
        return fig
    
    def _create_time_series_chart(self, validation_results: Dict[str, Any], time_range: str) -> go.Figure:
        """Create time series chart for tracking validation over time"""
        
        # Generate sample time series data (in real implementation, this would come from historical data)
        if time_range == 'last_day':
            dates = pd.date_range(start=datetime.now().replace(hour=0), periods=24, freq='H')
        elif time_range == 'last_week':
            dates = pd.date_range(start=datetime.now() - pd.Timedelta(weeks=1), periods=7, freq='D')
        elif time_range == 'last_month':
            dates = pd.date_range(start=datetime.now() - pd.Timedelta(days=30), periods=30, freq='D')
        else:  # all_time
            dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        
        # Sample data with some variation
        base_score = validation_results.get('overall_score', 70)
        overall_scores = np.random.normal(base_score, 5, len(dates))
        statistical_scores = np.random.normal(base_score - 5, 7, len(dates))
        code_scores = np.random.normal(base_score + 3, 4, len(dates))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=overall_scores,
            mode='lines+markers',
            name='Overall Score',
            line=dict(color='rgb(1,90,120)')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=statistical_scores,
            mode='lines+markers',
            name='Statistical Score',
            line=dict(color='rgb(255,127,14)')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=code_scores,
            mode='lines+markers',
            name='Code Quality Score',
            line=dict(color='rgb(44,160,44)')
        ))
        
        fig.update_layout(
            title='Validation Scores Over Time',
            xaxis_title='Date',
            yaxis_title='Score',
            yaxis=dict(range=[0, 100]),
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _render_tab_content(self, validation_results: Dict[str, Any], active_tab: str) -> html.Div:
        """Render content for active tab"""
        
        if active_tab == "statistical":
            return self._create_statistical_details(validation_results)
        elif active_tab == "code_quality":
            return self._create_code_quality_details(validation_results)
        elif active_tab == "rigor":
            return self._create_rigor_details(validation_results)
        elif active_tab == "bias":
            return self._create_bias_details(validation_results)
        else:
            return self._create_recommendations_details(validation_results)
    
    def _create_statistical_details(self, validation_results: Dict[str, Any]) -> html.Div:
        """Create statistical analysis details tab"""
        
        stat_data = validation_results.get('statistical_validation', {})
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Consistency Analysis"),
                            html.P(f"Consistent metrics: {stat_data.get('consistent_metrics', 'N/A')}"),
                            html.P(f"Inconsistent metrics: {stat_data.get('inconsistent_metrics', 'N/A')}")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Significance Tests"),
                            html.P(f"Significant results: {stat_data.get('significant_results', 'N/A')}"),
                            html.P(f"Non-significant results: {stat_data.get('non_significant_results', 'N/A')}")
                        ])
                    ])
                ], width=6)
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H5("Effect Sizes"),
                    dcc.Graph(
                        figure=self._create_effect_size_plot(stat_data)
                    )
                ], width=12)
            ])
        ])
    
    def _create_effect_size_plot(self, stat_data: Dict[str, Any]) -> go.Figure:
        """Create effect size visualization"""
        
        # Sample effect size data
        effects = ['Effect 1', 'Effect 2', 'Effect 3', 'Effect 4']
        sizes = [0.2, 0.5, 0.8, 0.3]
        colors = ['small' if s < 0.3 else 'medium' if s < 0.7 else 'large' for s in sizes]
        color_map = {'small': 'lightblue', 'medium': 'orange', 'large': 'red'}
        
        fig = go.Figure(data=[
            go.Bar(
                x=effects,
                y=sizes,
                marker_color=[color_map[c] for c in colors],
                text=[f"{s:.2f}" for s in sizes],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Effect Sizes by Analysis",
            xaxis_title="Analysis",
            yaxis_title="Effect Size",
            template='plotly_white'
        )
        
        return fig
    
    def _create_code_quality_details(self, validation_results: Dict[str, Any]) -> html.Div:
        """Create code quality details tab"""
        
        code_data = validation_results.get('code_quality', {})
        
        return html.Div([
            dbc.Alert(
                "Code quality analysis details would be displayed here.",
                color="info"
            )
        ])
    
    def _create_rigor_details(self, validation_results: Dict[str, Any]) -> html.Div:
        """Create scientific rigor details tab"""
        
        rigor_data = validation_results.get('scientific_rigor', {})
        
        return html.Div([
            dbc.Alert(
                "Scientific rigor analysis details would be displayed here.",
                color="info"
            )
        ])
    
    def _create_bias_details(self, validation_results: Dict[str, Any]) -> html.Div:
        """Create bias detection details tab"""
        
        bias_data = validation_results.get('bias_detection', {})
        
        return html.Div([
            dbc.Alert(
                "Bias detection results would be displayed here.",
                color="warning"
            )
        ])
    
    def _create_recommendations_details(self, validation_results: Dict[str, Any]) -> html.Div:
        """Create recommendations details tab"""
        
        recommendations = validation_results.get('recommendations', [])
        
        return html.Div([
            html.H5("Recommendations"),
            html.Ul([
                html.Li(rec) for rec in recommendations[:10]  # Show first 10 recommendations
            ]) if recommendations else html.P("No specific recommendations at this time.")
        ])
    
    def launch_dashboard(self, debug: bool = False, port: int = None) -> str:
        """Launch the interactive dashboard"""
        
        if not self.app:
            raise ValueError("Dashboard not created. Call create_dashboard first.")
        
        if port:
            self.dashboard_port = port
        
        try:
            self.app.run_server(debug=debug, port=self.dashboard_port, host='0.0.0.0')
            return f"Dashboard launched at http://localhost:{self.dashboard_port}"
        except Exception as e:
            self.logger.error(f"Failed to launch dashboard: {e}")
            return f"Failed to launch dashboard: {e}"
    
    def _get_dashboard_components(self, validation_results: Dict[str, Any]) -> List[str]:
        """Get list of dashboard components"""
        
        components = ['summary_cards', 'overview_charts', 'time_series']
        
        if 'statistical_validation' in validation_results:
            components.append('statistical_analysis')
        
        if 'code_quality' in validation_results:
            components.append('code_quality_analysis')
        
        if 'scientific_rigor' in validation_results:
            components.append('scientific_rigor_analysis')
        
        if 'bias_detection' in validation_results:
            components.append('bias_detection')
        
        return components
    
    def _generate_filter_options(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate filter options for dashboard"""
        
        filters = {
            'time_range': ['last_day', 'last_week', 'last_month', 'all_time'],
            'score_threshold': [50, 60, 70, 80, 90],
            'validation_type': ['all', 'statistical', 'code_quality', 'scientific_rigor']
        }
        
        return filters