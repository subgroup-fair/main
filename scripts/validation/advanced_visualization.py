"""
Advanced Visualization Engine
Comprehensive visualization suite with multiple chart libraries and export options
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import base64
import io

# Plotting and visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    from plotly.colors import qualitative
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Additional visualization libraries
try:
    import altair as alt
    import seaborn as sns
    import matplotlib.pyplot as plt
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False

# Network visualization
try:
    import networkx as nx
    import plotly.graph_objects as go
    NETWORK_VIZ_AVAILABLE = True
except ImportError:
    NETWORK_VIZ_AVAILABLE = False


@dataclass
class VisualizationConfig:
    """Advanced visualization configuration"""
    chart_type: str
    data_source: str
    interactive: bool
    responsive: bool
    theme: str
    color_scheme: List[str]
    annotations: List[Dict[str, Any]] = None
    animations: Dict[str, Any] = None


class AdvancedVisualizationEngine:
    """Advanced visualization engine with multiple chart libraries"""
    
    def __init__(self):
        self.logger = logging.getLogger("advanced_visualization")
        self.color_schemes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'scientific': ['#440154', '#31688e', '#26828e', '#1f9e89', '#6ece58'],
            'quality': ['#b2182b', '#d6604d', '#f4a582', '#92c5de', '#4393c3'],
            'performance': ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7'],
            'viridis': ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'],
            'plasma': ['#0c0786', '#5c01a6', '#900da4', '#bf3984', '#e16462', '#f89441', '#fcce25']
        }
    
    def create_comprehensive_visualization_suite(self, validation_results: Dict[str, Any],
                                               config: VisualizationConfig = None) -> Dict[str, Any]:
        """Create comprehensive visualization suite"""
        
        if not PLOTLY_AVAILABLE:
            self.logger.error("Plotly not available - cannot create advanced visualizations")
            return {"error": "Plotly not available"}
        
        if config is None:
            config = VisualizationConfig(
                chart_type="comprehensive",
                data_source="validation_results",
                interactive=True,
                responsive=True,
                theme="scientific",
                color_scheme=self.color_schemes['scientific']
            )
        
        visualizations = {}
        
        try:
            # Core validation charts
            visualizations['validation_overview'] = self._create_validation_overview_viz(validation_results, config)
            visualizations['score_distribution'] = self._create_score_distribution_viz(validation_results, config)
            visualizations['trend_analysis'] = self._create_trend_analysis_viz(validation_results, config)
            visualizations['comparison_matrix'] = self._create_comparison_matrix_viz(validation_results, config)
            
            # Statistical visualizations
            if 'statistical_validation' in validation_results:
                visualizations['statistical_heatmap'] = self._create_statistical_heatmap(validation_results, config)
                visualizations['effect_size_plot'] = self._create_effect_size_plot(validation_results, config)
                visualizations['power_analysis_viz'] = self._create_power_analysis_viz(validation_results, config)
            
            # Code quality visualizations
            if 'code_quality' in validation_results:
                visualizations['code_quality_radar'] = self._create_code_quality_radar(validation_results, config)
                visualizations['complexity_analysis'] = self._create_complexity_analysis(validation_results, config)
                visualizations['security_assessment'] = self._create_security_assessment_viz(validation_results, config)
            
            # Scientific rigor visualizations
            if 'scientific_rigor' in validation_results:
                visualizations['rigor_assessment'] = self._create_rigor_assessment_viz(validation_results, config)
                visualizations['methodology_compliance'] = self._create_methodology_compliance_viz(validation_results, config)
            
            # Bias detection visualizations
            if 'bias_detection' in validation_results or 'advanced_bias_detection' in validation_results:
                visualizations['bias_network'] = self._create_bias_network_viz(validation_results, config)
                visualizations['fairness_metrics'] = self._create_fairness_metrics_viz(validation_results, config)
            
            # Interactive composite views
            visualizations['executive_summary'] = self._create_executive_summary_viz(validation_results, config)
            visualizations['drill_down_analysis'] = self._create_drill_down_analysis_viz(validation_results, config)
            
        except Exception as e:
            self.logger.error(f"Error creating visualization suite: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _create_validation_overview_viz(self, validation_results: Dict[str, Any], 
                                       config: VisualizationConfig) -> Dict[str, Any]:
        """Create validation overview visualization"""
        
        try:
            # Extract key metrics
            metrics = self._extract_key_metrics(validation_results)
            
            # Create subplot with different chart types
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Overall Scores', 'Category Breakdown', 'Trend Indicators', 'Quality Gates'),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "indicator"}]]
            )
            
            # Bar chart - overall scores
            categories = list(metrics['scores'].keys())
            values = list(metrics['scores'].values())
            
            fig.add_trace(
                go.Bar(x=categories, y=values, name="Scores", marker_color=config.color_scheme),
                row=1, col=1
            )
            
            # Pie chart - category breakdown
            fig.add_trace(
                go.Pie(labels=categories, values=values, name="Distribution"),
                row=1, col=2
            )
            
            # Scatter plot - trend indicators
            if 'trends' in metrics:
                fig.add_trace(
                    go.Scatter(x=metrics['trends']['dates'], y=metrics['trends']['values'], 
                              mode='lines+markers', name="Trend"),
                    row=2, col=1
                )
            
            # Gauge chart - quality gate
            overall_score = metrics.get('overall_score', 70)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Quality"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen" if overall_score >= 80 else "orange" if overall_score >= 60 else "red"},
                        'steps': [{'range': [0, 60], 'color': "lightgray"},
                                 {'range': [60, 80], 'color': "gray"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Validation Overview Dashboard",
                showlegend=False,
                template='plotly_white'
            )
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'config': {'responsive': True, 'displayModeBar': True}
            }
            
        except Exception as e:
            self.logger.error(f"Error creating validation overview: {e}")
            return {'error': str(e)}
    
    def _create_statistical_heatmap(self, validation_results: Dict[str, Any],
                                   config: VisualizationConfig) -> Dict[str, Any]:
        """Create statistical analysis heatmap"""
        
        try:
            stat_data = validation_results.get('statistical_validation', {})
            
            # Create correlation matrix for statistical metrics
            metrics = ['consistency', 'significance', 'effect_size', 'power']
            
            # Generate sample correlation data (in real implementation, this would be calculated from actual data)
            correlation_matrix = np.random.rand(len(metrics), len(metrics))
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
            np.fill_diagonal(correlation_matrix, 1)  # Perfect correlation with self
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=metrics,
                y=metrics,
                colorscale='RdBu',
                zmid=0,
                hoverongaps=False,
                hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Correlation")
            ))\n            
            # Add text annotations
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    fig.add_annotation(
                        x=metrics[j],
                        y=metrics[i],
                        text=f"{correlation_matrix[i][j]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(correlation_matrix[i][j]) > 0.5 else "black")
                    )
            
            fig.update_layout(
                title='Statistical Metrics Correlation Matrix',
                template='plotly_white',
                xaxis_title="Metric",
                yaxis_title="Metric"
            )
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'config': {'responsive': True}
            }
            
        except Exception as e:
            self.logger.error(f"Error creating statistical heatmap: {e}")
            return {'error': str(e)}
    
    def _create_code_quality_radar(self, validation_results: Dict[str, Any],
                                  config: VisualizationConfig) -> Dict[str, Any]:
        """Create code quality radar chart"""
        
        try:
            code_data = validation_results.get('code_quality', {})
            
            # Define quality dimensions
            dimensions = ['Maintainability', 'Readability', 'Performance', 'Security', 'Testing', 'Documentation']
            
            # Extract scores (with defaults)
            scores = [
                code_data.get('maintainability_score', 70),
                code_data.get('readability_score', 75),
                code_data.get('performance_score', 80),
                code_data.get('security_score', 85),
                code_data.get('testing_score', 60),
                code_data.get('documentation_score', 70)
            ]
            
            # Create radar chart
            fig = go.Figure()
            
            # Add current scores
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=dimensions,
                fill='toself',
                name='Current Quality',
                line_color=config.color_scheme[0],
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(config.color_scheme[0])) + [0.3])}"
            ))
            
            # Add target benchmark
            target_scores = [85] * len(dimensions)
            fig.add_trace(go.Scatterpolar(
                r=target_scores,
                theta=dimensions,
                fill='toself',
                name='Target Benchmark',
                line_color=config.color_scheme[1],
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(config.color_scheme[1])) + [0.1])}",
                line_dash='dash'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Code Quality Assessment Radar",
                showlegend=True,
                template='plotly_white'
            )
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'config': {'responsive': True}
            }
            
        except Exception as e:
            self.logger.error(f"Error creating code quality radar: {e}")
            return {'error': str(e)}
    
    def _create_bias_network_viz(self, validation_results: Dict[str, Any],
                                config: VisualizationConfig) -> Dict[str, Any]:
        """Create bias detection network visualization"""
        
        try:
            if not NETWORK_VIZ_AVAILABLE:
                return {'error': 'Network visualization libraries not available'}
            
            bias_data = validation_results.get('bias_detection', {})
            advanced_bias_data = validation_results.get('advanced_bias_detection', {})
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes for different bias types
            bias_types = ['Selection', 'Confirmation', 'Algorithmic', 'Measurement', 'Cognitive']
            for bias_type in bias_types:
                severity = np.random.choice(['low', 'medium', 'high'])  # Sample data
                G.add_node(bias_type, severity=severity)
            
            # Add edges representing bias relationships
            edges = [('Selection', 'Confirmation'), ('Algorithmic', 'Measurement'), 
                    ('Cognitive', 'Confirmation'), ('Selection', 'Measurement')]
            G.add_edges_from(edges)
            
            # Get node positions
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Extract edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create edge trace
            edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                  line=dict(width=2, color='#888'),
                                  hoverinfo='none',
                                  mode='lines')
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            
            color_map = {'low': 'green', 'medium': 'orange', 'high': 'red'}
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                severity = G.nodes[node]['severity']
                node_colors.append(color_map[severity])
            
            node_trace = go.Scatter(x=node_x, y=node_y,
                                  mode='markers+text',
                                  hoverinfo='text',
                                  text=node_text,
                                  textposition="middle center",
                                  marker=dict(size=30,
                                            color=node_colors,
                                            line=dict(width=2, color='black')))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title='Bias Detection Network',
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Node colors: Green=Low, Orange=Medium, Red=High severity",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor="left", yanchor="bottom",
                                  font=dict(color="black", size=10)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'config': {'responsive': True}
            }
            
        except Exception as e:
            self.logger.error(f"Error creating bias network visualization: {e}")
            return {'error': str(e)}
    
    def _create_executive_summary_viz(self, validation_results: Dict[str, Any],
                                     config: VisualizationConfig) -> Dict[str, Any]:
        """Create executive summary visualization"""
        
        try:
            # Create a comprehensive dashboard-style visualization
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=('Overall Score', 'Quality Metrics', 'Trend Analysis',
                              'Risk Assessment', 'Recommendations', 'Compliance Status',
                              'Performance Indicators', 'Action Items', 'Summary'),
                specs=[[{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
                       [{"type": "pie"}, {"type": "table"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "table"}, {"type": "indicator"}]]
            )
            
            # Overall score indicator
            overall_score = validation_results.get('overall_score', 70)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    title={'text': "Overall Score"},
                    gauge={'axis': {'range': [None, 100]},
                          'bar': {'color': "darkgreen" if overall_score >= 80 else "orange"},
                          'steps': [{'range': [0, 60], 'color': "lightgray"},
                                   {'range': [60, 80], 'color': "gray"}]}
                ),
                row=1, col=1
            )
            
            # Quality metrics bar chart
            metrics = self._extract_key_metrics(validation_results)
            fig.add_trace(
                go.Bar(x=list(metrics['scores'].keys()), y=list(metrics['scores'].values()),
                      marker_color=config.color_scheme),
                row=1, col=2
            )
            
            # Add more components...
            # (Additional subplots would be added here for a complete executive summary)
            
            fig.update_layout(
                title_text="Executive Summary Dashboard",
                showlegend=False,
                template='plotly_white',
                height=800
            )
            
            return {
                'type': 'plotly',
                'figure': fig.to_dict(),
                'config': {'responsive': True, 'displayModeBar': True}
            }
            
        except Exception as e:
            self.logger.error(f"Error creating executive summary: {e}")
            return {'error': str(e)}
    
    def _extract_key_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from validation results"""
        
        metrics = {
            'scores': {
                'Statistical': validation_results.get('statistical_score', 70),
                'Code Quality': validation_results.get('code_quality_score', 75),
                'Scientific Rigor': validation_results.get('rigor_score', 65),
                'Overall': validation_results.get('overall_score', 70)
            },
            'overall_score': validation_results.get('overall_score', 70),
            'trends': {
                'dates': pd.date_range(start='2024-01-01', periods=30, freq='D'),
                'values': np.random.normal(70, 5, 30)  # Sample trend data
            }
        }
        
        return metrics
    
    def export_visualizations(self, visualizations: Dict[str, Any], 
                            output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """Export visualizations in multiple formats"""
        
        if formats is None:
            formats = ['html', 'png', 'svg', 'pdf']
        
        output_files = {}
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        for viz_name, viz_data in visualizations.items():
            if 'figure' in viz_data and viz_data['figure']:
                try:
                    fig = go.Figure(viz_data['figure'])
                    
                    for fmt in formats:
                        try:
                            if fmt == 'html':
                                file_path = output_path / f"{viz_name}.html"
                                fig.write_html(str(file_path))
                            elif fmt in ['png', 'svg', 'pdf']:
                                file_path = output_path / f"{viz_name}.{fmt}"
                                fig.write_image(str(file_path), format=fmt, width=1200, height=800)
                            
                            output_files[f"{viz_name}_{fmt}"] = str(file_path)
                            
                        except Exception as e:
                            self.logger.error(f"Error exporting {viz_name} as {fmt}: {e}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing visualization {viz_name}: {e}")
        
        self.logger.info(f"Exported {len(output_files)} visualization files to {output_dir}")
        return output_files
    
    # Additional helper methods for specific chart types
    def _create_score_distribution_viz(self, validation_results: Dict[str, Any], 
                                      config: VisualizationConfig) -> Dict[str, Any]:
        """Create score distribution visualization"""
        
        try:
            scores = list(self._extract_key_metrics(validation_results)['scores'].values())
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=scores, nbinsx=10, name="Score Distribution"))
            fig.update_layout(
                title="Validation Score Distribution",
                xaxis_title="Score",
                yaxis_title="Frequency",
                template='plotly_white'
            )
            
            return {'type': 'plotly', 'figure': fig.to_dict(), 'config': {'responsive': True}}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_trend_analysis_viz(self, validation_results: Dict[str, Any], 
                                  config: VisualizationConfig) -> Dict[str, Any]:
        """Create trend analysis visualization"""
        
        try:
            metrics = self._extract_key_metrics(validation_results)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics['trends']['dates'],
                y=metrics['trends']['values'],
                mode='lines+markers',
                name='Validation Score Trend'
            ))
            
            fig.update_layout(
                title="Validation Score Trends",
                xaxis_title="Date",
                yaxis_title="Score",
                template='plotly_white'
            )
            
            return {'type': 'plotly', 'figure': fig.to_dict(), 'config': {'responsive': True}}
            
        except Exception as e:
            return {'error': str(e)}
    
    # Placeholder methods for additional chart types
    def _create_comparison_matrix_viz(self, validation_results: Dict[str, Any], 
                                     config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Comparison matrix visualization'}
    
    def _create_effect_size_plot(self, validation_results: Dict[str, Any], 
                                config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Effect size plot visualization'}
    
    def _create_power_analysis_viz(self, validation_results: Dict[str, Any], 
                                  config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Power analysis visualization'}
    
    def _create_complexity_analysis(self, validation_results: Dict[str, Any], 
                                   config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Complexity analysis visualization'}
    
    def _create_security_assessment_viz(self, validation_results: Dict[str, Any], 
                                       config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Security assessment visualization'}
    
    def _create_rigor_assessment_viz(self, validation_results: Dict[str, Any], 
                                    config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Rigor assessment visualization'}
    
    def _create_methodology_compliance_viz(self, validation_results: Dict[str, Any], 
                                          config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Methodology compliance visualization'}
    
    def _create_fairness_metrics_viz(self, validation_results: Dict[str, Any], 
                                    config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Fairness metrics visualization'}
    
    def _create_drill_down_analysis_viz(self, validation_results: Dict[str, Any], 
                                       config: VisualizationConfig) -> Dict[str, Any]:
        return {'type': 'placeholder', 'message': 'Drill-down analysis visualization'}