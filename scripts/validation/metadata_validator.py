"""
Metadata Validation System
Advanced metadata validation and analysis with graph-based relationship modeling
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Network analysis for metadata relationships
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .scientific_rigor_validator import MetadataValidation


class MetadataValidator:
    """Advanced metadata validation and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger("metadata_validator")
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> MetadataValidation:
        """Comprehensive metadata validation"""
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness(metadata)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency(metadata)
        
        # Identify quality issues
        quality_issues = self._identify_quality_issues(metadata)
        
        # Identify missing fields
        missing_fields = self._identify_missing_fields(metadata)
        
        # Identify inconsistent fields
        inconsistent_fields = self._identify_inconsistent_fields(metadata)
        
        # Generate recommendations
        recommendations = self._generate_metadata_recommendations(metadata, quality_issues, missing_fields)
        
        # Metadata graph analysis
        graph_analysis = None
        if NETWORKX_AVAILABLE:
            graph_analysis = self._analyze_metadata_relationships(metadata)
        
        return MetadataValidation(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            quality_issues=quality_issues,
            missing_fields=missing_fields,
            inconsistent_fields=inconsistent_fields,
            recommendations=recommendations,
            metadata_graph_analysis=graph_analysis
        )
    
    def _calculate_completeness(self, metadata: Dict[str, Any]) -> float:
        """Calculate metadata completeness score"""
        
        required_fields = [
            "title", "description", "authors", "creation_date", 
            "version", "data_sources", "methodology", "parameters"
        ]
        
        present_fields = 0
        for field in required_fields:
            if field in metadata and metadata[field] is not None:
                if isinstance(metadata[field], str) and len(metadata[field].strip()) > 0:
                    present_fields += 1
                elif not isinstance(metadata[field], str) and metadata[field]:
                    present_fields += 1
        
        return (present_fields / len(required_fields)) * 100
    
    def _calculate_consistency(self, metadata: Dict[str, Any]) -> float:
        """Calculate metadata consistency score"""
        
        consistency_checks = 0
        total_checks = 0
        
        # Check date consistency
        if "creation_date" in metadata and "modification_date" in metadata:
            total_checks += 1
            try:
                creation = pd.to_datetime(metadata["creation_date"])
                modification = pd.to_datetime(metadata["modification_date"])
                if creation <= modification:
                    consistency_checks += 1
            except:
                pass
        
        # Check version consistency
        if "version" in metadata and "changelog" in metadata:
            total_checks += 1
            version = str(metadata["version"])
            changelog = str(metadata.get("changelog", ""))
            if version in changelog:
                consistency_checks += 1
        
        # Check author consistency
        if "authors" in metadata and "contributors" in metadata:
            total_checks += 1
            authors = set(str(a).lower() for a in metadata["authors"])
            contributors = set(str(c).lower() for c in metadata["contributors"])
            if authors.issubset(contributors) or contributors.issubset(authors):
                consistency_checks += 1
        
        return (consistency_checks / total_checks * 100) if total_checks > 0 else 100
    
    def _identify_quality_issues(self, metadata: Dict[str, Any]) -> List[str]:
        """Identify metadata quality issues"""
        
        issues = []
        
        # Check for placeholder values
        placeholders = ["TODO", "TBD", "placeholder", "example", "test", "temp"]
        for key, value in metadata.items():
            if isinstance(value, str):
                for placeholder in placeholders:
                    if placeholder.lower() in value.lower():
                        issues.append(f"Placeholder value detected in {key}: {value}")
        
        # Check for extremely short descriptions
        if "description" in metadata:
            desc = str(metadata["description"])
            if len(desc) < 50:
                issues.append(f"Description too short ({len(desc)} characters)")
        
        # Check for missing version information
        if "version" in metadata:
            version = str(metadata["version"])
            if not re.match(r"\d+\.\d+", version):
                issues.append(f"Version format not standard: {version}")
        
        # Check for outdated information
        if "creation_date" in metadata:
            try:
                creation_date = pd.to_datetime(metadata["creation_date"])
                days_old = (datetime.now() - creation_date).days
                if days_old > 365:
                    issues.append(f"Metadata is {days_old} days old - consider updating")
            except:
                issues.append("Invalid creation_date format")
        
        # Check for suspicious email addresses
        if "contact_email" in metadata:
            email = str(metadata["contact_email"])
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                issues.append(f"Invalid email format: {email}")
            elif any(domain in email.lower() for domain in ["example.com", "test.com", "fake.com"]):
                issues.append(f"Suspicious email domain: {email}")
        
        # Check for URL validity
        url_fields = ["repository_url", "documentation_url", "homepage"]
        for field in url_fields:
            if field in metadata:
                url = str(metadata[field])
                if not (url.startswith("http://") or url.startswith("https://")):
                    issues.append(f"Invalid URL format in {field}: {url}")
        
        # Check for license compliance
        if "license" in metadata:
            license_text = str(metadata["license"]).lower()
            recognized_licenses = ["mit", "apache", "gpl", "bsd", "creative commons", "proprietary"]
            if not any(lic in license_text for lic in recognized_licenses):
                issues.append("Unrecognized license type - consider using standard license")
        
        return issues
    
    def _identify_missing_fields(self, metadata: Dict[str, Any]) -> List[str]:
        """Identify missing metadata fields"""
        
        critical_fields = [
            "title", "description", "authors", "creation_date", "version"
        ]
        
        recommended_fields = [
            "keywords", "license", "data_sources", "methodology", 
            "parameters", "dependencies", "changelog", "contact_info"
        ]
        
        missing = []
        
        for field in critical_fields:
            if field not in metadata or not metadata[field]:
                missing.append(f"{field} (critical)")
        
        for field in recommended_fields:
            if field not in metadata or not metadata[field]:
                missing.append(f"{field} (recommended)")
        
        return missing
    
    def _identify_inconsistent_fields(self, metadata: Dict[str, Any]) -> List[str]:
        """Identify inconsistent metadata fields"""
        
        inconsistent = []
        
        # Check for data type inconsistencies
        expected_types = {
            "authors": list,
            "keywords": list,
            "parameters": dict,
            "creation_date": str,
            "version": str
        }
        
        for field, expected_type in expected_types.items():
            if field in metadata and not isinstance(metadata[field], expected_type):
                actual_type = type(metadata[field]).__name__
                inconsistent.append(f"{field}: expected {expected_type.__name__}, got {actual_type}")
        
        # Check for value range inconsistencies
        if "version" in metadata:
            version = str(metadata["version"])
            # Check for negative version numbers
            if re.search(r"-\d+", version):
                inconsistent.append("Version contains negative numbers")
        
        # Check for author name formatting
        if "authors" in metadata and isinstance(metadata["authors"], list):
            for i, author in enumerate(metadata["authors"]):
                author_str = str(author)
                if len(author_str) < 2:
                    inconsistent.append(f"Author {i+1} name too short: '{author_str}'")
                elif not re.match(r"[A-Za-z\s\-'\.]+", author_str):
                    inconsistent.append(f"Author {i+1} contains unusual characters: '{author_str}'")
        
        return inconsistent
    
    def _generate_metadata_recommendations(self, metadata: Dict[str, Any], 
                                         quality_issues: List[str],
                                         missing_fields: List[str]) -> List[str]:
        """Generate metadata improvement recommendations"""
        
        recommendations = []
        
        if len(quality_issues) > 0:
            recommendations.append(f"Address {len(quality_issues)} quality issues identified")
        
        critical_missing = [field for field in missing_fields if "critical" in field]
        if len(critical_missing) > 0:
            recommendations.append(f"Add {len(critical_missing)} missing critical fields")
        
        if len(missing_fields) > 5:
            recommendations.append("Add missing recommended fields for better discoverability")
        
        if "keywords" not in metadata:
            recommendations.append("Add keywords for better discoverability")
        
        if "license" not in metadata:
            recommendations.append("Specify license information for legal clarity")
        
        if "changelog" not in metadata:
            recommendations.append("Maintain a changelog for version tracking")
        
        if "data_sources" not in metadata:
            recommendations.append("Document data sources for reproducibility")
        
        if "methodology" not in metadata:
            recommendations.append("Document methodology for scientific rigor")
        
        # Advanced recommendations based on content analysis
        if "description" in metadata:
            desc = str(metadata["description"])
            if len(desc.split()) < 20:
                recommendations.append("Expand description to include more detail")
            if "experimental" not in desc.lower() and "method" not in desc.lower():
                recommendations.append("Include experimental methodology in description")
        
        # Versioning recommendations
        if "version" in metadata:
            version = str(metadata["version"])
            if version in ["1.0", "0.1", "1"]:
                recommendations.append("Consider using semantic versioning (e.g., 1.0.0)")
        
        return recommendations
    
    def _analyze_metadata_relationships(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships in metadata using graph analysis"""
        
        try:
            G = nx.Graph()
            
            # Add nodes for metadata fields
            for key, value in metadata.items():
                G.add_node(key, value=str(value), field_type=type(value).__name__)
            
            # Add edges based on relationships
            self._add_metadata_edges(G, metadata)
            
            # Calculate graph metrics
            analysis = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "connected_components": nx.number_connected_components(G)
            }
            
            # Find central nodes (most connected metadata fields)
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                analysis["most_central_fields"] = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Identify isolated fields
            isolated_nodes = list(nx.isolates(G))
            analysis["isolated_fields"] = isolated_nodes
            
            # Calculate clustering coefficient
            if G.number_of_nodes() > 2:
                analysis["clustering_coefficient"] = nx.average_clustering(G)
            
            # Identify field importance based on connections
            analysis["field_importance"] = self._calculate_field_importance(G, metadata)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Metadata graph analysis failed: {e}")
            return {"error": str(e)}
    
    def _add_metadata_edges(self, G: 'nx.Graph', metadata: Dict[str, Any]):
        """Add edges to metadata graph based on semantic relationships"""
        
        # Authors-Contributors relationship
        if "authors" in metadata and "contributors" in metadata:
            G.add_edge("authors", "contributors", relationship="collaboration", weight=0.8)
        
        # Version-Changelog relationship
        if "version" in metadata and "changelog" in metadata:
            G.add_edge("version", "changelog", relationship="versioning", weight=0.9)
        
        # Data_sources-Methodology relationship
        if "data_sources" in metadata and "methodology" in metadata:
            G.add_edge("data_sources", "methodology", relationship="methodology", weight=0.7)
        
        # Title-Description relationship
        if "title" in metadata and "description" in metadata:
            G.add_edge("title", "description", relationship="content", weight=0.6)
        
        # Keywords-Title relationship
        if "keywords" in metadata and "title" in metadata:
            G.add_edge("keywords", "title", relationship="content", weight=0.5)
        
        # Parameters-Methodology relationship
        if "parameters" in metadata and "methodology" in metadata:
            G.add_edge("parameters", "methodology", relationship="configuration", weight=0.8)
        
        # License-Authors relationship
        if "license" in metadata and "authors" in metadata:
            G.add_edge("license", "authors", relationship="legal", weight=0.4)
        
        # Creation_date-Modification_date relationship
        if "creation_date" in metadata and "modification_date" in metadata:
            G.add_edge("creation_date", "modification_date", relationship="temporal", weight=0.9)
        
        # Contact_info-Authors relationship
        if "contact_info" in metadata and "authors" in metadata:
            G.add_edge("contact_info", "authors", relationship="communication", weight=0.6)
    
    def _calculate_field_importance(self, G: 'nx.Graph', metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate importance scores for metadata fields"""
        
        importance_scores = {}
        
        if G.number_of_nodes() == 0:
            return importance_scores
        
        # Base importance on degree centrality
        centrality = nx.degree_centrality(G)
        
        # Add field-specific importance weights
        field_weights = {
            "title": 1.0,
            "description": 0.9,
            "authors": 0.8,
            "methodology": 0.8,
            "data_sources": 0.7,
            "version": 0.6,
            "parameters": 0.6,
            "creation_date": 0.5,
            "keywords": 0.5,
            "license": 0.4
        }
        
        for field in metadata:
            base_score = centrality.get(field, 0)
            weight = field_weights.get(field, 0.3)
            importance_scores[field] = base_score * weight
        
        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        if max_score > 0:
            importance_scores = {field: score / max_score for field, score in importance_scores.items()}
        
        return importance_scores