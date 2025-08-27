"""
Real-time Resource Monitoring System
Monitors CPU, memory, disk usage, and GPU if available
"""

import psutil
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path
import logging
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

@dataclass
class ResourceSnapshot:
    """Single point-in-time resource usage snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_io: Dict[str, int]
    gpu_usage: Optional[List[Dict[str, Any]]] = None
    process_count: int = 0
    load_average: Optional[List[float]] = None

class ResourceMonitor:
    """
    Real-time resource monitoring with alerting and logging
    """
    
    def __init__(self,
                 monitoring_interval: float = 1.0,
                 max_history_points: int = 3600,  # 1 hour at 1s intervals
                 enable_gpu_monitoring: bool = True,
                 enable_network_monitoring: bool = True,
                 alert_thresholds: Dict[str, float] = None):
        
        self.monitoring_interval = monitoring_interval
        self.max_history_points = max_history_points
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_AVAILABLE
        self.enable_network_monitoring = enable_network_monitoring
        
        # Default alert thresholds
        if alert_thresholds is None:
            self.alert_thresholds = {
                'cpu_percent': 90.0,
                'memory_percent': 90.0,
                'disk_usage_percent': 95.0,
                'gpu_memory_percent': 95.0
            }
        else:
            self.alert_thresholds = alert_thresholds
        
        # State
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history: List[ResourceSnapshot] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # Logging
        self.logger = logging.getLogger("resource_monitor")
        
        # Baseline measurements for network I/O
        self.baseline_network_io = None
        
        self.logger.info("ResourceMonitor initialized")
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        
        if self.monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Get baseline network I/O
        if self.enable_network_monitoring:
            self.baseline_network_io = psutil.net_io_counters()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        
        if not self.monitoring:
            return
        
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        
        while self.monitoring:
            try:
                snapshot = self._capture_resource_snapshot()
                
                # Add to history
                self.resource_history.append(snapshot)
                
                # Maintain history size
                if len(self.resource_history) > self.max_history_points:
                    self.resource_history.pop(0)
                
                # Check for alerts
                self._check_resource_alerts(snapshot)
                
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _capture_resource_snapshot(self) -> ResourceSnapshot:
        """Capture current resource usage snapshot"""
        
        current_time = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_gb = memory.used / (1024**3)
        
        # Disk usage (for current working directory)
        disk = psutil.disk_usage('.')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        # Network I/O (delta from baseline)
        network_io = {}
        if self.enable_network_monitoring and self.baseline_network_io:
            current_net = psutil.net_io_counters()
            network_io = {
                'bytes_sent': current_net.bytes_sent - self.baseline_network_io.bytes_sent,
                'bytes_recv': current_net.bytes_recv - self.baseline_network_io.bytes_recv,
                'packets_sent': current_net.packets_sent - self.baseline_network_io.packets_sent,
                'packets_recv': current_net.packets_recv - self.baseline_network_io.packets_recv
            }
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix/Linux only)
        load_average = None
        if hasattr(psutil, 'getloadavg'):
            try:
                load_average = list(psutil.getloadavg())
            except:
                pass
        
        # GPU usage
        gpu_usage = None
        if self.enable_gpu_monitoring:
            gpu_usage = self._get_gpu_usage()
        
        return ResourceSnapshot(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_gb=memory_gb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_io=network_io,
            gpu_usage=gpu_usage,
            process_count=process_count,
            load_average=load_average
        )
    
    def _get_gpu_usage(self) -> Optional[List[Dict[str, Any]]]:
        """Get GPU usage information"""
        
        if not GPU_AVAILABLE:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,  # Convert to percentage
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                })
            
            return gpu_info
        
        except Exception as e:
            self.logger.debug(f"GPU monitoring error: {e}")
            return None
    
    def _check_resource_alerts(self, snapshot: ResourceSnapshot):
        """Check if any resource usage exceeds alert thresholds"""
        
        alerts = []
        
        # CPU alert
        if snapshot.cpu_percent > self.alert_thresholds.get('cpu_percent', 90):
            alerts.append({
                'type': 'cpu_high',
                'message': f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                'value': snapshot.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent'],
                'timestamp': snapshot.timestamp
            })
        
        # Memory alert
        if snapshot.memory_percent > self.alert_thresholds.get('memory_percent', 90):
            alerts.append({
                'type': 'memory_high',
                'message': f"High memory usage: {snapshot.memory_percent:.1f}% ({snapshot.memory_gb:.1f} GB)",
                'value': snapshot.memory_percent,
                'threshold': self.alert_thresholds['memory_percent'],
                'timestamp': snapshot.timestamp
            })
        
        # Disk alert
        if snapshot.disk_usage_percent > self.alert_thresholds.get('disk_usage_percent', 95):
            alerts.append({
                'type': 'disk_high',
                'message': f"High disk usage: {snapshot.disk_usage_percent:.1f}% ({snapshot.disk_free_gb:.1f} GB free)",
                'value': snapshot.disk_usage_percent,
                'threshold': self.alert_thresholds['disk_usage_percent'],
                'timestamp': snapshot.timestamp
            })
        
        # GPU alerts
        if snapshot.gpu_usage:
            for gpu in snapshot.gpu_usage:
                if gpu['memory_percent'] > self.alert_thresholds.get('gpu_memory_percent', 95):
                    alerts.append({
                        'type': 'gpu_memory_high',
                        'message': f"High GPU memory usage on GPU {gpu['id']}: {gpu['memory_percent']:.1f}%",
                        'value': gpu['memory_percent'],
                        'threshold': self.alert_thresholds['gpu_memory_percent'],
                        'timestamp': snapshot.timestamp
                    })
        
        # Log and store alerts
        for alert in alerts:
            self.alerts.append(alert)
            self.logger.warning(alert['message'])
            
            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]
    
    def get_current_usage(self) -> Optional[ResourceSnapshot]:
        """Get most recent resource usage snapshot"""
        
        if not self.resource_history:
            return None
        
        return self.resource_history[-1]
    
    def get_usage_history(self, minutes: int = 10) -> List[ResourceSnapshot]:
        """Get resource usage history for specified time period"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [snapshot for snapshot in self.resource_history 
                if snapshot.timestamp >= cutoff_time]
    
    def get_summary(self, minutes: int = None) -> Dict[str, Any]:
        """Get resource usage summary statistics"""
        
        if not self.resource_history:
            return {'message': 'No monitoring data available'}
        
        # Get data for specified time period
        if minutes:
            data = self.get_usage_history(minutes)
        else:
            data = self.resource_history
        
        if not data:
            return {'message': 'No data for specified time period'}
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in data]
        memory_values = [s.memory_percent for s in data]
        memory_gb_values = [s.memory_gb for s in data]
        disk_values = [s.disk_usage_percent for s in data]
        
        summary = {
            'monitoring_period': {
                'start_time': data[0].timestamp.isoformat(),
                'end_time': data[-1].timestamp.isoformat(),
                'duration_minutes': (data[-1].timestamp - data[0].timestamp).total_seconds() / 60,
                'data_points': len(data)
            },
            'cpu_usage': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory_usage': {
                'mean_percent': np.mean(memory_values),
                'max_percent': np.max(memory_values),
                'mean_gb': np.mean(memory_gb_values),
                'max_gb': np.max(memory_gb_values),
                'std_percent': np.std(memory_values)
            },
            'disk_usage': {
                'mean': np.mean(disk_values),
                'max': np.max(disk_values),
                'current_free_gb': data[-1].disk_free_gb
            },
            'alerts': {
                'total_alerts': len(self.alerts),
                'recent_alerts': [alert for alert in self.alerts 
                                if alert['timestamp'] >= data[0].timestamp]
            }
        }
        
        # Add GPU summary if available
        if any(s.gpu_usage for s in data if s.gpu_usage):
            gpu_data = [s.gpu_usage for s in data if s.gpu_usage]
            if gpu_data:
                # Flatten GPU data
                all_gpu_loads = []
                all_gpu_memory = []
                
                for gpu_snapshot in gpu_data:
                    for gpu in gpu_snapshot:
                        all_gpu_loads.append(gpu['load'])
                        all_gpu_memory.append(gpu['memory_percent'])
                
                if all_gpu_loads:
                    summary['gpu_usage'] = {
                        'mean_load': np.mean(all_gpu_loads),
                        'max_load': np.max(all_gpu_loads),
                        'mean_memory': np.mean(all_gpu_memory),
                        'max_memory': np.max(all_gpu_memory)
                    }
        
        # Network I/O summary
        if any(s.network_io for s in data):
            network_data = [s.network_io for s in data if s.network_io]
            if network_data:
                total_bytes_sent = sum(n.get('bytes_sent', 0) for n in network_data)
                total_bytes_recv = sum(n.get('bytes_recv', 0) for n in network_data)
                
                summary['network_io'] = {
                    'total_bytes_sent': total_bytes_sent,
                    'total_bytes_recv': total_bytes_recv,
                    'total_mb_sent': total_bytes_sent / (1024**2),
                    'total_mb_recv': total_bytes_recv / (1024**2)
                }
        
        return summary
    
    def export_data(self, filepath: Path, format: str = 'json'):
        """Export monitoring data to file"""
        
        if not self.resource_history:
            raise ValueError("No monitoring data to export")
        
        if format.lower() == 'json':
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'monitoring_interval': self.monitoring_interval,
                    'data_points': len(self.resource_history)
                },
                'resource_history': [
                    {
                        'timestamp': snapshot.timestamp.isoformat(),
                        'cpu_percent': snapshot.cpu_percent,
                        'memory_percent': snapshot.memory_percent,
                        'memory_gb': snapshot.memory_gb,
                        'disk_usage_percent': snapshot.disk_usage_percent,
                        'disk_free_gb': snapshot.disk_free_gb,
                        'network_io': snapshot.network_io,
                        'gpu_usage': snapshot.gpu_usage,
                        'process_count': snapshot.process_count,
                        'load_average': snapshot.load_average
                    }
                    for snapshot in self.resource_history
                ],
                'alerts': self.alerts,
                'summary': self.get_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == 'csv':
            import pandas as pd
            
            # Convert to DataFrame
            df_data = []
            for snapshot in self.resource_history:
                row = {
                    'timestamp': snapshot.timestamp,
                    'cpu_percent': snapshot.cpu_percent,
                    'memory_percent': snapshot.memory_percent,
                    'memory_gb': snapshot.memory_gb,
                    'disk_usage_percent': snapshot.disk_usage_percent,
                    'disk_free_gb': snapshot.disk_free_gb,
                    'process_count': snapshot.process_count
                }
                
                # Add network I/O if available
                if snapshot.network_io:
                    for key, value in snapshot.network_io.items():
                        row[f'network_{key}'] = value
                
                # Add GPU data if available
                if snapshot.gpu_usage:
                    for i, gpu in enumerate(snapshot.gpu_usage):
                        row[f'gpu_{i}_load'] = gpu['load']
                        row[f'gpu_{i}_memory_percent'] = gpu['memory_percent']
                        row[f'gpu_{i}_temperature'] = gpu['temperature']
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported monitoring data to {filepath}")
    
    def get_resource_alerts(self) -> List[Dict[str, Any]]:
        """Get all resource usage alerts"""
        return self.alerts.copy()
    
    def clear_alerts(self):
        """Clear all stored alerts"""
        self.alerts.clear()
        self.logger.info("Cleared all resource alerts")
    
    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active"""
        return self.monitoring
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get static system information"""
        
        info = {
            'cpu': {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                'current_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'disk': {
                'total_gb': psutil.disk_usage('.').total / (1024**3),
                'free_gb': psutil.disk_usage('.').free / (1024**3)
            },
            'system': {
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'platform': psutil.os.name
            }
        }
        
        # Add GPU info if available
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                info['gpu'] = [
                    {
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'driver': getattr(gpu, 'driver', 'unknown')
                    }
                    for gpu in gpus
                ]
            except:
                info['gpu'] = 'GPU information unavailable'
        else:
            info['gpu'] = 'GPU monitoring not available (GPUtil not installed)'
        
        return info

# Convenience function for quick monitoring
def monitor_experiment(duration_minutes: int = 10,
                      output_file: str = None) -> Dict[str, Any]:
    """
    Monitor system resources for specified duration and return summary
    
    Args:
        duration_minutes: How long to monitor
        output_file: Optional file to save detailed data
    
    Returns:
        Resource usage summary
    """
    
    monitor = ResourceMonitor(monitoring_interval=1.0)
    
    print(f"Starting resource monitoring for {duration_minutes} minutes...")
    monitor.start_monitoring()
    
    try:
        time.sleep(duration_minutes * 60)
    except KeyboardInterrupt:
        print("Monitoring interrupted by user")
    finally:
        monitor.stop_monitoring()
    
    summary = monitor.get_summary()
    
    if output_file:
        monitor.export_data(Path(output_file), 'json')
        print(f"Detailed monitoring data saved to {output_file}")
    
    return summary

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Resource Monitor")
    parser.add_argument("--duration", type=int, default=5, help="Monitoring duration in minutes")
    parser.add_argument("--output", type=str, help="Output file for detailed data")
    
    args = parser.parse_args()
    
    summary = monitor_experiment(args.duration, args.output)
    
    print("\n=== Resource Monitoring Summary ===")
    print(f"CPU Usage: {summary['cpu_usage']['mean']:.1f}% avg, {summary['cpu_usage']['max']:.1f}% max")
    print(f"Memory Usage: {summary['memory_usage']['mean_gb']:.1f} GB avg, {summary['memory_usage']['max_gb']:.1f} GB max")
    print(f"Disk Usage: {summary['disk_usage']['mean']:.1f}%")
    
    if 'gpu_usage' in summary:
        print(f"GPU Usage: {summary['gpu_usage']['mean_load']:.1f}% avg load")
    
    if summary['alerts']['total_alerts'] > 0:
        print(f"⚠️ Total Alerts: {summary['alerts']['total_alerts']}")