"""
Logging utilities for subgroup fairness experiments
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logging(log_file: Optional[Path] = None, 
                 level: str = "INFO",
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for experiments
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger("subgroup_fairness")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info("Logging initialized")
    return logger

class ExperimentLogger:
    """Enhanced logger for experiment tracking"""
    
    def __init__(self, experiment_name: str, output_dir: Path):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logging(log_file=log_file)
        
        # Track experiment progress
        self.experiment_start_time = datetime.now()
        self.current_stage = "initialization"
        
        self.logger.info(f"Starting experiment: {experiment_name}")
    
    def log_experiment_start(self, config: dict):
        """Log experiment configuration at start"""
        self.logger.info("="*80)
        self.logger.info(f"EXPERIMENT: {self.experiment_name}")
        self.logger.info("="*80)
        
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("-"*80)
    
    def log_stage_start(self, stage_name: str, details: Optional[str] = None):
        """Log the start of an experiment stage"""
        self.current_stage = stage_name
        self.logger.info(f"Starting stage: {stage_name}")
        if details:
            self.logger.info(f"  Details: {details}")
    
    def log_stage_end(self, stage_name: str, results: Optional[dict] = None):
        """Log the end of an experiment stage"""
        self.logger.info(f"Completed stage: {stage_name}")
        if results:
            self.logger.info("  Results:")
            for key, value in results.items():
                self.logger.info(f"    {key}: {value}")
    
    def log_method_performance(self, method_name: str, metrics: dict):
        """Log performance metrics for a method"""
        self.logger.info(f"Method: {method_name}")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")
    
    def log_comparison(self, method_1: str, method_2: str, 
                      metric: str, value_1: float, value_2: float):
        """Log comparison between two methods"""
        diff = value_1 - value_2
        pct_diff = (diff / value_2) * 100 if value_2 != 0 else float('inf')
        
        self.logger.info(f"Comparison - {metric}:")
        self.logger.info(f"  {method_1}: {value_1:.4f}")
        self.logger.info(f"  {method_2}: {value_2:.4f}")
        self.logger.info(f"  Difference: {diff:+.4f} ({pct_diff:+.1f}%)")
    
    def log_progress(self, current: int, total: int, message: str = ""):
        """Log progress through a process"""
        percent = (current / total) * 100
        self.logger.info(f"Progress: {current}/{total} ({percent:.1f}%) {message}")
    
    def log_experiment_summary(self, results: dict):
        """Log final experiment summary"""
        duration = datetime.now() - self.experiment_start_time
        
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("="*80)
        
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Status: COMPLETED")
        
        if results:
            self.logger.info("\nKey Results:")
            for key, value in results.items():
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("="*80)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error(f"Error in {context}: {error}")
        self.logger.error(f"Error type: {type(error).__name__}")
        
        # Log traceback in debug mode
        import traceback
        self.logger.debug(f"Traceback: {traceback.format_exc()}")
    
    def log_warning(self, message: str, context: str = ""):
        """Log warning with context"""
        if context:
            self.logger.warning(f"{context}: {message}")
        else:
            self.logger.warning(message)

def create_progress_logger(name: str, total_steps: int):
    """Create a simple progress logger"""
    logger = logging.getLogger(f"progress_{name}")
    current_step = 0
    
    def log_step(step_name: str = ""):
        nonlocal current_step
        current_step += 1
        percent = (current_step / total_steps) * 100
        logger.info(f"Step {current_step}/{total_steps} ({percent:.1f}%): {step_name}")
    
    return log_step