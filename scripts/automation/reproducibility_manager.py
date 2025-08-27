"""
Reproducibility Management System
Handles random seed management, environment capturing, version control integration
"""

import os
import sys
import json
import hashlib
import random
import subprocess
import platform
import pkg_resources
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import pickle
import numpy as np
import torch

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class ReproducibilityManager:
    """
    Comprehensive reproducibility management system
    """
    
    def __init__(self, base_output_dir: str = "reproducibility_artifacts"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger("reproducibility_manager")
        
        # Current experiment context
        self.current_seed = None
        self.current_env_hash = None
        self.current_git_state = None
        
        self.logger.info("ReproducibilityManager initialized")
    
    def setup_reproducible_environment(self, 
                                     seed: int = 42,
                                     deterministic_algorithms: bool = True) -> Dict[str, Any]:
        """
        Setup reproducible environment with fixed seed and deterministic operations
        
        Args:
            seed: Random seed to use
            deterministic_algorithms: Whether to enforce deterministic algorithms
            
        Returns:
            Reproducibility information dictionary
        """
        
        self.current_seed = seed
        
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy random seed
        np.random.seed(seed)
        
        # Set PyTorch seeds if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                
            if deterministic_algorithms:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # Set deterministic algorithms (PyTorch 1.7+)
                if hasattr(torch, 'use_deterministic_algorithms'):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except:
                        self.logger.warning("Could not enable deterministic algorithms")
                        
        except ImportError:
            self.logger.debug("PyTorch not available for seed setting")
        
        # Set TensorFlow seeds if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            self.logger.debug("TensorFlow not available for seed setting")
        
        # Set scikit-learn random state (environment variable)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Capture environment information
        env_info = self._capture_environment_info()
        git_info = self._capture_git_state()
        
        reproducibility_info = {
            'seed': seed,
            'deterministic_algorithms': deterministic_algorithms,
            'timestamp': datetime.now().isoformat(),
            'environment': env_info,
            'git_state': git_info,
            'python_path': sys.path.copy(),
            'working_directory': str(Path.cwd())
        }
        
        self.current_env_hash = self._compute_environment_hash(reproducibility_info)
        reproducibility_info['environment_hash'] = self.current_env_hash
        
        self.logger.info(f"Reproducible environment setup with seed {seed}")
        
        return reproducibility_info
    
    def _capture_environment_info(self) -> Dict[str, Any]:
        """Capture comprehensive environment information"""
        
        env_info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'python_implementation': platform.python_implementation()
            },
            'installed_packages': self._get_installed_packages(),
            'environment_variables': self._get_relevant_env_vars(),
            'hardware_info': self._get_hardware_info(),
            'conda_info': self._get_conda_info()
        }
        
        return env_info
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get all installed Python packages and their versions"""
        
        packages = {}
        
        try:
            # Get packages from pip
            installed_packages = [pkg for pkg in pkg_resources.working_set]
            for pkg in installed_packages:
                packages[pkg.project_name] = pkg.version
        except Exception as e:
            self.logger.warning(f"Could not get package list: {e}")
            
            # Fallback: try to get key packages
            key_packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 
                          'torch', 'tensorflow', 'matplotlib', 'seaborn']
            
            for pkg_name in key_packages:
                try:
                    pkg = pkg_resources.get_distribution(pkg_name)
                    packages[pkg_name] = pkg.version
                except:
                    continue
        
        return packages
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables for reproducibility"""
        
        relevant_vars = [
            'PYTHONPATH',
            'PYTHONHASHSEED', 
            'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'CONDA_DEFAULT_ENV',
            'VIRTUAL_ENV'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value:
                env_vars[var] = value
        
        return env_vars
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        
        hardware_info = {
            'cpu_count': os.cpu_count(),
            'cpu_physical_cores': None,
            'memory_total_gb': None,
            'gpu_info': []
        }
        
        # Get CPU info
        try:
            import psutil
            hardware_info['cpu_physical_cores'] = psutil.cpu_count(logical=False)
            hardware_info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
        
        # Get GPU info
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                hardware_info['gpu_info'].append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal
                })
        except (ImportError, Exception):
            pass
        
        return hardware_info
    
    def _get_conda_info(self) -> Optional[Dict[str, Any]]:
        """Get conda environment information if available"""
        
        try:
            # Check if we're in a conda environment
            conda_env = os.environ.get('CONDA_DEFAULT_ENV')
            if not conda_env:
                return None
            
            # Get conda info
            result = subprocess.run(['conda', 'info', '--json'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                conda_info = json.loads(result.stdout)
                return {
                    'active_env': conda_env,
                    'conda_version': conda_info.get('conda_version'),
                    'python_version': conda_info.get('python_version'),
                    'env_vars': conda_info.get('env_vars', {})
                }
        except Exception as e:
            self.logger.debug(f"Could not get conda info: {e}")
        
        return None
    
    def _capture_git_state(self) -> Optional[Dict[str, Any]]:
        """Capture current git repository state"""
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            git_info = {}
            
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Get repository status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                git_info['dirty'] = bool(result.stdout.strip())
                git_info['untracked_files'] = [
                    line[3:] for line in result.stdout.strip().split('\n')
                    if line.startswith('??')
                ]
                git_info['modified_files'] = [
                    line[3:] for line in result.stdout.strip().split('\n')
                    if line.startswith(' M') or line.startswith('M ')
                ]
            
            # Get remote origin URL
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                git_info['remote_origin'] = result.stdout.strip()
            
            # Get last commit info
            result = subprocess.run(['git', 'log', '-1', '--pretty=format:%H,%s,%an,%ad', '--date=iso'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                parts = result.stdout.strip().split(',', 3)
                if len(parts) == 4:
                    git_info['last_commit'] = {
                        'hash': parts[0],
                        'message': parts[1],
                        'author': parts[2],
                        'date': parts[3]
                    }
            
            self.current_git_state = git_info
            return git_info
            
        except Exception as e:
            self.logger.debug(f"Could not capture git state: {e}")
            return None
    
    def _compute_environment_hash(self, repro_info: Dict[str, Any]) -> str:
        """Compute hash of environment for reproducibility verification"""
        
        # Create a stable representation of the environment
        env_signature = {
            'python_version': repro_info['environment']['platform']['python_version'],
            'packages': repro_info['environment']['installed_packages'],
            'platform': repro_info['environment']['platform']['system'],
            'git_commit': repro_info['git_state']['commit_hash'] if repro_info['git_state'] else None
        }
        
        # Convert to stable JSON string
        signature_str = json.dumps(env_signature, sort_keys=True)
        
        # Compute hash
        env_hash = hashlib.sha256(signature_str.encode()).hexdigest()[:16]
        
        return env_hash
    
    def save_reproducibility_artifacts(self, output_dir: Path):
        """Save all reproducibility artifacts to specified directory"""
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save environment snapshot
        if self.current_seed is not None:
            repro_info = self.setup_reproducible_environment(self.current_seed)
            
            with open(output_dir / "environment_snapshot.json", 'w') as f:
                json.dump(repro_info, f, indent=2, default=str)
        
        # Save requirements.txt equivalent
        self._save_requirements_file(output_dir / "requirements.txt")
        
        # Save conda environment if available
        self._save_conda_environment(output_dir / "conda_environment.yml")
        
        # Save git patch if repository is dirty
        self._save_git_patch(output_dir / "uncommitted_changes.patch")
        
        # Save random state
        self._save_random_state(output_dir / "random_state.pkl")
        
        self.logger.info(f"Reproducibility artifacts saved to {output_dir}")
    
    def _save_requirements_file(self, filepath: Path):
        """Save requirements.txt file"""
        
        try:
            result = subprocess.run(['pip', 'freeze'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                with open(filepath, 'w') as f:
                    f.write(result.stdout)
                
                self.logger.debug(f"Requirements saved to {filepath}")
        except Exception as e:
            self.logger.warning(f"Could not save requirements: {e}")
    
    def _save_conda_environment(self, filepath: Path):
        """Save conda environment YAML if available"""
        
        try:
            if os.environ.get('CONDA_DEFAULT_ENV'):
                result = subprocess.run(['conda', 'env', 'export'], 
                                      capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    with open(filepath, 'w') as f:
                        f.write(result.stdout)
                    
                    self.logger.debug(f"Conda environment saved to {filepath}")
        except Exception as e:
            self.logger.debug(f"Could not save conda environment: {e}")
    
    def _save_git_patch(self, filepath: Path):
        """Save git patch for uncommitted changes"""
        
        try:
            # Check if there are uncommitted changes
            result = subprocess.run(['git', 'diff', '--quiet'], 
                                  capture_output=True, timeout=10)
            
            if result.returncode != 0:  # There are changes
                result = subprocess.run(['git', 'diff', 'HEAD'], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and result.stdout:
                    with open(filepath, 'w') as f:
                        f.write(result.stdout)
                    
                    self.logger.debug(f"Git patch saved to {filepath}")
        except Exception as e:
            self.logger.debug(f"Could not save git patch: {e}")
    
    def _save_random_state(self, filepath: Path):
        """Save current random state"""
        
        try:
            random_state = {
                'python_random_state': random.getstate(),
                'numpy_random_state': np.random.get_state(),
                'seed_used': self.current_seed
            }
            
            # Add PyTorch random state if available
            try:
                import torch
                random_state['torch_random_state'] = torch.get_rng_state()
                if torch.cuda.is_available():
                    random_state['torch_cuda_random_state'] = torch.cuda.get_rng_state_all()
            except ImportError:
                pass
            
            with open(filepath, 'wb') as f:
                pickle.dump(random_state, f)
            
            self.logger.debug(f"Random state saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Could not save random state: {e}")
    
    def restore_random_state(self, filepath: Path):
        """Restore random state from saved file"""
        
        try:
            with open(filepath, 'rb') as f:
                random_state = pickle.load(f)
            
            # Restore Python random state
            if 'python_random_state' in random_state:
                random.setstate(random_state['python_random_state'])
            
            # Restore NumPy random state
            if 'numpy_random_state' in random_state:
                np.random.set_state(random_state['numpy_random_state'])
            
            # Restore PyTorch random state if available
            try:
                import torch
                if 'torch_random_state' in random_state:
                    torch.set_rng_state(random_state['torch_random_state'])
                if 'torch_cuda_random_state' in random_state and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(random_state['torch_cuda_random_state'])
            except ImportError:
                pass
            
            self.logger.info(f"Random state restored from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Could not restore random state: {e}")
    
    def verify_reproducibility(self, 
                              baseline_env_file: Path,
                              current_repro_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Verify if current environment matches baseline for reproducibility"""
        
        try:
            # Load baseline environment
            with open(baseline_env_file, 'r') as f:
                baseline_env = json.load(f)
            
            # Get current environment if not provided
            if current_repro_info is None:
                current_repro_info = self.setup_reproducible_environment(42)
            
            verification_result = {
                'reproducible': True,
                'mismatches': [],
                'warnings': [],
                'environment_hash_match': False
            }
            
            # Check environment hashes
            baseline_hash = baseline_env.get('environment_hash')
            current_hash = current_repro_info.get('environment_hash')
            
            if baseline_hash and current_hash:
                verification_result['environment_hash_match'] = (baseline_hash == current_hash)
                if not verification_result['environment_hash_match']:
                    verification_result['warnings'].append("Environment hash mismatch - some differences detected")
            
            # Check critical components
            critical_checks = [
                ('python_version', ['environment', 'platform', 'python_version']),
                ('git_commit', ['git_state', 'commit_hash']),
                ('key_packages', ['environment', 'installed_packages'])
            ]
            
            for check_name, path in critical_checks:
                baseline_val = self._get_nested_value(baseline_env, path)
                current_val = self._get_nested_value(current_repro_info, path)
                
                if check_name == 'key_packages':
                    # Check key packages only
                    key_packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'torch']
                    package_mismatches = self._check_package_versions(
                        baseline_val or {}, current_val or {}, key_packages
                    )
                    if package_mismatches:
                        verification_result['mismatches'].extend(package_mismatches)
                        verification_result['reproducible'] = False
                
                elif baseline_val != current_val:
                    verification_result['mismatches'].append({
                        'component': check_name,
                        'baseline': baseline_val,
                        'current': current_val
                    })
                    
                    if check_name in ['python_version', 'git_commit']:
                        verification_result['reproducible'] = False
            
            return verification_result
            
        except Exception as e:
            return {
                'reproducible': False,
                'error': str(e),
                'mismatches': [],
                'warnings': []
            }
    
    def _get_nested_value(self, data: Dict, path: List[str]):
        """Get nested value from dictionary using path"""
        
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _check_package_versions(self, 
                               baseline_packages: Dict[str, str],
                               current_packages: Dict[str, str],
                               key_packages: List[str]) -> List[Dict[str, Any]]:
        """Check package version mismatches for key packages"""
        
        mismatches = []
        
        for package in key_packages:
            baseline_version = baseline_packages.get(package)
            current_version = current_packages.get(package)
            
            if baseline_version and current_version:
                if baseline_version != current_version:
                    mismatches.append({
                        'component': f'package_{package}',
                        'baseline': baseline_version,
                        'current': current_version
                    })
            elif baseline_version and not current_version:
                mismatches.append({
                    'component': f'package_{package}',
                    'baseline': baseline_version,
                    'current': 'not_installed'
                })
            elif not baseline_version and current_version:
                mismatches.append({
                    'component': f'package_{package}',
                    'baseline': 'not_installed',
                    'current': current_version
                })
        
        return mismatches
    
    def create_reproducibility_archive(self, 
                                     experiment_results: Dict[str, Any],
                                     output_dir: Path) -> Path:
        """Create complete reproducibility archive"""
        
        archive_dir = output_dir / f"repro_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_dir.mkdir(exist_ok=True, parents=True)
        
        # Save reproducibility artifacts
        self.save_reproducibility_artifacts(archive_dir / "environment")
        
        # Save experiment results
        with open(archive_dir / "experiment_results.json", 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
        
        # Create reproduction script
        self._create_reproduction_script(archive_dir / "reproduce.py")
        
        # Create README
        self._create_reproduction_readme(archive_dir / "README.md")
        
        self.logger.info(f"Reproducibility archive created at {archive_dir}")
        
        return archive_dir
    
    def _create_reproduction_script(self, filepath: Path):
        """Create script to reproduce the experiment"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Automatic reproduction script
Generated by ReproducibilityManager on {datetime.now().isoformat()}
"""

import sys
import json
from pathlib import Path

# Add the original project path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def reproduce_experiment():
    """Reproduce the experiment with saved configuration"""
    
    # Load environment snapshot
    with open("environment/environment_snapshot.json", 'r') as f:
        env_info = json.load(f)
    
    print(f"Reproducing experiment with seed: {{env_info['seed']}}")
    print(f"Original environment hash: {{env_info['environment_hash']}}")
    
    # Import and setup reproducibility manager
    from scripts.automation.reproducibility_manager import ReproducibilityManager
    from scripts.automation.experiment_executor import ExperimentExecutor
    
    repro_manager = ReproducibilityManager()
    
    # Setup reproducible environment
    current_env = repro_manager.setup_reproducible_environment(env_info['seed'])
    
    # Verify environment compatibility
    verification = repro_manager.verify_reproducibility(
        Path("environment/environment_snapshot.json"),
        current_env
    )
    
    if not verification['reproducible']:
        print("⚠️  Warning: Environment differences detected:")
        for mismatch in verification['mismatches']:
            print(f"  - {{mismatch['component']}}: {{mismatch['baseline']}} → {{mismatch['current']}}")
    else:
        print("✅ Environment verification passed")
    
    # TODO: Add specific experiment reproduction code here
    print("Add your experiment reproduction code here")
    
    return verification

if __name__ == "__main__":
    reproduce_experiment()
'''
        
        with open(filepath, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        try:
            import stat
            filepath.chmod(filepath.stat().st_mode | stat.S_IEXEC)
        except:
            pass
    
    def _create_reproduction_readme(self, filepath: Path):
        """Create README for reproduction"""
        
        readme_content = f"""# Experiment Reproduction Guide

This directory contains all necessary artifacts to reproduce the experiment.

Generated on: {datetime.now().isoformat()}

## Files

- `environment/` - Environment snapshots and requirements
- `experiment_results.json` - Original experiment results  
- `reproduce.py` - Automated reproduction script
- `README.md` - This file

## Quick Start

1. Install the same Python version and packages:
   ```bash
   pip install -r environment/requirements.txt
   ```
   
   Or if using conda:
   ```bash
   conda env create -f environment/conda_environment.yml
   ```

2. Run the reproduction script:
   ```bash
   python reproduce.py
   ```

## Environment Verification

The reproduction script will automatically verify that your environment 
matches the original experiment environment and warn about any differences.

## Manual Reproduction

If you need to reproduce manually:

1. Set the same random seed from `environment/environment_snapshot.json`
2. Use the same package versions from `requirements.txt`
3. Apply any uncommitted changes from `uncommitted_changes.patch` if present
4. Run your experiment with the same parameters

## Notes

- Environment hash provides a quick way to verify compatibility
- Git commit hash ensures you're using the same code version
- Random state files can be loaded to continue from exact same point
"""

        with open(filepath, 'w') as f:
            f.write(readme_content)

# Convenience functions
def setup_reproducible_experiment(seed: int = 42) -> Tuple[ReproducibilityManager, Dict[str, Any]]:
    """Quick setup for reproducible experiment"""
    
    manager = ReproducibilityManager()
    repro_info = manager.setup_reproducible_environment(seed)
    
    return manager, repro_info

def create_experiment_archive(results: Dict[str, Any], 
                            output_dir: str = "experiment_archives") -> Path:
    """Create complete experiment archive with reproducibility info"""
    
    manager = ReproducibilityManager()
    archive_path = manager.create_reproducibility_archive(
        results, Path(output_dir)
    )
    
    return archive_path

if __name__ == "__main__":
    # Example usage
    print("ReproducibilityManager - ensuring experiment reproducibility")
    
    # Setup reproducible environment
    manager, repro_info = setup_reproducible_experiment(seed=42)
    
    print(f"Environment hash: {repro_info['environment_hash']}")
    print(f"Git commit: {repro_info['git_state']['commit_hash'] if repro_info['git_state'] else 'N/A'}")
    print(f"Python version: {repro_info['environment']['platform']['python_version']}")