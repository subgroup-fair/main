"""
Quick Test for Validation System
Basic functionality test without external dependencies
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test basic validation imports
        from validation.unified_validation_controller import (
            ValidationConfig,
            UnifiedValidationController,
            create_validation_config
        )
        print("OK: Unified validation controller import successful")
        
        from validation.validation_cli import ValidationCLI
        print("OK: Validation CLI import successful")
        
        # Test individual validators (with graceful failure handling)
        try:
            from validation.statistical_validator import StatisticalValidator
            print("OK: Statistical validator import successful")
        except ImportError as e:
            print(f"WARN: Statistical validator import failed (expected): {e}")
        
        try:
            from validation.code_quality_validator import CodeQualityValidator  
            print("OK: Code quality validator import successful")
        except ImportError as e:
            print(f"WARN: Code quality validator import failed: {e}")
        
        try:
            from validation.scientific_rigor_validator import ScientificRigorValidator
            print("OK: Scientific rigor validator import successful")  
        except ImportError as e:
            print(f"WARN: Scientific rigor validator import failed: {e}")
        
        try:
            from validation.validation_reporter import ValidationReporter
            print("OK: Validation reporter import successful")
        except ImportError as e:
            print(f"WARN: Validation reporter import failed: {e}")
            
        return True
        
    except ImportError as e:
        print(f"FAIL: Critical import failed: {e}")
        return False

def test_config_creation():
    """Test validation configuration creation"""
    print("\nTesting configuration creation...")
    
    try:
        from validation.unified_validation_controller import create_validation_config
        
        config = create_validation_config(
            experiment_name="test_experiment",
            experiment_path="./test_path",
            run_statistical=True,
            run_code_quality=True,
            run_scientific_rigor=True
        )
        
        assert config.experiment_name == "test_experiment"
        assert config.experiment_path == "./test_path"
        assert config.run_statistical == True
        assert config.run_code_quality == True
        assert config.run_scientific_rigor == True
        
        print("OK: Configuration creation successful")
        return True
        
    except Exception as e:
        print(f"FAIL: Configuration creation failed: {e}")
        return False

def test_cli_parser():
    """Test CLI argument parser"""
    print("\nTesting CLI parser...")
    
    try:
        from validation.validation_cli import ValidationCLI
        
        cli = ValidationCLI()
        parser = cli.parser
        
        # Test that parser exists and has expected commands
        assert parser is not None
        
        # Test help (will raise SystemExit, which is expected)
        try:
            args = parser.parse_args(['--help'])
        except SystemExit:
            pass  # Expected behavior for help command
        
        # Test valid command parsing
        args = parser.parse_args(['init-config', '--output', 'test.yaml'])
        assert args.command == 'init-config'
        assert args.output == 'test.yaml'
        
        print("OK: CLI parser test successful")
        return True
        
    except Exception as e:
        print(f"FAIL: CLI parser test failed: {e}")
        return False

def test_file_structure():
    """Test that all expected files exist"""
    print("\nTesting file structure...")
    
    validation_dir = Path(__file__).parent
    expected_files = [
        "unified_validation_controller.py",
        "validation_cli.py", 
        "statistical_validator.py",
        "code_quality_validator.py",
        "scientific_rigor_validator.py",
        "validation_reporter.py",
        "interactive_dashboard.py",
        "advanced_visualization.py",
        "ai_code_analyzer.py",
        "advanced_bias_detector.py",
        "metadata_validator.py",
        "causal_inference_validator.py",
        "test_validation_system.py",
        "README.md"
    ]
    
    missing_files = []
    for filename in expected_files:
        file_path = validation_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            print(f"✓ {filename} exists")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All expected files present")
        return True

def test_controller_basic():
    """Test basic controller functionality"""
    print("\nTesting basic controller functionality...")
    
    try:
        from validation.unified_validation_controller import (
            UnifiedValidationController,
            create_validation_config
        )
        
        # Create controller
        controller = UnifiedValidationController()
        print("✓ Controller created successfully")
        
        # Test workflow listing (should be empty initially)
        active_workflows = controller.get_active_workflows()
        assert isinstance(active_workflows, list)
        print("✓ Active workflows listing works")
        
        # Test historical results (should work even if empty)
        history = controller.get_historical_results(limit=5)
        assert isinstance(history, list)
        print("✓ Historical results retrieval works")
        
        return True
        
    except Exception as e:
        print(f"✗ Controller basic test failed: {e}")
        return False

def run_quick_tests():
    """Run all quick tests"""
    print("=" * 60)
    print("QUICK VALIDATION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Config Creation", test_config_creation), 
        ("CLI Parser", test_cli_parser),
        ("File Structure", test_file_structure),
        ("Controller Basic", test_controller_basic)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("QUICK TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Success rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All quick tests passed! The validation system structure is working correctly.")
        print("Note: Full functionality requires installing dependencies from requirements.txt")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)