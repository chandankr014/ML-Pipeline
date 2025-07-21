"""
Simple test script to verify the models structure works correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from models import ModelFactory, RandomForestModel, LightGBMModel
        from models import tune_all_models, compare_tuned_vs_untuned
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from models import ModelFactory
        
        # Test available models
        available_models = ModelFactory.get_available_models()
        print(f"✓ Available models: {list(available_models.keys())}")
        
        # Test creating individual models
        rf_model = ModelFactory.create_model("Random Forest", random_state=42)
        lgb_model = ModelFactory.create_model("LightGBM", random_state=42)
        
        print("✓ Model creation successful!")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        traceback.print_exc()
        return False

def test_model_methods():
    """Test model methods."""
    print("\nTesting model methods...")
    
    try:
        from models import ModelFactory
        
        # Test Random Forest model
        rf_model = ModelFactory.create_model("Random Forest", random_state=42)
        
        # Test get_model method
        model_instance = rf_model.get_model()
        print(f"✓ get_model() returns: {type(model_instance).__name__}")
        
        # Test get_param_distributions method
        param_dist = rf_model.get_param_distributions()
        print(f"✓ get_param_distributions() returns {len(param_dist)} parameters")
        
        # Test get_default_params method
        default_params = rf_model.get_default_params()
        print(f"✓ get_default_params() returns {len(default_params)} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model methods error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("TESTING MODELS STRUCTURE")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_model_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! The models structure is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 