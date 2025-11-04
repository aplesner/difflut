"""
Comprehensive forward pass tests for all encoder types.
Tests: forward pass shape and range [0,1] for all encoders.
"""

import sys
import torch
from test_utils import (
    print_section,
    print_subsection,
    print_test_result,
    get_all_registered_encoders,
    instantiate_encoder,
    generate_uniform_input,
    assert_shape_equal,
    assert_range,
    IgnoreWarnings,
)


class EncoderTester:
    """Helper class for testing encoders."""
    
    def __init__(self, encoder_name: str, encoder_class: type):
        self.encoder_name = encoder_name
        self.encoder_class = encoder_class
        self.tests_passed = 0
        self.tests_failed = 0
    
    def test_forward_pass_shape_flatten_true(self):
        """Test 3: Forward pass with flatten=True produces correct shape."""
        try:
            with IgnoreWarnings():
                encoder = instantiate_encoder(self.encoder_class, num_bits=8, flatten=True)
            encoder.eval()
            
            # Create input data
            input_data = generate_uniform_input((10, 50), seed=42)  # 10 samples, 50 features
            
            # Fit encoder
            encoder.fit(input_data)
            
            # Test forward pass
            with torch.no_grad():
                output = encoder(input_data)
            
            # With flatten=True, output should be 2D: (batch_size, num_features * num_bits)
            expected_shape = (10, 50 * 8)  # 50 features * 8 bits
            assert_shape_equal(output, expected_shape)
            
            print_test_result(f"{self.encoder_name}: Shape (flatten=True)", True)
            self.tests_passed += 1
            return True
            
        except TypeError as e:
            # Some encoders might not support flatten parameter
            if "flatten" in str(e):
                print_test_result(f"{self.encoder_name}: Shape (flatten=True)", None, "flatten not supported")
                return None
            print_test_result(f"{self.encoder_name}: Shape (flatten=True)", False, str(e))
            self.tests_failed += 1
            return False
        except Exception as e:
            print_test_result(f"{self.encoder_name}: Shape (flatten=True)", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_forward_pass_shape_flatten_false(self):
        """Test 3: Forward pass with flatten=False produces correct shape."""
        try:
            with IgnoreWarnings():
                encoder = instantiate_encoder(self.encoder_class, num_bits=8, flatten=False)
            encoder.eval()
            
            # Create input data
            input_data = generate_uniform_input((10, 50), seed=42)  # 10 samples, 50 features
            
            # Fit encoder
            encoder.fit(input_data)
            
            # Test forward pass
            with torch.no_grad():
                output = encoder(input_data)
            
            # With flatten=False, output should be 3D: (batch_size, num_features, num_bits)
            expected_shape = (10, 50, 8)  # 10 batch, 50 features, 8 bits
            assert_shape_equal(output, expected_shape)
            
            print_test_result(f"{self.encoder_name}: Shape (flatten=False)", True)
            self.tests_passed += 1
            return True
            
        except TypeError as e:
            # Some encoders might not support flatten parameter
            if "flatten" in str(e):
                print_test_result(f"{self.encoder_name}: Shape (flatten=False)", None, "flatten not supported")
                return None
            print_test_result(f"{self.encoder_name}: Shape (flatten=False)", False, str(e))
            self.tests_failed += 1
            return False
        except Exception as e:
            print_test_result(f"{self.encoder_name}: Shape (flatten=False)", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_output_range_flatten_true(self):
        """Test 3: Output range is [0, 1] with flatten=True."""
        try:
            with IgnoreWarnings():
                encoder = instantiate_encoder(self.encoder_class, num_bits=8, flatten=True)
            encoder.eval()
            
            # Create and fit encoder
            input_data = generate_uniform_input((20, 30), seed=42)
            encoder.fit(input_data)
            
            # Test multiple random inputs
            for seed in [42, 123, 456]:
                test_input = generate_uniform_input((10, 30), seed=seed)
                
                with torch.no_grad():
                    output = encoder(test_input)
                
                assert_range(output, 0.0, 1.0, msg=f"seed={seed}, flatten=True")
            
            print_test_result(f"{self.encoder_name}: Range [0,1] (flatten=True)", True)
            self.tests_passed += 1
            return True
            
        except TypeError as e:
            if "flatten" in str(e):
                print_test_result(f"{self.encoder_name}: Range [0,1] (flatten=True)", None, "flatten not supported")
                return None
            print_test_result(f"{self.encoder_name}: Range [0,1] (flatten=True)", False, str(e))
            self.tests_failed += 1
            return False
        except Exception as e:
            print_test_result(f"{self.encoder_name}: Range [0,1] (flatten=True)", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_output_range_flatten_false(self):
        """Test 3: Output range is [0, 1] with flatten=False."""
        try:
            with IgnoreWarnings():
                encoder = instantiate_encoder(self.encoder_class, num_bits=8, flatten=False)
            encoder.eval()
            
            # Create and fit encoder
            input_data = generate_uniform_input((20, 30), seed=42)
            encoder.fit(input_data)
            
            # Test multiple random inputs
            for seed in [42, 123, 456]:
                test_input = generate_uniform_input((10, 30), seed=seed)
                
                with torch.no_grad():
                    output = encoder(test_input)
                
                assert_range(output, 0.0, 1.0, msg=f"seed={seed}, flatten=False")
            
            print_test_result(f"{self.encoder_name}: Range [0,1] (flatten=False)", True)
            self.tests_passed += 1
            return True
            
        except TypeError as e:
            if "flatten" in str(e):
                print_test_result(f"{self.encoder_name}: Range [0,1] (flatten=False)", None, "flatten not supported")
                return None
            print_test_result(f"{self.encoder_name}: Range [0,1] (flatten=False)", False, str(e))
            self.tests_failed += 1
            return False
        except Exception as e:
            print_test_result(f"{self.encoder_name}: Range [0,1] (flatten=False)", False, str(e))
            self.tests_failed += 1
            return False
    
    def test_fit_and_encode(self):
        """Test: Encoder fit() and encode work correctly."""
        try:
            with IgnoreWarnings():
                encoder = instantiate_encoder(self.encoder_class, num_bits=4)
            
            # Create training data
            train_data = generate_uniform_input((100, 20), seed=42)
            
            # Fit encoder
            encoder.fit(train_data)
            
            # Encode data
            test_data = generate_uniform_input((10, 20), seed=123)
            with torch.no_grad():
                encoded = encoder(test_data)
            
            assert encoded is not None
            assert encoded.numel() > 0
            
            print_test_result(f"{self.encoder_name}: Fit & Encode", True)
            self.tests_passed += 1
            return True
            
        except Exception as e:
            print_test_result(f"{self.encoder_name}: Fit & Encode", False, str(e))
            self.tests_failed += 1
            return False


def test_all_encoders():
    """Test all registered encoders."""
    print_section("ENCODER FORWARD PASS TESTS")
    
    encoders = get_all_registered_encoders()
    print(f"\nTesting {len(encoders)} encoders: {list(encoders.keys())}\n")
    
    all_passed = 0
    all_failed = 0
    
    for encoder_name, encoder_class in encoders.items():
        print_subsection(f"Encoder: {encoder_name}")
        
        tester = EncoderTester(encoder_name, encoder_class)
        
        # Run all tests
        tester.test_forward_pass_shape_flatten_true()
        tester.test_forward_pass_shape_flatten_false()
        tester.test_output_range_flatten_true()
        tester.test_output_range_flatten_false()
        tester.test_fit_and_encode()
        
        all_passed += tester.tests_passed
        all_failed += tester.tests_failed
        
        print(f"  → {tester.tests_passed} passed, {tester.tests_failed} failed")
    
    return all_passed, all_failed


def main():
    """Run all encoder tests."""
    print("\n" + "=" * 70)
    print("  ENCODER FORWARD PASS TEST SUITE")
    print("=" * 70)
    
    passed, failed = test_all_encoders()
    
    # Summary
    print_section("SUMMARY")
    print(f"  Total: {passed} passed, {failed} failed")
    
    if failed > 0:
        print(f"\n⚠ {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("\n✓ All encoder tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
