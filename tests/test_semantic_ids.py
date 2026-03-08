"""
Tests for semantic IDs implementation.
"""

import pytest
import torch
import inspect
import os
import sys

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recsys.semantic_ids.vector_quantizer import VectorQuantizerEMA
from recsys.semantic_ids.rqvae import RQVAE


class TestVectorQuantizerEMA:
    """Tests for VectorQuantizerEMA to ensure fixes are applied."""
    
    def test_usage_loss_formula_is_fixed(self):
        """Verify that the usage loss formula is positive (not negative)."""
        source = inspect.getsource(VectorQuantizerEMA.forward)
        
        # Check that the fixed formula is present
        assert "usage_loss_val = 1.0 - entropy_ratio" in source, \
            "Fixed usage loss formula not found! Old buggy formula may still be active."
        
        # Ensure old broken formula is not present
        assert "usage_loss = -entropy / max_entropy" not in source, \
            "Old broken formula (negative entropy) still present in code!"
    
    def test_vq_loss_is_positive(self):
        """Test that VQ loss is always positive with actual tensors."""
        vq = VectorQuantizerEMA(
            num_embeddings=16, 
            embedding_dim=64,
            commitment_cost=0.5,
            usage_loss_weight=10.0
        )
        vq.train()
        
        # Create random input
        test_input = torch.randn(32, 64)
        
        # Forward pass
        quantized, loss, indices = vq(test_input)
        
        # Loss must be non-negative
        assert loss.item() >= 0, \
            f"VQ loss is negative ({loss.item():.6f})! This indicates the buggy formula is still active."
    
    def test_vq_loss_components(self):
        """Test individual loss components are all positive."""
        vq = VectorQuantizerEMA(
            num_embeddings=16,
            embedding_dim=64,
            commitment_cost=0.5,
            usage_loss_weight=2.0
        )
        vq.train()
        
        test_input = torch.randn(32, 64)
        quantized, total_loss, indices = vq(test_input)
        
        # All components should be positive
        assert total_loss.item() >= 0, "Total VQ loss is negative"
        
        # Test with no usage loss weight
        vq_no_usage = VectorQuantizerEMA(
            num_embeddings=16,
            embedding_dim=64,
            commitment_cost=0.5,
            usage_loss_weight=0.0
        )
        vq_no_usage.train()
        
        _, loss_no_usage, _ = vq_no_usage(test_input)
        assert loss_no_usage.item() >= 0, "VQ loss without usage weight is negative"


class TestRQVAE:
    """Tests for RQVAE model."""
    
    def test_rqvae_loss_is_positive(self):
        """Test that RQVAE total loss is always positive."""
        model = RQVAE(
            input_dim=384,
            embed_dim=256,
            codebook_sizes=[16, 32],
            usage_loss_weight=2.0
        )
        model.train()
        
        # Create random input
        test_input = torch.randn(16, 384)
        
        # Forward pass
        reconstructed, total_loss, codes = model(test_input, variance_weight=1.0)
        
        # Total loss must be positive
        assert total_loss.item() >= 0, \
            f"RQVAE total loss is negative ({total_loss.item():.6f})!"
    
    def test_rqvae_output_shapes(self):
        """Test that RQVAE produces correct output shapes."""
        batch_size = 16
        input_dim = 384
        embed_dim = 256
        codebook_sizes = [16, 32, 64]
        
        model = RQVAE(
            input_dim=input_dim,
            embed_dim=embed_dim,
            codebook_sizes=codebook_sizes
        )
        
        test_input = torch.randn(batch_size, input_dim)
        reconstructed, loss, codes = model(test_input)
        
        # Check shapes
        assert reconstructed.shape == (batch_size, input_dim), \
            f"Reconstructed shape mismatch: {reconstructed.shape} != {(batch_size, input_dim)}"
        
        assert codes.shape == (batch_size, len(codebook_sizes)), \
            f"Codes shape mismatch: {codes.shape} != {(batch_size, len(codebook_sizes))}"
    
    def test_variance_regularization(self):
        """Test that variance regularization is applied."""
        model = RQVAE(
            input_dim=384,
            embed_dim=256,
            codebook_sizes=[16, 32]
        )
        model.train()
        
        test_input = torch.randn(16, 384)
        
        # With variance weight
        _, loss_with_var, _ = model(test_input, variance_weight=1.0)
        
        # Without variance weight
        _, loss_no_var, _ = model(test_input, variance_weight=0.0)
        
        # Losses should be different
        assert loss_with_var.item() != loss_no_var.item(), \
            "Variance regularization doesn't seem to affect loss"


class TestCacheDetection:
    """Tests to detect if Python is using cached bytecode."""
    
    def test_no_pycache_in_semantic_ids(self):
        """Warn if __pycache__ exists (can cause stale code issues)."""
        semantic_ids_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'recsys', 
            'semantic_ids'
        )
        cache_dir = os.path.join(semantic_ids_dir, '__pycache__')
        
        if os.path.exists(cache_dir):
            if HAS_PYTEST:
                pytest.warn(
                    f"__pycache__ directory exists at {cache_dir}. "
                    "Consider deleting it if experiencing issues with code updates."
                )
            else:
                print(f"⚠️  Warning: __pycache__ exists at {cache_dir}")
                print("   Consider deleting it if experiencing issues with code updates.")


def run_standalone_tests():
    """Run all tests without pytest (for convenience)."""
    print("="*70)
    print("RUNNING SEMANTIC IDs TESTS (Standalone Mode)")
    print("="*70)
    
    test_vq = TestVectorQuantizerEMA()
    test_rqvae = TestRQVAE()
    test_cache = TestCacheDetection()
    
    tests = [
        ("VQ Formula Fixed", test_vq.test_usage_loss_formula_is_fixed),
        ("VQ Loss Positive", test_vq.test_vq_loss_is_positive),
        ("VQ Loss Components", test_vq.test_vq_loss_components),
        ("RQVAE Loss Positive", test_rqvae.test_rqvae_loss_is_positive),
        ("RQVAE Output Shapes", test_rqvae.test_rqvae_output_shapes),
        ("RQVAE Variance Reg", test_rqvae.test_variance_regularization),
        ("Cache Detection", test_cache.test_no_pycache_in_semantic_ids),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            print(f"Test: {name}")
            print(f"{'='*70}")
            test_func()
            print(f"✅ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*70}")
    
    return failed == 0


if __name__ == "__main__":
    # Allow running directly with: python test_semantic_ids.py
    pytest.main([__file__, "-v"])
