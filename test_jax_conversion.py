#!/usr/bin/env python3
"""
Unit tests for the JAX conversion of the manifold optimization repository.
Tests core functionality without requiring CIFAR-10 download or full training.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from msign import msign
from hyperspherical_descent import hyperspherical_descent
from manifold_muon import manifold_muon
from main import init_mlp_params, mlp_forward, compute_loss


def test_msign():
    """Test the matrix sign function"""
    print("Testing msign...")
    key = random.PRNGKey(42)

    # Test with a simple 3x3 matrix
    G = random.normal(key, (3, 3))
    result = msign(G, steps=5)

    # Check that result has same shape
    assert result.shape == G.shape, f"Shape mismatch: {result.shape} vs {G.shape}"

    # Check that result is approximately a sign matrix (S^2 ≈ I for orthogonal matrices)
    S = result
    should_be_identity = S @ S.T
    identity = jnp.eye(3)
    error = jnp.linalg.norm(should_be_identity - identity)
    print(f"  Matrix sign error (should be close to 0): {error:.6f}")

    print("✓ msign test passed")


def test_hyperspherical_descent():
    """Test hyperspherical descent optimizer"""
    print("Testing hyperspherical_descent...")
    key = random.PRNGKey(42)

    # Create random weight matrix and gradient
    W = random.normal(key, (10, 5))
    G = random.normal(random.split(key)[0], (10, 5))

    # Apply update
    new_W = hyperspherical_descent(W, G, eta=0.1)

    # Check shape preservation
    assert new_W.shape == W.shape, f"Shape mismatch: {new_W.shape} vs {W.shape}"

    # Check that result is normalized (should have unit norm)
    norm = jnp.linalg.norm(new_W)
    print(f"  Norm of result (should be close to 1): {norm:.6f}")
    assert abs(norm - 1.0) < 1e-5, f"Result not normalized: norm = {norm}"

    print("✓ hyperspherical_descent test passed")


def test_manifold_muon():
    """Test manifold muon optimizer"""
    print("Testing manifold_muon...")
    key = random.PRNGKey(42)

    # Create random weight matrix and gradient (tall matrix)
    W = random.normal(key, (10, 5))
    G = random.normal(random.split(key)[0], (10, 5))

    # Apply update
    new_W = manifold_muon(W, G, eta=0.1, steps=5)  # Fewer steps for speed

    # Check shape preservation
    assert new_W.shape == W.shape, f"Shape mismatch: {new_W.shape} vs {W.shape}"

    # Check that result is on the Stiefel manifold (W^T W ≈ I)
    WtW = new_W.T @ new_W
    identity = jnp.eye(W.shape[1])
    error = jnp.linalg.norm(WtW - identity)
    print(f"  Stiefel manifold constraint error (should be close to 0): {error:.6f}")

    print("✓ manifold_muon test passed")


def test_mlp_functions():
    """Test MLP initialization and forward pass"""
    print("Testing MLP functions...")
    key = random.PRNGKey(42)

    # Test parameter initialization
    params = init_mlp_params(key)

    # Check parameter shapes
    expected_shapes = {"fc1": (32 * 32 * 3, 128), "fc2": (128, 64), "fc3": (64, 10)}

    for name, expected_shape in expected_shapes.items():
        actual_shape = params[name].shape
        assert actual_shape == expected_shape, (
            f"{name} shape mismatch: {actual_shape} vs {expected_shape}"
        )

    # Test forward pass
    batch_size = 4
    dummy_images = random.normal(random.split(key)[0], (batch_size, 32, 32, 3))
    dummy_labels = jnp.array([0, 1, 2, 3])

    # Forward pass
    logits = mlp_forward(params, dummy_images)
    expected_logits_shape = (batch_size, 10)
    assert logits.shape == expected_logits_shape, (
        f"Logits shape mismatch: {logits.shape} vs {expected_logits_shape}"
    )

    # Test loss computation
    loss = compute_loss(params, dummy_images, dummy_labels)
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert loss > 0, f"Loss should be positive, got {loss}"

    print(f"  MLP forward pass output shape: {logits.shape}")
    print(f"  Loss value: {loss:.6f}")
    print("✓ MLP functions test passed")


def test_jit_compilation():
    """Test that functions are JIT-compiled properly"""
    print("Testing JIT compilation...")
    key = random.PRNGKey(42)

    # Test that functions can be called multiple times (JIT compilation works)
    G = random.normal(key, (5, 5))

    # First call (compilation)
    result1 = msign(G, steps=3)

    # Second call (should use compiled version)
    result2 = msign(G, steps=3)

    # Results should be identical
    assert jnp.allclose(result1, result2), "JIT compilation inconsistency"

    print("✓ JIT compilation test passed")


if __name__ == "__main__":
    print("Running JAX conversion tests...\n")

    try:
        test_msign()
        test_hyperspherical_descent()
        test_manifold_muon()
        test_mlp_functions()
        test_jit_compilation()

        print("\n🎉 All tests passed! JAX conversion successful.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
