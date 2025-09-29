import math
import jax
import jax.numpy as jnp
import functools
from msign import msign


@functools.partial(jax.jit, static_argnums=(4,))
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6):
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T

    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)

    # Define the loop body for jax.lax.fori_loop
    def loop_body(step, Lambda):
        # Update the candidate direction A
        A = msign(G + 2 * W @ Lambda, steps=10)  # Fixed steps for msign
        # Measure deviation of A from the tangent space:
        H = W.T @ A + A.T @ W
        # Update the dual variable (no early stopping for JIT compatibility)
        Lambda = Lambda - alpha * (1 - step / steps) * H
        return Lambda

    # Run the fixed number of steps using jax.lax.fori_loop
    Lambda = jax.lax.fori_loop(0, steps, loop_body, Lambda)

    # Final update direction
    A = msign(G + 2 * W @ Lambda, steps=10)

    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W, steps=10)

    # Restore the shape of the solution and return
    return new_W.T if should_tranpose else new_W
