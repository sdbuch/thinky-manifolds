import jax
import jax.numpy as jnp


@jax.jit
def hyperspherical_descent(W, G, eta=0.1):
    w = W.flatten()
    g = G.flatten()
    # Compute update direction
    a = g - w * jnp.dot(w, g)
    a /= jnp.linalg.norm(a) + 1e-12
    # Apply update
    new_w = w - eta * a
    # Retract to the manifold
    new_w /= jnp.linalg.norm(new_w)
    # Restore the shape of the solution and return
    return new_w.reshape(W.shape)
