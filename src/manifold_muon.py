import math

import torch
import wandb

from msign import msign


@torch.no_grad()
def dual_ascent_update_subgradient(W, G, Lambda):
    # Update the candidate direction A
    A = msign(G + 2 * W @ Lambda)
    # Measure deviation of A from the tangent space:
    H = W.T @ A + A.T @ W
    return A, H


@torch.no_grad()
def dual_ascent_update_admm(W, G, Lambda, Omega, D, rho):
    # Update for Lambda (orthonormal least-squares solve)
    P = W.mT @ (1 / rho * D + Omega - G)
    Lambda_upd = 0.25 * (P + P.mT)
    # Update for Omega (singular value thresholding)
    B = G + 2 * W @ Lambda_upd - 1 / rho * D
    # Clip with SVD (sad face)
    # U, s, VT = torch.linalg.svd(B, full_matrices=False)
    # s_upd = torch.clamp(s - 1 / rho, min=0.0)
    # Omega_upd = U @ torch.diag(s_upd) @ VT
    # Clip with msign (happy face?)
    eye = torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
    P_pos = 0.5 * (eye + msign(B.mT @ B - 1 / rho**2 * eye))
    Omega_upd = (B - 1/rho * msign(B)) @ P_pos
    # Update for D (dual ascent)
    D_upd = D + rho * (Omega_upd - 2 * W @ Lambda_upd - G)
    return Lambda_upd, Omega_upd, D_upd


@torch.no_grad()
def manifold_muon(
    W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6, prefix=None, admm_rho=None
):
    """
    Note that this actually implements GD on || G + W @ (L + L.mT) ||_*,
    whereas the blog discusses the parameterization with an extra factor of 2 on L
    It exploits the property that if L is initialized symmetric, it stays symmetric
    """
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    if admm_rho is None:
        # Initialize the dual variable
        Lambda = -0.25 * (W.T @ G + G.T @ W)
    else:
        # Initializations for ADMM
        # Lambda = -0.25 * (W.T @ G + G.T @ W)
        # Omega = G + 2 * W @ Lambda
        Lambda = torch.zeros_like(W.T @ G)
        Omega = G
        D = torch.zeros_like(Omega)

    # Track losses and effective step sizes
    dual_losses = []
    effective_step_sizes = []

    # Ascend on the dual problem to find the update direction A
    for step in range(steps):
        # Compute and log the dual ascent loss
        loss_matrix = G + 2 * W @ Lambda
        dual_loss = torch.linalg.svdvals(loss_matrix).sum()
        dual_losses.append(dual_loss.item())

        # Compute and log the effective step size
        effective_step_size = (1 - step / steps) * alpha
        effective_step_sizes.append(effective_step_size)

        # Log to wandb if outer_step is provided
        if prefix is not None:
            wandb.log(
                {
                    f"{prefix}/inner_step": step,
                    f"{prefix}/dual_loss": dual_loss.item(),
                    f"{prefix}/effective_step_size": effective_step_size,
                }
            )

        if admm_rho is None:
            # Do subgradient descent
            A, H = dual_ascent_update_subgradient(W, G, Lambda)

            # Check the stopping criterion
            if torch.norm(H) / math.sqrt(H.numel()) < tol:
                break
            # Update the dual variable
            Lambda -= alpha * (1 - step / steps) * H
        else:
            # Do ADMM
            Lambda, Omega, D = dual_ascent_update_admm(W, G, Lambda, Omega, D, admm_rho)

    if admm_rho is not None:
        # Calculate A for ADMM
        A = msign(G + 2 * W @ Lambda)

    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    result = new_W.T if should_tranpose else new_W
    return result, dual_losses, effective_step_sizes
