import math
import torch
from msign import msign
import wandb


@torch.no_grad()
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6, outer_step=None):
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)

    # Track losses and effective step sizes
    dual_losses = []
    effective_step_sizes = []

    # Ascend on the dual problem to find the update direction A
    for step in range(steps):
        # Compute and log the dual ascent loss
        loss_matrix = G + 2 * W @ (Lambda + Lambda.mT)
        dual_loss = torch.linalg.svdvals(loss_matrix).sum()
        dual_losses.append(dual_loss.item())

        # Compute and log the effective step size
        effective_step_size = (1 - step / steps) * alpha
        effective_step_sizes.append(effective_step_size)

        # Log to wandb if outer_step is provided
        if outer_step is not None:
            wandb.log(
                {
                    f"outer_step_{outer_step}/dual_loss": dual_loss.item(),
                    f"outer_step_{outer_step}/effective_step_size": effective_step_size,
                },
                step=step,
            )

        # Update the candidate direction A
        A = msign(G + 2 * W @ Lambda)
        # Measure deviation of A from the tangent space:
        H = W.T @ A + A.T @ W
        # Check the stopping criterion
        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break
        # Update the dual variable
        Lambda -= alpha * (1 - step / steps) * H
    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    result = new_W.T if should_tranpose else new_W
    return result, dual_losses, effective_step_sizes
