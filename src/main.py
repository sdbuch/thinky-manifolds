import argparse
import os
import pickle
import time
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader

from hyperspherical_descent import hyperspherical_descent
from manifold_muon import manifold_muon

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
        ),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transform, download=True
)


def seed_worker(worker_id):
    import numpy as np
    import random

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(seed):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)
    return train_loader, test_loader


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(
    epochs,
    initial_lr,
    update,
    wd,
    train_loader,
    test_loader,
    manifold_muon_params=None,
    device="cuda:0",
):
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()

    if update == AdamW:
        optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=wd)
    else:
        assert update in [manifold_muon, hyperspherical_descent]
        optimizer = None

    steps = epochs * len(train_loader)
    step = 0

    # Calculate number of digits needed for zero-padding
    num_epoch_digits = len(str(epochs - 1))
    num_dataloader_digits = len(str(len(train_loader) - 1))

    if optimizer is None:
        # Project the weights to the manifold
        for p in model.parameters():
            if update == manifold_muon:
                p.data, _, _ = update(
                    p.data,
                    torch.zeros_like(p.data),
                    eta=0,
                    prefix=None,
                    **manifold_muon_params,
                )
            else:
                p.data = update(p.data, torch.zeros_like(p.data), eta=0)

    epoch_losses = []
    epoch_times = []

    for epoch in range(epochs):
        for dataloader_idx in range(len(train_loader)):
            for param, _ in model.named_parameters():
                prefix = f"{param}/e{epoch:0{num_epoch_digits}d}_d{dataloader_idx:0{num_dataloader_digits}d}"
                wandb.define_metric(f"{prefix}/inner_step")
                wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/inner_step")
    wandb.define_metric("outer_loop/global_step")
    wandb.define_metric("outer_loop/*", step_metric="outer_loop/global_step")

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            model.zero_grad()
            loss.backward()

            # Save parameters and gradients after Epoch 0, Batch 1
            if epoch == 0 and i == 1:
                import numpy as np
                params_and_grads = {}
                for n, p in model.named_parameters():
                    params_and_grads[f"{n}_param"] = p.data.cpu().numpy()
                    params_and_grads[f"{n}_grad"] = p.grad.cpu().numpy()
                os.makedirs("debug", exist_ok=True)
                np.savez("debug/epoch0_batch1_params_grads.npz", **params_and_grads)
                print("Saved parameters and gradients to debug/epoch0_batch1_params_grads.npz")

            lr = initial_lr * (1 - step / steps)
            with torch.no_grad():
                if optimizer is None:
                    for n, p in model.named_parameters():
                        prefix = f"{n}/e{epoch:0{num_epoch_digits}d}_d{i:0{num_dataloader_digits}d}"
                        if update == manifold_muon:
                            new_p, dual_losses, effective_step_sizes = update(
                                p,
                                p.grad,
                                eta=lr,
                                prefix=prefix,
                                **manifold_muon_params,
                            )
                            p.data = new_p
                        else:
                            p.data = update(p, p.grad, eta=lr)
                else:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    optimizer.step()

            # Log training loss and learning rate to wandb in outer_loop namespace
            wandb.log(
                {
                    "outer_loop/global_step": step,
                    "outer_loop/train_loss": loss.item(),
                    "outer_loop/learning_rate": lr,
                }
            )

            step += 1

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        end_time = time.time()
        epoch_loss = running_loss / len(train_loader)
        epoch_time = end_time - start_time
        epoch_losses.append(epoch_loss)
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Time: {epoch_time:.4f} seconds")
    return model, epoch_losses, epoch_times


def eval(model, train_loader, test_loader, device="cuda:0"):
    # Test the model
    model.eval()
    with torch.no_grad():
        accs = []
        for dataloader in [test_loader, train_loader]:
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accs.append(100 * correct / total)

    print(
        f"Accuracy of the network on the {len(test_loader.dataset)} test images: {accs[0]} %"
    )
    print(
        f"Accuracy of the network on the {len(train_loader.dataset)} train images: {accs[1]} %"
    )
    return accs


def weight_stats(model):
    singular_values = []
    norms = []
    for p in model.parameters():
        u, s, v = torch.svd(p)
        singular_values.append(s)
        norms.append(p.norm())
    return singular_values, norms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10.")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument(
        "--update",
        type=str,
        default="manifold_muon",
        choices=["manifold_muon", "hyperspherical_descent", "adam"],
        help="Update rule to use.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator."
    )
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay for AdamW.")
    parser.add_argument(
        "--manifold_alpha",
        type=float,
        default=0.01,
        help="Alpha parameter for manifold_muon.",
    )
    parser.add_argument(
        "--manifold_steps",
        type=int,
        default=100,
        help="Steps parameter for manifold_muon.",
    )
    parser.add_argument(
        "--manifold_tol",
        type=float,
        default=1e-6,
        help="Tolerance parameter for manifold_muon.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (e.g., cuda:0, cuda:1).",
    )
    parser.add_argument(
        "--admm_rho",
        type=float,
        default=None,
        help="ADMM rho parameter for manifold_muon. If None, ADMM mode is disabled.",
    )
    args = parser.parse_args()

    # determinism flags
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    update_rules = {
        "manifold_muon": manifold_muon,
        "hyperspherical_descent": hyperspherical_descent,
        "adam": AdamW,
    }

    update = update_rules[args.update]

    # Create descriptive run name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    update_name = (
        f"{args.update}-admm-rho{args.admm_rho}"
        if args.admm_rho is not None
        else args.update
    )
    run_name = f"{update_name}-lr{args.lr}-{timestamp}"

    # Initialize wandb
    wandb.init(
        project="thinky-manifolds",
        name=run_name,
        config={
            "epochs": args.epochs,
            "initial_lr": args.lr,
            "update_rule": args.update,
            "seed": args.seed,
            "weight_decay": args.wd,
            "model_architecture": "MLP",
            "model_layers": [32 * 32 * 3, 128, 64, 10],
            "batch_size": 1024,
        },
    )

    # Add manifold_muon specific config if using that optimizer
    if args.update == "manifold_muon":
        wandb.config.update(
            {
                "manifold_alpha": args.manifold_alpha,
                "manifold_steps": args.manifold_steps,
                "manifold_tol": args.manifold_tol,
                "admm_rho": args.admm_rho,
            }
        )

    manifold_muon_params = (
        {
            "alpha": args.manifold_alpha,
            "steps": args.manifold_steps,
            "tol": args.manifold_tol,
            "admm_rho": args.admm_rho,
        }
        if args.update == "manifold_muon"
        else None
    )

    print(f"Training with: {args.update}")
    if args.admm_rho is not None:
        print(f"Using ADMM mode with rho={args.admm_rho}")
    print(
        f"Epochs: {args.epochs} --- LR: {args.lr}",
        f"--- WD: {args.wd}" if args.update == "adam" else "",
    )

    # Create dataloaders with seed for determinism
    train_loader, test_loader = get_dataloaders(args.seed)

    model, epoch_losses, epoch_times = train(
        epochs=args.epochs,
        initial_lr=args.lr,
        update=update,
        wd=args.wd,
        train_loader=train_loader,
        test_loader=test_loader,
        manifold_muon_params=manifold_muon_params,
        device=args.device,
    )
    test_acc, train_acc = eval(model, train_loader, test_loader, device=args.device)
    singular_values, norms = weight_stats(model)

    results = {
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "wd": args.wd,
        "update": args.update,
        "epoch_losses": epoch_losses,
        "epoch_times": epoch_times,
        "test_acc": test_acc,
        "train_acc": train_acc,
        "singular_values": singular_values,
        "norms": norms,
    }

    filename = f"update-{args.update}-lr-{args.lr}-wd-{args.wd}-seed-{args.seed}.pkl"
    os.makedirs("results", exist_ok=True)

    print(f"Saving results to {os.path.join('results', filename)}")
    with open(os.path.join("results", filename), "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {os.path.join('results', filename)}")

    # Finish wandb run
    wandb.finish()
