import argparse
from datetime import datetime

import pandas as pd
import wandb

# Parse command line arguments
parser = argparse.ArgumentParser(description="Load wandb results filtered by datetime.")
parser.add_argument(
    "--after",
    type=str,
    default=None,
    help='Filter runs after this datetime (format: YYYYMMDD-HHMMSS, e.g., "20251006-143000")',
)
args = parser.parse_args()

# Initialize wandb API
api = wandb.Api()

# Build filter
filters = {}
if args.after:
    # Parse the datetime string (format: YYYYMMDD-HHMMSS)
    dt = datetime.strptime(args.after, "%Y%m%d-%H%M%S")
    # Convert to ISO format for wandb API
    filters["created_at"] = {"$gt": dt.isoformat()}
    print(f"Filtering runs created after: {dt}")

# Get filtered runs from the project
runs = api.runs("thinky-manifolds", filters=filters if filters else None)

# Collect data from all runs
data = []
for run in runs:
    # Get config parameters
    config = {
        "run_name": run.name,
        "manifold_steps": run.config.get("manifold_steps"),
        "admm_rho": run.config.get("admm_rho"),
        "lr": run.config.get("initial_lr"),
        "epochs": run.config.get("epochs"),
        "update_rule": run.config.get("update_rule"),
    }

    # Get final metrics
    summary = run.summary._json_dict
    config.update(
        {
            "test_accuracy": summary.get("final/test_accuracy"),
            "train_accuracy": summary.get("final/train_accuracy"),
            "test_loss": summary.get("final/test_loss"),
            "train_loss": summary.get("final/train_loss"),
        }
    )

    data.append(config)

# Convert to DataFrame
df = pd.DataFrame(data)

# Sort by manifold_steps and admm_rho for easier viewing
df = df.sort_values(["manifold_steps", "admm_rho"])

# Display the table
print(df.to_string(index=False))

# Save to CSV
df.to_csv("wandb_results.csv", index=False)
print("\nResults saved to wandb_results.csv")
