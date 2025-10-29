import argparse

import matplotlib.pyplot as plt
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot dual loss data from multiple run CSVs."
)
parser.add_argument(
    "--run-ids",
    type=str,
    nargs="+",
    required=True,
    help="Run IDs to plot (space-separated list, will load <run-id>.csv files)",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output filename prefix (without extension). If not specified, will auto-generate from trace info.",
)
parser.add_argument(
    "--step-divisor",
    type=float,
    default=2.0,
    help="Divide step values by this number (default: 2.0 to convert every-two to every-one)",
)
parser.add_argument(
    "--label-format",
    type=str,
    default="K={manifold_steps}, ρ={admm_rho}",
    help="Format string for legend labels using CSV columns (default: 'K={manifold_steps}, ρ={admm_rho}').",
)
parser.add_argument(
    "--title",
    type=str,
    default=None,
    help="Plot title. If not specified, will auto-generate from trace info in CSV.",
)
args = parser.parse_args()

# Set up matplotlib fonts
# Try to use Open Sans, fall back to sans-serif if not available
plt.rcParams["font.sans-serif"] = ["Open Sans", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12

# Load data from each run
dataframes = []
labels = []
trace_key = None  # Will be extracted from the first CSV

for run_id in args.run_ids:
    csv_file = f"{run_id}.csv"
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} data points from {csv_file}")

        # Extract trace key from the first CSV (all should have the same trace)
        if trace_key is None and "trace" in df.columns:
            trace_key = df["trace"].iloc[0]
            print(f"  Detected trace: {trace_key}")

        # Normalize steps: subtract minimum to start from 0, then divide
        min_step = df["step"].min()
        df["step"] = (df["step"] - min_step) / args.step_divisor
        print(
            f"  Normalized steps: subtracted {min_step}, divided by {args.step_divisor}"
        )

        dataframes.append(df)

        # Generate label from CSV metadata
        # Get the first row since all rows should have the same metadata
        row_dict = df.iloc[0].to_dict()

        # Special handling for admm_rho: if None/NaN, use "DA" instead
        admm_rho_val = row_dict.get("admm_rho")
        if pd.isna(admm_rho_val):
            admm_rho_str = "DA"
        else:
            admm_rho_str = str(admm_rho_val)

        # Convert manifold_steps to int if it's a valid number
        manifold_steps_val = row_dict.get("manifold_steps")
        if not pd.isna(manifold_steps_val):
            try:
                manifold_steps_str = str(int(manifold_steps_val))
            except (ValueError, TypeError):
                manifold_steps_str = str(manifold_steps_val)
        else:
            manifold_steps_str = "None"

        # Create aligned label with proper spacing
        # Format: "K=<val>, ρ=<val>" with alignment
        label = f"K={manifold_steps_str:<4} ρ={admm_rho_str}"

        labels.append(label)
        print(f"  Label: {label}")

    except FileNotFoundError:
        print(f"Warning: Could not find {csv_file}, skipping...")
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")

if not dataframes:
    print("\nError: No data files were successfully loaded!")
    exit(1)

print(f"\nSuccessfully loaded {len(dataframes)} run(s)")

# Determine the metric column name from the trace
if trace_key:
    metric_column_name = trace_key.split("/")[-1]  # e.g., "dual_loss"
else:
    # Try to find a metric column (anything that's not metadata)
    metadata_cols = {
        "step",
        "run_id",
        "run_name",
        "trace",
        "manifold_steps",
        "admm_rho",
        "lr",
        "update_rule",
    }
    data_cols = set(dataframes[0].columns) - metadata_cols
    if data_cols:
        metric_column_name = list(data_cols)[0]
    else:
        print("\nError: Could not determine metric column name!")
        exit(1)

print(f"Plotting metric: {metric_column_name}")

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Use colorblind-safe colors from the Tol color scheme
# Extended palette for more runs
colors = [
    "#0077BB",  # Blue
    "#CC3311",  # Red
    "#009988",  # Teal
    "#EE7733",  # Orange
    "#33BBEE",  # Cyan
    "#EE3377",  # Magenta
    "#BBBBBB",  # Grey
    "#000000",  # Black
]

# Plot each run
for i, (df, label) in enumerate(zip(dataframes, labels)):
    color = colors[i % len(colors)]
    ax.plot(
        df["step"],
        df[metric_column_name],
        color=color,
        linewidth=2.5,
        label=label,
    )

# Styling
ax.set_xlabel("Inner Loop Step", fontsize=14)
# Generate Y-axis label from metric name
y_label = " ".join(word.capitalize() for word in metric_column_name.split("_"))
ax.set_ylabel(y_label, fontsize=14)

# Set title
if args.title:
    title = args.title
elif trace_key:
    # Extract parameter name and metric name from trace
    # e.g., "fc1.weight/e0_d00/dual_loss" -> "Dual Loss: fc1.weight (Epoch 0, Outer Loop Step 0)"
    parts = trace_key.split("/")
    if len(parts) >= 3:
        param_name = parts[0]  # e.g., "fc1.weight"
        epoch_step = parts[1]  # e.g., "e0_d00"

        # Parse epoch/step format: "e0_d00" -> "Epoch 0, Outer Loop Step 0"
        import re

        match = re.match(r"e(\d+)_d(\d+)", epoch_step)
        if match:
            epoch_num = int(match.group(1))
            outer_step_num = int(match.group(2))
            location = f"Epoch {epoch_num}, Outer Loop Step {outer_step_num}"
        else:
            location = epoch_step

        # Convert metric name: split on underscore, capitalize each word
        # "dual_loss" -> "Dual Loss", "primal_residual" -> "Primal Residual"
        metric_name = " ".join(word.capitalize() for word in parts[-1].split("_"))
        title = f"{metric_name}: {param_name} ({location})"
    else:
        title = trace_key
else:
    title = "Metric Comparison"
ax.set_title(title, fontsize=16)

legend = ax.legend(fontsize=12, frameon=True, fancybox=False, edgecolor="black")
# Set monospace font for legend text to ensure alignment
for text in legend.get_texts():
    text.set_fontfamily("monospace")
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Use tight layout to prevent label cutoff
plt.tight_layout()

# Generate output filename
if args.output:
    output_prefix = args.output
elif trace_key:
    # Extract parameter name, metric name and epoch/step from trace
    # e.g., "fc1.weight/e0_d00/dual_loss" -> "fc1.weight_e0_d00_dual_loss"
    parts = trace_key.split("/")
    if len(parts) >= 3:
        param_name = parts[0].replace(".", "_")  # e.g., "fc1.weight" -> "fc1_weight"
        epoch_step = parts[1]  # e.g., "e0_d00"
        metric_name = parts[-1]  # e.g., "dual_loss"
        output_prefix = f"{param_name}_{epoch_step}_{metric_name}"
    else:
        output_prefix = "comparison"
else:
    output_prefix = "comparison"

# Save the figure
pdf_file = f"{output_prefix}.pdf"
png_file = f"{output_prefix}.png"
plt.savefig(pdf_file, dpi=300, bbox_inches="tight")
plt.savefig(png_file, dpi=300, bbox_inches="tight")

print(f"\nPlots saved to {pdf_file} and {png_file}")

# Create Open Graph version (1200x630)
fig_og, ax_og = plt.subplots(figsize=(1200 / 100, 630 / 100), dpi=100)

# Plot the data again
for i, (df, label) in enumerate(zip(dataframes, labels)):
    color = colors[i % len(colors)]
    ax_og.plot(
        df["step"],
        df[metric_column_name],
        color=color,
        linewidth=3,
        label=label,
    )

# Styling for OG image (slightly larger elements for better visibility at small sizes)
ax_og.set_xlabel("Inner Loop Step", fontsize=16)
ax_og.set_ylabel(y_label, fontsize=16)
ax_og.set_title(title, fontsize=18)
legend_og = ax_og.legend(fontsize=14, frameon=True, fancybox=False, edgecolor="black")
# Set monospace font for legend text to ensure alignment
for text in legend_og.get_texts():
    text.set_fontfamily("monospace")
ax_og.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax_og.tick_params(labelsize=12)

plt.tight_layout()
og_file = f"{output_prefix}.png"
plt.savefig(og_file, dpi=100, bbox_inches="tight")
plt.close(fig_og)

print(f"Open Graph image saved to {og_file} (1200x630)")

# Display the original plot (will show in terminal if matplotlib-backend-sixel is configured)
plt.figure(fig.number)
plt.show()
