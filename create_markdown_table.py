import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
df = pd.read_csv("wandb_results.csv")

# Rename manifold_steps to inner_loop_steps
df = df.rename(columns={"manifold_steps": "inner_loop_steps"})

# Filter the desired columns
columns_to_keep = ["run_name", "inner_loop_steps", "admm_rho", "test_accuracy", "train_accuracy"]
df_filtered = df[columns_to_keep]

# Print table WITH run names
print("=" * 80)
print("TABLE WITH RUN NAMES")
print("=" * 80)
header_with_names = "| Run Name | Inner Loop Steps | ADMM Rho | Test Accuracy | Train Accuracy |"
separator_with_names = "|:---------|:----------------:|:--------:|:-------------:|:--------------:|"

print(header_with_names)
print(separator_with_names)

for _, row in df_filtered.iterrows():
    run_name = row["run_name"]
    inner_loop_steps = row["inner_loop_steps"]
    admm_rho = row["admm_rho"]
    test_acc = row["test_accuracy"]
    train_acc = row["train_accuracy"]

    # Format the values
    inner_loop_steps_str = f"{int(inner_loop_steps)}" if pd.notna(inner_loop_steps) else "N/A"
    admm_rho_str = f"{admm_rho:.4f}" if pd.notna(admm_rho) else "N/A"
    test_acc_str = f"{test_acc:.2f}%" if pd.notna(test_acc) else "N/A"
    train_acc_str = f"{train_acc:.2f}%" if pd.notna(train_acc) else "N/A"

    print(f"| {run_name} | {inner_loop_steps_str} | {admm_rho_str} | {test_acc_str} | {train_acc_str} |")

# Print table WITHOUT run names
print("\n" + "=" * 80)
print("TABLE WITHOUT RUN NAMES")
print("=" * 80)
header_no_names = "| Inner Loop Steps | ADMM Rho | Test Accuracy | Train Accuracy |"
separator_no_names = "|:----------------:|:--------:|:-------------:|:--------------:|"

print(header_no_names)
print(separator_no_names)

for _, row in df_filtered.iterrows():
    inner_loop_steps = row["inner_loop_steps"]
    admm_rho = row["admm_rho"]
    test_acc = row["test_accuracy"]
    train_acc = row["train_accuracy"]

    # Format the values
    inner_loop_steps_str = f"{int(inner_loop_steps)}" if pd.notna(inner_loop_steps) else "N/A"
    admm_rho_str = f"{admm_rho:.4f}" if pd.notna(admm_rho) else "N/A"
    test_acc_str = f"{test_acc:.2f}%" if pd.notna(test_acc) else "N/A"
    train_acc_str = f"{train_acc:.2f}%" if pd.notna(train_acc) else "N/A"

    print(f"| {inner_loop_steps_str} | {admm_rho_str} | {test_acc_str} | {train_acc_str} |")

print("\n" + "=" * 80)
print("Copy either table above and paste it into your Jekyll blog")
print("=" * 80)

# Create visualization
print("\n" + "=" * 80)
print("CREATING PLOT...")
print("=" * 80)

# Group data by inner_loop_steps
grouped = df_filtered.groupby('inner_loop_steps')

# Define the order for x-axis (ADMM rho values)
rho_order = [2.0, 4.0, 8.0, 16.0, np.nan]
x_positions = list(range(len(rho_order)))
x_labels = ['2', '4', '8', '16', 'DA']

# Create the plot
plt.figure(figsize=(10, 6))

# Get unique inner_loop_steps values and sort them
inner_loop_steps_values = sorted([ms for ms in df_filtered['inner_loop_steps'].unique() if pd.notna(ms)])

# Colorblind-friendly colors (using Paul Tol's vibrant scheme)
# These colors are distinguishable for most types of colorblindness
colorblind_colors = [
    '#0077BB',  # Blue
    '#33BBEE',  # Cyan
    '#009988',  # Teal
    '#EE7733',  # Orange
    '#CC3311',  # Red
    '#EE3377',  # Magenta
    '#BBBBBB',  # Grey
]

# Different marker styles for each curve
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

# Plot each inner_loop_steps group
for idx, inner_loop_steps_val in enumerate(inner_loop_steps_values):
    group_data = df_filtered[df_filtered['inner_loop_steps'] == inner_loop_steps_val]
    color = colorblind_colors[idx % len(colorblind_colors)]
    marker = markers[idx % len(markers)]

    if inner_loop_steps_val == 0:
        # For inner_loop_steps = 0, draw a horizontal dashed line
        test_acc = group_data['test_accuracy'].iloc[0]
        plt.axhline(y=test_acc, linestyle='--', color=color,
                   label=f'Steps = {int(inner_loop_steps_val)}', linewidth=2.5)
    else:
        # For other inner_loop_steps, plot points for each rho value
        y_values = []
        x_vals = []

        for i, rho in enumerate(rho_order):
            if pd.isna(rho):
                # Handle N/A case (dual ascent - no ADMM rho)
                row = group_data[pd.isna(group_data['admm_rho'])]
            else:
                row = group_data[group_data['admm_rho'] == rho]

            if not row.empty:
                y_values.append(row['test_accuracy'].iloc[0])
                x_vals.append(i)

        if y_values:
            plt.plot(x_vals, y_values, marker=marker, linestyle='-',
                    color=color, label=f'Steps = {int(inner_loop_steps_val)}',
                    linewidth=2.5, markersize=10, markeredgewidth=1.5,
                    markeredgecolor='white')

plt.xlabel('ADMM Rho', fontsize=12, fontweight='bold')
plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Test Accuracy vs ADMM Rho for Different Inner Loop Steps', fontsize=14, fontweight='bold')
plt.xticks(x_positions, x_labels)
plt.legend(loc='best', fontsize=10, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save the plot
plt.savefig('manifold_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'manifold_accuracy_comparison.png'")
plt.show()
