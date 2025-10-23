import matplotlib.pyplot as plt
import pandas as pd

# Set up matplotlib fonts
# Try to use Open Sans, fall back to sans-serif if not available
plt.rcParams["font.sans-serif"] = ["Open Sans", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12

# Load the data
admm_df = pd.read_csv("admm.csv")
dual_ascent_df = pd.read_csv("dual-ascent.csv")

# Convert steps to every-one (they're currently every-two)
admm_df["step"] = admm_df["step"] / 2
dual_ascent_df["step"] = dual_ascent_df["step"] / 2

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Use colorblind-safe colors from the Tol color scheme
# These are designed to be distinguishable for people with color vision deficiencies
color_admm = "#0077BB"  # Blue
color_dual_ascent = "#CC3311"  # Red

# Plot ADMM data (using Unicode rho character: ρ)
ax.plot(
    admm_df["step"],
    admm_df["dual_loss"],
    color=color_admm,
    linewidth=2.5,
    label="ADMM, ρ=16",
)

# Plot Dual Ascent data
ax.plot(
    dual_ascent_df["step"],
    dual_ascent_df["dual_loss"],
    color=color_dual_ascent,
    linewidth=2.5,
    label="Dual Ascent",
)

# Styling
ax.set_xlabel("Step", fontsize=14)
ax.set_ylabel("Dual Loss", fontsize=14)
ax.legend(fontsize=12, frameon=True, fancybox=False, edgecolor="black")
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Use tight layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig("dual_loss_comparison.pdf", dpi=300, bbox_inches="tight")
plt.savefig("dual_loss_comparison.png", dpi=300, bbox_inches="tight")

print("Plots saved to dual_loss_comparison.pdf and dual_loss_comparison.png")

# Create Open Graph version (1200x630)
fig_og, ax_og = plt.subplots(figsize=(1200 / 100, 630 / 100), dpi=100)

# Plot the data again
ax_og.plot(
    admm_df["step"],
    admm_df["dual_loss"],
    color=color_admm,
    linewidth=3,
    label="ADMM, ρ=16",
)
ax_og.plot(
    dual_ascent_df["step"],
    dual_ascent_df["dual_loss"],
    color=color_dual_ascent,
    linewidth=3,
    label="Dual Ascent",
)

# Styling for OG image (slightly larger elements for better visibility at small sizes)
ax_og.set_xlabel("Step", fontsize=16)
ax_og.set_ylabel("Dual Loss", fontsize=16)
ax_og.legend(fontsize=14, frameon=True, fancybox=False, edgecolor="black")
ax_og.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax_og.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("dual_loss_comparison_og.png", dpi=100, bbox_inches="tight")
plt.close(fig_og)

print("Open Graph image saved to dual_loss_comparison_og.png (1200x630)")

# Display the original plot (will show in terminal if matplotlib-backend-sixel is configured)
plt.figure(fig.number)
plt.show()
