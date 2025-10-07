import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep

# Apply the CMS plotting style
plt.style.use(hep.style.CMS)

output="output/mu_projection_plot_ZH"

# --- Input Data ---
# Luminosity in inverse femtobarns (fb^-1)
luminosities = {
    'Run 2': 138,
    'Run 3': 60,
    'Combined': 138 + 60,
}

# Expected signal strength modifier (mu)
### ZZ
# mu_values = {
#     'Run 2': 3.73,
#     'Run 3': 3.75,
#     'Combined': 2.59,
# }

### ZH
mu_values = {
    'Run 2': 2.88,
    'Run 3': 3.12,
    'Combined': 2.07,
}

# --- Projection Calculation ---
# Use the combined result as the basis for our projection
L_ref = luminosities['Combined']
mu_ref = mu_values['Combined']

# Calculate the scaling constant k where mu = k / sqrt(L)
k = mu_ref * np.sqrt(L_ref)

# Determine the projected luminosity required to reach mu = 1
L_projection_target = k**2
mu_projection_target = 1.0

# Generate a range of luminosities for plotting the projection curve
L_curve = np.linspace(50, L_projection_target * 1.1, 500)
mu_curve = k / np.sqrt(L_curve)


# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the projection curve
ax.plot(L_curve, mu_curve, 'r--', label=r'Projection ($\mu \propto 1/\sqrt{L}$)')

# Plot the individual data points
points_l = [luminosities['Run 2'], luminosities['Run 3'], luminosities['Combined'], L_projection_target]
points_mu = [mu_values['Run 2'], mu_values['Run 3'], mu_values['Combined'], mu_projection_target]
point_labels = ['Run 2', 'Run 3', 'Combined', 'Projection']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Different colors for each point

# Scatter plot for the points
for i, label in enumerate(point_labels):
    ax.scatter(points_l[i], points_mu[i], label=label, color=colors[i], s=100, zorder=5)
    # Add text annotation for each point
    ax.text(points_l[i] * 1.05, points_mu[i], f'({points_l[i]:.0f} fb⁻¹, {points_mu[i]:.2f})', fontsize=12, verticalalignment='center')


# --- Style and Labels ---
ax.set_xlabel('Integrated Luminosity [fb⁻¹]', fontsize=18)
ax.set_ylabel(r'Expected Limit', fontsize=18)
# ax.set_title('Projection of Expected Signal Strength Limit', fontsize=20, pad=20)

# Set plot limits and grid
ax.set_xlim(0, L_projection_target * 1.1)
ax.set_ylim(0, max(points_mu) * 1.2)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(fontsize=14)

# Add CMS label using mplhep
hep.cms.label(
    "Simulation Preliminary",
    data=False,
    lumi=luminosities['Combined'],
    year="Run 2 + 3",
    ax=ax,
    fontsize=16
)

# Ensure tight layout and save the figure
plt.tight_layout()
plt.savefig(f"{output}.png", dpi=300)
plt.savefig(f"{output}.pdf")

print(f"Plot saved as {output}.png and {output}.pdf")
print(f"Projected luminosity for mu=1: {L_projection_target:.2f} fb^-1")
