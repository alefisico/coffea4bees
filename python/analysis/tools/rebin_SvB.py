import uproot
import numpy as np
from hist import Hist

# List of ROOT files
root_files = [
    "output/tmp/GluGluToHHTo4B_cHHH1_UL16_preVFP/SvB_MA.root",
    "output/tmp/GluGluToHHTo4B_cHHH1_UL16_postVFP/SvB_MA.root",
    "output/tmp/GluGluToHHTo4B_cHHH1_UL17/SvB_MA.chunk0.root",
    "output/tmp/GluGluToHHTo4B_cHHH1_UL17/SvB_MA.chunk1.root",
    "output/tmp/GluGluToHHTo4B_cHHH1_UL18/SvB_MA.chunk0.root",
    "output/tmp/GluGluToHHTo4B_cHHH1_UL18/SvB_MA.chunk1.root",
]

# Initialize an empty list to collect ps_hh arrays and weights
ps_hh_arrays = []
weights_arrays = []

# Loop over each ROOT file
for file in root_files:
    # Open the ROOT file
    with uproot.open(file) as f:
        # Access the Events tree
        events_tree = f["Events"]
        # Read the ps_hh array
        ps_hh_array = events_tree["ps_hh"].array()
        weights_array = events_tree["weight"].array()
        # Append the arrays to the lists
        ps_hh_arrays.append(ps_hh_array)
        weights_arrays.append(weights_array)

# Concatenate the arrays
combined_ps_hh_array = np.concatenate(ps_hh_arrays)
combined_weights_array = np.concatenate(weights_arrays)

# print(combined_weights_array[:10])
# sys.exit(0)
# Apply the threshold to the combined array
mask = combined_ps_hh_array > 0
combined_ps_hh_array = combined_ps_hh_array[mask]
combined_weights_array = combined_weights_array[mask]

# Define the fixed bin edge
fixed_bin_edge = 0.98

# Calculate the content of the fixed bin [0.97, 1]
fixed_bin_content = np.sum(combined_weights_array[combined_ps_hh_array >= fixed_bin_edge])
print(fixed_bin_content)

# Calculate the total content excluding the fixed bin
remaining_content = np.sum(combined_weights_array[combined_ps_hh_array < fixed_bin_edge])

# Determine the desired content per bin for the remaining bins
desired_content_per_bin = fixed_bin_content  # Adjust this value as needed

# Sort the combined array and weights
sorted_indices = np.argsort(combined_ps_hh_array[combined_ps_hh_array < fixed_bin_edge])
sorted_array = combined_ps_hh_array[combined_ps_hh_array < fixed_bin_edge][sorted_indices]
sorted_weights = combined_weights_array[combined_ps_hh_array < fixed_bin_edge][sorted_indices]

# Calculate the cumulative sum of the sorted weights
cumulative_sum = np.cumsum(sorted_weights)

# Calculate the bin edges based on the cumulative sum from the end
bin_edges = [1]
current_content = 0
for i, value in enumerate(reversed(sorted_array)):
    current_content += sorted_weights[-(i+1)]
    if current_content >= desired_content_per_bin:
        bin_edges.append(value)
        current_content = 0

bin_edges.append(fixed_bin_edge)
bin_edges.append(0)
bin_edges = np.unique(bin_edges)

# Create the histogram using Hist with variable binning
hist = Hist.new.Variable(bin_edges, name="ps_hh", label="ps_hh").Double()

# Fill the histogram with the combined_ps_hh_array data and weights
hist.fill(combined_ps_hh_array, weight=combined_weights_array)
print(hist)
print(hist.axes[0])
