import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def combine_counts(file_paths):
    combined_data = defaultdict(float)  # Initialize a dictionary with default float values for summing counts

    # Loop through each YML file and add the counts to combined_data
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            for label, count in data.items():
                combined_data[label] += count  # Add counts for each label


    sorted_data = dict(sorted(combined_data.items(), key=lambda item: item[1], reverse=True))
    return sorted_data

colors = ["r", "b"]
alphas = [1, 1]

def make_histogram(plot_name, sorted_data_list):

    plt.figure(figsize=(10, 6))

    data_list = []

    for i, _sorted_data in enumerate(sorted_data_list):
        # Separate labels and counts for plotting
        labels = list(_sorted_data.keys())
        counts = list(_sorted_data.values())
        labels = [l.replace("0b","").replace("0j","") for l in labels]

        norm_counts = counts / np.sum(counts)

        # Create histogram
        plt.bar(labels, norm_counts, alpha=alphas[i], edgecolor=colors[i], facecolor="none", linewidth=2)
        #plt.hist(positions, bins=np.arange(len(labels) + 1) - 0.5, weights=counts,
        #         alpha=0.6, label=f"File {i+1}", color=colors[i % len(colors)], rwidth=0.9)


    plt.xlabel('Splitting Type')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plot_name}.pdf")
    plt.yscale("log")
    #plt.show()
    plt.savefig(f"{plot_name}_log.pdf")

# Replace 'data.yml' with the path to your YML file
input_count_files_22v23 = [["jet_clustering/jet-splitting-PDFs-00-08-01/all_splittings_multiplicities_2022_EE.yml",
                            "jet_clustering/jet-splitting-PDFs-00-08-01/all_splittings_multiplicities_2022_preEE.yml"],
                           ["jet_clustering/jet-splitting-PDFs-00-08-01/all_splittings_multiplicities_2023_preBPix.yml",
                            "jet_clustering/jet-splitting-PDFs-00-08-01/all_splittings_multiplicities_2023_BPix.yml",
                            ]
                           ]

sorted_data = []
for _f in input_count_files_22v23:
    sorted_data.append(combine_counts(_f))

make_histogram("jet_clustering/jet-splitting-PDFs-00-08-01/splitting_multiplicities_2022_vs_2023", sorted_data)


#
#
#


input_count_files_Run2vRun3 = [["jet_clustering/jet-splitting-PDFs-00-08-01/all_splittings_multiplicities_2022_EE.yml",
                            "jet_clustering/jet-splitting-PDFs-00-08-01/all_splittings_multiplicities_2022_preEE.yml",
                            "jet_clustering/jet-splitting-PDFs-00-08-01/all_splittings_multiplicities_2023_preBPix.yml",
                            "jet_clustering/jet-splitting-PDFs-00-08-01/all_splittings_multiplicities_2023_BPix.yml",
                            ],
                           ["jet_clustering/jet-splitting-PDFs-00-08-00/all_splittings_multiplicities_UL16_preVFP.yml",
                            "jet_clustering/jet-splitting-PDFs-00-08-00/all_splittings_multiplicities_UL16_postVFP.yml",
                            "jet_clustering/jet-splitting-PDFs-00-08-00/all_splittings_multiplicities_UL17.yml",
                            "jet_clustering/jet-splitting-PDFs-00-08-00/all_splittings_multiplicities_UL18.yml",
                            ],
                           ]

sorted_data = []
for _f in input_count_files_Run2vRun3:
    sorted_data.append(combine_counts(_f))

make_histogram("jet_clustering/jet-splitting-PDFs-00-08-01/splitting_multiplicities_RunII_vs_Run3", sorted_data)
