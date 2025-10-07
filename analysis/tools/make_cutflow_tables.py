import argparse
import csv
from coffea.util import load
from tabulate import tabulate

def sum_values_by_year(data_dict):
    dataset_sums = {}

    for dataset, cuts in data_dict.items():
        
        parts = dataset.split('_UL')
        dataset_name = parts[0]

        if dataset_name not in dataset_sums:
            dataset_sums[dataset_name] = {}

        for cut, value in cuts.items():
            if cut not in dataset_sums[dataset_name]:
                dataset_sums[dataset_name][cut] = 0
            dataset_sums[dataset_name][cut] += value

    return dataset_sums

def dict_to_markdown_table(data_dict, cutflow_type):
    headers = ["Cut"] + list(data_dict.keys())
    cuts = set(cut for cuts in data_dict.values() for cut in cuts.keys())
    rows = [[cut] + [data_dict[dataset].get(cut, "") for dataset in data_dict.keys()] for cut in cuts]
    table = tabulate(rows, headers, tablefmt="pipe")
    return f"## {cutflow_type}\n\n{table}\n"

def dict_to_latex_table(data_dict, cutflow_type):
    headers = ["Cut"] + list(data_dict.keys())
    cuts = set(cut for cuts in data_dict.values() for cut in cuts.keys())
    rows = [[cut] + [data_dict[dataset].get(cut, "") for dataset in data_dict.keys()] for cut in cuts]
    table = tabulate(rows, headers, tablefmt="latex")
    return f"\\section*{{{cutflow_type}}}\n{table}\n"

def dict_to_csv(data_dict, cutflow_type, output_file):
    headers = ["Cut"] + list(data_dict.keys())
    cuts = set(cut for cuts in data_dict.values() for cut in cuts.keys())
    rows = [[cut] + [data_dict[dataset].get(cut, "") for dataset in data_dict.keys()] for cut in cuts]
    
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([cutflow_type])
        writer.writerow(headers)
        writer.writerows(rows)
        writer.writerow([])  # Add an empty row for separation

def main(input_file, output_file, output_ext):

    with open(output_file, 'w') as f:
        for cutflow_tag in ['cutFlowFourTag', 'cutFlowFourTagUnitWeight', 'cutFlowThreeTag', 'cutFlowThreeTagUnitWeight']:
            data_dict = load(input_file)[cutflow_tag]
            dataset_sums = sum_values_by_year(data_dict)
            
            if output_ext == 'markdown':
                markdown_table = dict_to_markdown_table(dataset_sums, cutflow_tag)
                f.write(markdown_table)
                f.write("\n\n")
            elif output_ext == 'latex':
                latex_table = dict_to_latex_table(dataset_sums, cutflow_tag)
                f.write(latex_table)
                f.write("\n\n")
            elif output_ext == 'csv':
                dict_to_csv(dataset_sums, cutflow_tag, output_file)
            else:
                print(f"Unsupported output extension: {output_ext}. Printing to console instead.")
                print(dataset_sums)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a Coffea file and output year sums to a text file.")
    parser.add_argument("-i", "--input_file", required=True, help="Path to the input Coffea file")
    parser.add_argument("-e", "--output_ext", default='markdown', help="Output file extension")
    parser.add_argument("-o", "--output_file", default='cutflow.txt', help="Path to the output text file")
    
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.output_ext)