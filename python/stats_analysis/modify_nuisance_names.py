import json
import os
import argparse

def main(json_file_path, output_file_path):
    # Check if the file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"The file {json_file_path} does not exist.")

    # Open and read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize an empty dictionary
    result_dict = {}

    # Iterate through the JSON data to find the "name" parameter
    for key, value in data.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and ("name" in item):
                    name = item["name"]
                    if not 'rgg' in name:
                        if 'prop_' in name:
                            tmp_name = name.split('_')
                            out_string = f"{tmp_name[2]} {tmp_name[1][3:5]} bin {tmp_name[3][-2:]}"
                        elif 'datadriven' in name:
                            tmp_name = name.split('datadriven_')[1].split('_')  
                            out_string = f"{tmp_name[-1]}: HH {tmp_name[0][:-1]} {tmp_name[0][-1]}"
                        else:
                            out_string = name
                        result_dict[name] = out_string


    # Print the resulting dictionary
    print(result_dict)
    # Save the resulting dictionary as a JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(result_dict, outfile, indent=4)

    print(f"Resulting dictionary saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a JSON file to extract names.')
    parser.add_argument('-i', '--json_file', dest='json_file', type=str, help='Path to the JSON file')
    parser.add_argument('--output_file', dest='output_file', default="nuisance_names.json", type=str, help='Path to the output JSON file')
    args = parser.parse_args()
    main(args.json_file, args.output_file)