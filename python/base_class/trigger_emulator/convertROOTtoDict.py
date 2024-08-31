#
# python base_class/trigger_emulator/convertROOTtoDict.py
#

import ROOT
import yaml


def tgraph_to_dict(tgraph):
    """
    Convert a ROOT.TGraphAsymmErrors object to a Python dictionary.

    Parameters:
        tgraph (ROOT.TGraphAsymmErrors): The ROOT TGraphAsymmErrors object.

    Returns:
        dict: A dictionary with keys 'x', 'y', 'ex_low', 'ex_high', 'ey_low', 'ey_high'.
    """
    data = {
        'x': [],
        'eff': [],
        'eff_err': [],
        'high_bin_edge': [],
        'ex_low': [],
        'ex_high': [],
        'ey_low': [],
        'ey_high': []
    }

    for i in range(tgraph.GetN()):
        x = tgraph.GetX()[i]
        y = tgraph.GetY()[i]

        ex_low = tgraph.GetErrorXlow(i)
        ex_high = tgraph.GetErrorXhigh(i)
        ey_low = tgraph.GetErrorYlow(i)
        ey_high = tgraph.GetErrorYhigh(i)

        data['x'].append(x)

        data['ex_low'].append(ex_low)
        data['ex_high'].append(ex_high)

        data['ey_low'].append(ey_low)
        data['ey_high'].append(ey_high)

        data['high_bin_edge'].append(x + ex_high)
        data['eff'].append(y)
        data['eff_err'].append(0.5* (ey_low + ey_high) )

    return data

def tfile_to_dict(tfile):
    """
    Convert all TGraphAsymmErrors in a ROOT TFile to a dictionary of dictionaries.

    Parameters:
        tfile (ROOT.TFile): The ROOT TFile object.

    Returns:
        dict: A dictionary where each key is the name of the TGraphAsymmErrors and the value is the dictionary
              produced by the tgraph_to_dict function.
    """
    result = {}

    # Loop over all keys in the TFile
    for key in tfile.GetListOfKeys():
        obj = key.ReadObj()

        # Check if the object is a TGraphAsymmErrors
        if isinstance(obj, ROOT.TGraphAsymmErrors):
            graph_name = key.GetName()
            result[graph_name] = tgraph_to_dict(obj)

    return result



def save_to_yaml(data, filename):
    """
    Save a dictionary to a YAML file.

    Parameters:
        data (dict): The dictionary to save.
        filename (str): The name of the output YAML file.
    """
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=True)


def convert_ROOT_file_to_dict(input_file_name, base_dir="../src/TriggerEmulator/data/"):
    """
    Convert all TGraphAsymmErrors in a ROOT TFile to a dictionary of dictionaries.

    Parameters:
        tfile (ROOT.TFile): The ROOT TFile object.

    Returns:
        dict: A dictionary where each key is the name of the TGraphAsymmErrors and the value is the dictionary
              produced by the tgraph_to_dict function.
    """

    input_file = ROOT.TFile(f"{base_dir}/{input_file_name}","READ")

    result = {}


    for key in input_file.GetListOfKeys():
        obj = key.ReadObj()

        # Check if the object is a TGraphAsymmErrors
        if isinstance(obj, ROOT.TGraphAsymmErrors):
            graph_name = key.GetName()
            result[graph_name] = tgraph_to_dict(obj)

    return result

def convert_ROOT_file_to_yaml(input_file_name, out_dir="base_class/trigger_emulator/data"):
    data_dict = convert_ROOT_file_to_dict(input_file_name)

    save_to_yaml(data_dict, f'{out_dir}/{input_file_name.replace("root","yaml")}')


input_file_names = ["haddOutput_All_MC2018_11Nov_fittedTurnOns.root",
                    "haddOutput_All_Data2018_11Nov_fittedTurnOns.root",
                    "haddOutput_All_MC2017_11Nov_fittedTurnOns.root",
                    "haddOutput_All_Data2017_11Nov_fittedTurnOns.root",
                    ]

for input_file_name in input_file_names:
    print(f"Converting {input_file_name}")
    convert_ROOT_file_to_yaml(input_file_name)
