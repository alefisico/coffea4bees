import hist
import numpy as np
import os
import yaml
from hist.intervals import ratio_uncertainty
from base_class.physics.di_higgs import Coupling, ggF

epsilon = 0.001
phi = (1 + np.sqrt(5)) / 2

colors = ["xkcd:black",  "xkcd:red",    "xkcd:off green", "xkcd:blue",
          "xkcd:orange", "xkcd:violet", "xkcd:grey",      "xkcd:pink" ,
          "xkcd:pale blue",
          "xkcd:black",  "xkcd:red",    "xkcd:off green", "xkcd:blue",
          "xkcd:orange", "xkcd:violet", "xkcd:grey",      "xkcd:pink" ,
          ]


def get_value_nested_dict(nested_dict, target_key):
    """ Return the first value from mathching key from nested dict
    """
    for k, v in nested_dict.items():
        if k == target_key:
            return v


        if type(v) is dict:
            try:
                return get_value_nested_dict(v, target_key)
            except ValueError:
                continue

    raise ValueError(f"\t target_key {target_key} not in nested_dict")


def make_hist(*, edges, values, variances, x_label, under_flow, over_flow, add_flow):
    hist_obj = hist.Hist(
        hist.axis.Variable(edges, name=x_label),  # Define variable-width bins
        storage=hist.storage.Weight()           # Use Weight storage for counts and variances
    )

    if add_flow:
        values[0]  += under_flow
        values[-1] += over_flow

    hist_obj[...] = np.array(list(zip(values, variances)), dtype=[("value", "f8"), ("variance", "f8")])

    return hist_obj


def make_2d_hist(*, x_edges, y_edges, values, variances, x_label, y_label):

    # Create a 2D histogram
    hist_obj = hist.Hist(
        hist.axis.Variable(x_edges, name=x_label),  # Define the x-axis
        hist.axis.Variable(y_edges, name=y_label),  # Define the y-axis
        storage=hist.storage.Weight()          # Use Weight storage for counts and variances
    )

    # Populate the histogram with counts and variances
    hist_obj[...] = np.array(
        list(zip(np.ravel(values), np.ravel(variances))),
        dtype=[("value", "f8"), ("variance", "f8")]
    ).reshape(len(x_edges) - 1, len(y_edges) - 1)

    return hist_obj


def make_klambda_hist(kl_value, plot_data):

    kl_target = float(kl_value.replace("HH4b_kl",""))

    plot_data_0    = get_value_nested_dict(plot_data, "HH4b_kl0")
    plot_data_1    = get_value_nested_dict(plot_data, "HH4b_kl1")
    plot_data_2_45 = get_value_nested_dict(plot_data, "HH4b_kl2p45")
    plot_data_5    = get_value_nested_dict(plot_data, "HH4b_kl5")

    basis = ggF(Coupling(dict(kl=0.0), dict(kl=1.0),  dict(kl=2.45), dict(kl=5.0)))
    target_weights = basis.weight(Coupling(kl=kl_target))[0]

    w_0, w_1, w_2_45, w_5 = target_weights

    plot_data_kl = {}
    for _k in ["values", "variances", "under_flow", "over_flow"]:

        plot_data_kl[_k] =  w_0    * np.array(plot_data_0[_k])
        plot_data_kl[_k] += w_1    * np.array(plot_data_1[_k])
        plot_data_kl[_k] += w_2_45 * np.array(plot_data_2_45[_k])
        plot_data_kl[_k] += w_5    * np.array(plot_data_5[_k])

    return plot_data_kl



def savefig(fig, file_name, *args):

    args_str = []
    for _arg in args:
        if type(_arg) is list:
            args_str.append( "_vs_".join(_arg) )
        else:
            args_str.append(_arg)

    outputPath = "/".join(args_str)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    file_name = file_name if type(file_name) is str else "_vs_".join(file_name)
    file_path_and_name = outputPath + "/" + file_name.replace(".", '_').replace("/","_") + ".pdf"
    print(f"wrote pdf:  {file_path_and_name}")
    fig.savefig(file_path_and_name)
    return


def save_yaml(plot_data, var, *args):

    args_str = []
    for _arg in args:
        if type(_arg) is list:
            args_str.append( "_vs_".join(_arg) )
        else:
            args_str.append(_arg)

    outputPath = "/".join(args_str)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    varStr = var if type(var) is str else "_vs_".join(var)

    file_name = outputPath + "/" + varStr.replace(".", '_').replace("/","_") + ".yaml"
    print(f"wrote yaml:  {file_name}")

    # Write data to a YAML file
    with open(file_name, "w") as yfile:  # Use "w" for writing in text mode
        yaml.dump(plot_data, yfile, default_flow_style=False, sort_keys=False)

    return


def get_cut_dict(cut, cutList):
    cutDict = {}
    for c in cutList:
        cutDict[c] = sum
    cutDict[cut] = True
    return cutDict


def get_label(default_str, override_list, i):
    return override_list[i] if (override_list and len(override_list) > i) else default_str


def makeRatio(numValues, numVars, denValues, denVars, epsilon=0.001, **kwargs):

    ratios = numValues / denValues

    ratios[np.isnan(ratios)] = 0

    if kwargs.get("norm", False):
        numSF = np.sum(numValues, axis=0)
        denSF = np.sum(denValues, axis=0)
        ratios *= denSF / numSF

    # Set 0 and inf to nan to hide during plotting
    ratios[ratios == 0] = np.nan
    ratios[np.isinf(ratios)] = np.nan

    # if no den set to np.nan
    ratios[denValues == 0] = np.nan

    # Both num and denom. uncertianties
    # ratio_uncert = np.abs(ratios) * np.sqrt(numVars * np.power(numValues, -2.0) + denVars * np.power(denValues, -2.0 ))
    #denValues[denValues == 0] = epsilon
    numValues[numValues == 0] = epsilon
    ratio_uncert = np.abs(ratios) * np.sqrt(numVars * np.power(numValues, -2.0))
    ratio_uncert = np.nan_to_num(ratio_uncert,nan=1)

    ### https://github.com/scikit-hep/hist/blob/main/src/hist/intervals.py
    #ratio_uncert = ratio_uncertainty(
    #    num=numValues,
    #    denom=denValues,
    #    uncertainty_type=kwargs.get("uncertainty_type", "efficiency"),
    #)

    return ratios, ratio_uncert



def get_year_str(year):

    if type(year) is list:
        year_str = "_vs_".join(year)
    else:
        year_str = year.replace("UL", "20")
    return year_str

def get_region_str(region):

    if type(region) is list:
        region_str = " vs ".join(region)
    else:
        region_str = region

    return region_str

def compare_dict_keys_with_list(dict1, list2):
  """
  Compares the keys of a dictionary with the elements of a list.

  Args:
    dict1: The dictionary.
    list2: The list.

  Returns:
    A tuple containing two sets:
      - common_keys: The set of keys from the dictionary that are present in the list.
      - unique_to_dict1: The set of keys from the dictionary that are not in the list.
  """

  keys1 = set(dict1.keys())
  list2_set = set(list2)

  common_keys = keys1.intersection(list2_set)
  unique_to_dict1 = keys1.difference(list2_set)

  return common_keys, unique_to_dict1
