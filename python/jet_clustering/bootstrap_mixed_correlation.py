import os
import sys
import time
import copy
import pickle
import tempfile
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import awkward as ak
import yaml
import hist

# Configure matplotlib to use temporary directory
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

# Add current directory to path for local imports
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args
import base_class.plots.helpers_make_plot_dict as plot_helpers_make_plot_dict
import base_class.plots.helpers_make_plot as plot_helpers_make_plot
import base_class.plots.iPlot_config as cfg

# Configure numpy error handling
np.seterr(divide='ignore', invalid='ignore')

def add_to_dict(event_id: Tuple[int, int], SvB: float, SvB_per_event: Dict[Tuple[int, int], List[float]]) -> None:
    """Add SvB value to the dictionary for a specific event.
    
    Args:
        event_id: Tuple of (event, run) numbers
        SvB: SvB value to add
        SvB_per_event: Dictionary to store SvB values per event
    """
    if event_id not in SvB_per_event:
        SvB_per_event[event_id] = []

    SvB_per_event[event_id].append(SvB)

def load_and_process_data() -> Tuple[Dict[str, List[float]], List[float], Dict[str, List[float]]]:
    """Load and process the mixed data.
    
    Returns:
        Tuple containing:
        - SvB_per_event: Dictionary of SvB values per event
        - weights: List of weights
        - SvB_vX: Dictionary of SvB values per vX
    """
    with open('merged_mixedData_SvB.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    vX = [f"v{i}" for i in range(15)]
    eras = ["2016", "2017", "2018"]

    SvB_per_event = {}
    weights = []
    SvB_vX = {}
    
    for _vX in vX:
        SvB_vX[_vX] = []
        for _era in eras:
            _key = f"mix_{_vX}_{_era}"
            #_key = f"mix_v0_{_era}"
            print(_key)

            SvB_vX[_vX] += loaded_data["SvB_MA_ps"][_key]

            nEvents = len(loaded_data["event"][_key])
            for _iE in range(nEvents):
                SvB = loaded_data["SvB_MA_ps"][_key][_iE]
                if SvB < 0.001: continue

                event  = loaded_data["event"]          [_key][_iE]
                run    = loaded_data["run"]            [_key][_iE]

                h1_weights = loaded_data["h1_weights"][_key][_iE]
                h2_weights = loaded_data["h2_weights"][_key][_iE]
                weights. append( (h1_weights - 1) * (h2_weights - 1) + 1 )
                #weights. append( (h1_weights + h2_weights) / 2 )

                event_id = (event, run)
                add_to_dict(event_id, SvB, SvB_per_event)

    return SvB_per_event, weights, SvB_vX

def compute_statistics(SvB_per_event: Dict[str, List[float]], weights: List[float], vX: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute statistical measures from the data.
    
    Args:
        SvB_per_event: Dictionary of SvB values per event
        weights: List of weights
        vX: List of vX values
        
    Returns:
        Tuple containing:
        - counts_ave: Average counts
        - bin_edges: Bin edges for histogram
        - bootstrap_rms: Bootstrap RMS values
        - naive_error: Naive error estimates
        - bin_centers: Bin centers
    """
    SvB_values = ak.Array(list(SvB_per_event.values()))
    count_SvB_values = ak.num(SvB_values)

    counts_tot, bin_edges = np.histogram(ak.flatten(SvB_values), bins=40)
    counts_ave = counts_tot / len(vX)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (bin_centers[1] - bin_centers[0])
    naive_error = np.sqrt(counts_tot)/16

    N_toy = len(weights[0])
    bootstrap_counts = np.zeros((N_toy, len(counts_tot)))
    weights = np.array(weights)
    
    for i in range(N_toy):
        weights_expanded = np.array([w for w, _count in zip(weights[:,i], count_SvB_values) for _ in range(_count)])
        weights_ave = weights_expanded * 1./len(vX)
        bootstrap_counts[i], _ = np.histogram(ak.flatten(SvB_values).to_numpy(), bins=bin_edges, weights=weights_ave)

    bootstrap_rms = np.sqrt(np.mean((bootstrap_counts - counts_ave)**2, axis=0))
    
    return counts_ave, bin_edges, bootstrap_rms, naive_error, bin_centers

def create_plot_dicts(counts_ave: np.ndarray, bin_edges: np.ndarray, bin_centers: np.ndarray, 
                     bootstrap_rms: np.ndarray, naive_error: np.ndarray, counts_vX: List[np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create dictionaries for plotting.
    
    Args:
        counts_ave: Average counts
        bin_edges: Bin edges
        bin_centers: Bin centers
        bootstrap_rms: Bootstrap RMS values
        naive_error: Naive error estimates
        counts_vX: List of counts per vX
        
    Returns:
        Tuple containing plot dictionaries for one vs average and average vs average plots
    """
    default_dict = {
        "tag": "fourTag", 
        "centers": bin_centers, 
        "edges": bin_edges, 
        "x_label": "SvB", 
        "under_flow": 0, 
        "over_flow": 0, 
        "edgecolor": "k", 
        "fillcolor": "k"
    }

    data_v0_dict = copy.deepcopy(default_dict)
    data_v0_dict.update({
        "label": "One Mixed Dataset",
        "values": counts_vX[0],
        "variances": counts_vX[0]
    })

    ave_dict = copy.deepcopy(default_dict)
    ave_dict.update({
        "label": "Average (Bootstrap Uncertainties)",
        "edgecolor": "k",
        "fillcolor": "#FFDF7Fff",
        "values": counts_ave,
        "variances": bootstrap_rms * bootstrap_rms
    })

    ave_dict_naive = copy.deepcopy(default_dict)
    ave_dict_naive.update({
        "label": "Average (Naive Uncertainties)",
        "edgecolor": "k",
        "fillcolor": "k",
        "values": counts_ave,
        "variances": naive_error * naive_error
    })

    return data_v0_dict, ave_dict, ave_dict_naive

def main():
    """Main function to process data and create plots."""
    # Load and process data
    SvB_per_event, weights, SvB_vX = load_and_process_data()
    vX = [f"v{i}" for i in range(15)]
    
    # Compute statistics
    counts_ave, bin_edges, bootstrap_rms, naive_error, bin_centers = compute_statistics(SvB_per_event, weights, vX)
    
    # Calculate counts per vX
    counts_vX = []
    diff_wrt_ave_vX = []
    for _vX in vX:
        _SvB = np.array(SvB_vX[_vX])
        _SvB = _SvB[_SvB > 0.001]
        _counts_vX, _ = np.histogram(_SvB, bins=bin_edges)
        counts_vX.append(_counts_vX)
        diff_wrt_ave_vX.append(_counts_vX - counts_ave)

    np.mean(counts_vX, axis=0)
    np.mean(diff_wrt_ave_vX, axis=0) # sould be all zero
    var_vX_wrt_ave = np.mean(np.square(diff_wrt_ave_vX), axis=0)
    rms_wrt_ave = np.sqrt(var_vX_wrt_ave)

    # Create plot dictionaries
    data_v0_dict, ave_dict, ave_dict_naive = create_plot_dicts(
        counts_ave, bin_edges, bin_centers, bootstrap_rms, naive_error, counts_vX
    )

    # Create and make plots
    plot_data_one = {
        "hists": {"v0": data_v0_dict},
        "stack": {"ave": ave_dict_naive},
        "ratio": {},
        "region": "SR",
        "cut": "one_vs_ave_mixed",
        "var": "SvB",
        "kwargs": {
            "debug": True,
            "doratio": False,
            "outputFolder": "./",
            "yscale": "log",
            "rlim": [0.9, 1.1]
        }
    }

    ratio_config = {
        "v0_to_ave": {
            "numerator": {"type": "hists", "key": "v0"},
            "denominator": {"type": "stack"},
            "uncertainty": "nominal",
            "color": "k",
            "marker": "o"
        }
    }
    plot_helpers_make_plot_dict.add_ratio_plots(ratio_config, plot_data_one, *{})
    plot_helpers_make_plot.make_plot_from_dict(plot_data_one)

    # Create and make average plots
    plot_data_ave = {
        "hists": {"naive": ave_dict_naive},
        "stack": {"ave": ave_dict},
        "ratio": {},
        "region": "SR",
        "cut": "ave_vs_ave_mixed",
        "var": "SvB",
        "kwargs": {
            "debug": True,
            "doratio": False,
            "outputFolder": "./",
            "yscale": "log",
            "rlim": [0.9, 1.1]
        }
    }

    ratio_config = {
        "naive_to_ave": {
            "numerator": {"type": "hists", "key": "naive"},
            "denominator": {"type": "stack"},
            "uncertainty": "nominal",
            "color": "k",
            "marker": "o",
            "bkg_err_band": None
        }
    }

    default_band_config = {"color": "k", "type": "band", "hatch": "\\\\\\"}
    band_config = copy.deepcopy(default_band_config)
    band_config.update({
        "hatch": None,
        "facecolor": "#FFDF7Fff",
        "ratio": np.ones(len(bin_centers)).tolist(),
        "error": np.sqrt(ave_dict["variances"] * np.power(ave_dict["values"], -2.0)).tolist(),
        "centers": bin_centers
    })
    plot_data_ave["ratio"]["band_boot"] = band_config

    plot_helpers_make_plot_dict.add_ratio_plots(ratio_config, plot_data_ave, *{})
    plot_helpers_make_plot.make_plot_from_dict(plot_data_ave)

if __name__ == '__main__':
    main()
