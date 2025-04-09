import os
import sys
from typing import Optional, Union, List

# Third-party imports
import hist
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import (
    makePlot, make2DPlot, load_config, load_hists, 
    read_axes_and_cuts, parse_args, print_cfg
)
import base_class.plots.iPlot_config as cfg

# Constants
DEFAULT_OUTPUT_FILE = "test.pdf"

def ls(option: str = "var", var_match: Optional[str] = None) -> None:
    """List available variables in the configuration.
    
    Args:
        option: The type of labels to list (default: "var")
        var_match: Optional string to filter variables by
    """
    for k in cfg.axisLabels[option]:
        if var_match:
            if k.find(var_match) != -1:
                print(k)
        else:
            print(k)

def info() -> None:
    """Print the current configuration."""
    print_cfg(cfg)

def examples() -> None:
    """Print example usage of the plotting functions."""
    print("examples:\n\n")
    print(
        '# Nominal plot of data and background in the a region passing a cut \n'
        'plot("v4j.mass", region="SR", cut="passPreSel")\n\n'

        '# Can get a print out of the varibales\n'
        'ls()'
        'plot("*", region="SR", cut="passPreSel")\n'
        'plot("v4j*", region="SR", cut="passPreSel")\n\n'

        '# Can add ratio\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1)\n\n'

        '# Can rebin\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4)\n\n'

        '# Can normalize\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1)\n\n'

        '# Can set logy\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, yscale="log")\n\n'

        '# Can set ranges\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, rlim=[0.5,1.5])\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, xlim=[0,1000])\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, ylim=[0,0.01])\n\n'

        '# Can overlay different regions \n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="data", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="HH4b", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="Multijet", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="TTToHadronic", doRatio=1, rebin=4)\n\n'

        '# Can overlay different cuts \n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="data", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="HH4b", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="Multijet", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="TTToHadronic", doRatio=1, rebin=4, norm=1)\n\n'

        '# Can overlay different variables \n'
        'plot(["canJet0.pt","canJet1.pt"], region="SR",cut="passPreSel",doRatio=1,process="Multijet")\n'
        'plot(["canJet0.pt","canJet1.pt","canJet2.pt","canJet3.pt"], region="SR", cut="passPreSel",doRatio=1,process="Multijet")\n\n'

        '# Can plot a single process  \n'
        'plot("v4j.mass", region="SR", cut="passPreSel",process="data")\n\n'

        '# Can overlay processes  \n'
        'plot("v4j.mass", region="SR", cut="passPreSel",norm=1,process=["data","TTTo2L2Nu","HH4b","Multijet"],doRatio=1)\n\n'

        '# Can overlay years\n'
        'plot("canJet0.pt", region="SR",cut="passPreSel",doRatio=1,process="data", year=["UL16_preVFP","UL16_postVFP","UL17","UL18"])\n'

        '# Plot 2d hists \n'
        'plot2d("quadJet_min_dr.close_vs_other_m",process="Multijet",region="SR",cut="failSvB")\n'
        'plot2d("quadJet_min_dr.close_vs_other_m",process="Multijet",region="SR",cut="failSvB",full=True)\n\n'

        '# Unsup4b plots with SB and SRSB as composite regions \n'
        'plot("v4j.mass", region="SRSB", cut="passPreSel") \n'
        'plot2d("quadJet_selected.lead_vs_subl_m",process="data3b",region="SRSB") \n'
        'plot("leadStM_selected", region="SB", cut="passPreSel", process = ["data3b","mixeddata"]) \n'
        'plot("v4j.mass", region=["SR", "SB"], cut="passPreSel", process = "data3b") \n\n'


    )

def plot(var: Union[str, List[str]] = 'selJets.pt', *, 
         cut: str = "passPreSel", 
         region: str = "SR", 
         output_file: str = DEFAULT_OUTPUT_FILE,
         **kwargs) -> Optional[tuple]:
    """Create a 1D plot of the specified variable.
    
    Args:
        var: Variable(s) to plot. Can be a string or list of strings.
        cut: Selection cut to apply (default: "passPreSel")
        region: Region to plot (default: "SR")
        output_file: Name of the output file (default: "test.pdf")
        **kwargs: Additional plotting options
        
    Returns:
        Optional tuple of (figure, axes) if debug mode is enabled
    """
    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    # Handle wildcard matching
    if isinstance(var, str) and "*" in var:
        ls(var_match=var.replace("*", ""))
        return
    if isinstance(var, list) and var[0].find("*") != -1:
        ls(var_match=var[0].replace("*", ""))
        return

    try:
        if len(cfg.hists) > 1:
            fig, ax = makePlot(cfg, var=var, cut=cut, region=region,
                             outputFolder=cfg.outputFolder, fileLabels=cfg.fileLabels, **kwargs)
        else:
            fig, ax = makePlot(cfg, var=var, cut=cut, region=region,
                             outputFolder=cfg.outputFolder, **kwargs)
    except ValueError as e:
        print(f"Error creating plot: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    try:
        fig.savefig(output_file)
        plt.close()
        os.system(f"open {output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        return

    if kwargs.get("debug", False):
        return fig, ax

def plot2d(var: str = 'quadJet_selected.lead_vs_subl_m', 
           process: str = "HH4b",
           *, 
           cut: str = "passPreSel", 
           region: str = "SR",
           output_file: str = DEFAULT_OUTPUT_FILE,
           **kwargs) -> Optional[tuple]:
    """Create a 2D plot of the specified variable.
    
    Args:
        var: Variable to plot
        process: Process to plot (default: "HH4b")
        cut: Selection cut to apply (default: "passPreSel")
        region: Region to plot (default: "SR")
        output_file: Name of the output file (default: "test.pdf")
        **kwargs: Additional plotting options
        
    Returns:
        Optional tuple of (figure, axes) if debug mode is enabled
    """
    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    if "*" in var:
        ls(var_match=var.replace("*", ""))
        return

    try:
        fig, ax = make2DPlot(cfg, process, var=var, cut=cut,
                           region=region, outputFolder=cfg.outputFolder, **kwargs)
    except Exception as e:
        print(f"Error creating 2D plot: {e}")
        return

    try:
        fig.savefig(output_file)
        plt.close()
        os.system(f"open {output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        return

    if kwargs.get("debug", False):
        return fig, ax

if __name__ == '__main__':
    args = parse_args()
    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files
    
    if cfg.outputFolder and not os.path.exists(cfg.outputFolder):
        os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    print_cfg(cfg)
