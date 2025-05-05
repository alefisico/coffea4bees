import numpy as np
from scipy.special import comb
import logging
from coffea.util import load
from copy import copy
from typing import Tuple
from base_class.plots.helpers import get_cut_dict

def getPseudoTagProbs(nj: int, f: float, e: float = 0.0, d: float = 1.0, 
                     norm: float = 1.0) -> np.ndarray:
    """
    Calculate the pseudo-tag probabilities for a given jet multiplicity.
    
    Args:
        nj: Number of jets
        f: Pseudo-tag probability 
        e: Pair enhancement factor
        d: Pair enhancement decay parameter
        norm: Normalization factor
        
    Returns:
        Array of probabilities for each number of pseudo-tags
    """
    nbt = 3  # Number of required b-tags
    nlt = nj - nbt  # Number of selected untagged jets ("light" jets)
    nPseudoTagProb = np.zeros(nlt + 1)

    for npt in range(0, nlt + 1):   # npt is the number of pseudoTags in this combination
        nt = nbt + npt
        nnt = nlt - npt  # Number of not tagged

        # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^nlt * (1-pseudoTagProb)^{nlt-npt}
        w_npt = norm * comb(nlt, npt, exact=True) * f**npt * (1 - f)**nnt

        # Apply pair enhancement for even number of tags
        if (nt % 2) == 0:
            w_npt *= 1 + e / nlt**d

        nPseudoTagProb[npt] += w_npt
        
    return nPseudoTagProb


def getCombinatoricWeight(nj: int, f: float, e: float = 0.0, d: float = 1.0, 
                         norm: float = 1.0) -> float:
    """
    Calculate the combinatoric weight for a given jet multiplicity.
    
    Args:
        nj: Number of jets
        f: Pseudo-tag probability 
        e: Pair enhancement factor
        d: Pair enhancement decay parameter
        norm: Normalization factor
        
    Returns:
        The combinatoric weight
    """
    nPseudoTagProb = getPseudoTagProbs(nj, f, e, d, norm)
    return np.sum(nPseudoTagProb[1:])


def loadROOTHists(inputFile: str) -> Tuple:
    """
    Load histograms from a ROOT file converted to coffea format.
    
    Args:
        inputFile: Path to the input ROOT file
        
    Returns:
        Tuple of histograms:
        (data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, 
         data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags)
    """
    logger.info(f"Loading histograms from ROOT file: {inputFile}")
    
    h = load(inputFile)["Hists"]

    data4b = h["data4b"]
    data3b = h["data3b"]
    tt4b = h["tt4b"]
    tt3b = h["tt3b"]
    qcd4b = h["qcd4b"]
    qcd3b = h["qcd3b"]
    data4b_nTagJets = h["data4b_nTagJets"]
    tt4b_nTagJets = h["tt4b_nTagJets"]
    qcd3b_nTightTags = h["qcd3b_nTightTags"]

    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags


def loadCoffeaHists(cfg, *, cut: str = "passPreSel", year: str = "RunII", 
                   weightRegion: str = "SB", data4bName: str = 'data', logger=None) -> Tuple:
    """
    Load histograms from coffea files.
    
    Args:
        cfg: Configuration object with histogram data
        cut: Selection cut to apply
        year: Data-taking year
        weightRegion: Region for weight calculation (e.g., "SB" for sideband)
        data4bName: Name of the 4b data process
        logger: Logger instance
        
    Returns:
        Tuple of histograms:
        (data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, 
         data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags)
    """
    if logger is None:
        import logging
        logger = logging.getLogger('JCM')

    logger.info(f"Loading coffea histograms with cut={cut}, year={year}, weightRegion={weightRegion}")
    
    # Get dictionary of cuts
    cutDict = get_cut_dict(cut, cfg.cutList)

    # Handle special cases for year and region
    year_val = sum if year == "RunII" else year
    region_selection = sum if weightRegion in ["sum", sum] else weightRegion

    region_year_dict = {
        "year": year_val,
        "region": region_selection,
    }

    # Define dictionary keys for selections
    fourTag_dict = {"tag": "fourTag"}
    threeTag_dict = {"tag": "threeTag"}

    fourTag_data_dict = {"process": data4bName} | fourTag_dict | region_year_dict | cutDict
    threeTag_data_dict = {"process": 'data'} | threeTag_dict | region_year_dict | cutDict

    ttbar_list = ['TTTo2L2Nu', 'TTToSemiLeptonic', 'TTToHadronic']
    fourTag_ttbar_dict = {"process": ttbar_list} | fourTag_dict | region_year_dict | cutDict
    threeTag_ttbar_dict = {"process": ttbar_list} | threeTag_dict | region_year_dict | cutDict

    # Find the right histograms
    hists = cfg.hists[0]['hists']
    hists_data_4b = None
    
    for _input_data in cfg.hists:
        if ('selJets_noJCM.n' in _input_data['hists'] and 
            data4bName in _input_data['hists']['selJets_noJCM.n'].axes["process"]):
            hists_data_4b = _input_data['hists']
            break
    
    if hists_data_4b is None:
        raise ValueError(f"Could not find histograms for data4bName={data4bName}")

    # Extract histograms
    data4b = hists_data_4b['selJets_noJCM.n'][fourTag_data_dict]
    data4b_nTagJets = hists_data_4b['tagJets_noJCM.n'][fourTag_data_dict]

    data3b = hists['selJets_noJCM.n'][threeTag_data_dict]
    data3b_nTagJets = hists['tagJets_loose_noJCM.n'][threeTag_data_dict]
    data3b_nTagJets_tight = hists['tagJets_noJCM.n'][threeTag_data_dict]

    tt4b = hists['selJets_noJCM.n'][fourTag_ttbar_dict][sum, :]
    tt4b_nTagJets = hists['tagJets_noJCM.n'][fourTag_ttbar_dict][sum, :]

    tt3b = hists['selJets_noJCM.n'][threeTag_ttbar_dict][sum, :]
    tt3b_nTagJets = hists['tagJets_loose_noJCM.n'][threeTag_ttbar_dict][sum, :]
    tt3b_nTagJets_tight = hists['tagJets_noJCM.n'][threeTag_ttbar_dict][sum, :]

    # Calculate QCD (data - ttbar)
    qcd4b = copy(data4b)
    qcd4b.view().value = data4b.values() - tt4b.values()
    qcd4b.view().variance = data4b.variances() + tt4b.variances()

    qcd3b = copy(data3b)
    qcd3b.view().value = data3b.values() - tt3b.values()
    qcd3b.view().variance = data3b.variances() + tt3b.variances()

    qcd3b_nTightTags = copy(data3b_nTagJets_tight)
    qcd3b_nTightTags.view().value = data3b_nTagJets_tight.values() - tt3b_nTagJets_tight.values()
    qcd3b_nTightTags.view().variance = data3b_nTagJets_tight.variances() + tt3b_nTagJets_tight.variances()

    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags


def data_from_Hist(inputHist, maxBin: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data arrays from a histogram.
    
    Args:
        inputHist: Input histogram
        maxBin: Maximum bin to extract
        
    Returns:
        Tuple of (bin_centers, values, errors)
    """
    x_centers = inputHist.axes[0].centers
    values = inputHist.values()
    errors = np.sqrt(inputHist.variances())

    # Adjust bin centers if needed
    if x_centers[0] == 0.5:
        x_centers = x_centers - 0.5

    return x_centers[0:maxBin], values[0:maxBin], errors[0:maxBin]


def prepHists(data4b, qcd3b, tt4b, data4b_nTagJets, tt4b_nTagJets) -> None:
    """
    Prepare histograms for the JCM fit by combining different components.
    
    This modifies the input histograms in-place by setting values for the 
    first 4 bins to represent the number of additional tag jets.
    
    Args:
        data4b: Data 4-tag histogram
        qcd3b: QCD 3-tag histogram
        tt4b: tt 4-tag histogram
        data4b_nTagJets: Data 4-tag tagged jets histogram
        tt4b_nTagJets: tt 4-tag tagged jets histogram
    """
    # Put the number of additional tag jets in the first 4 bins of data4b
    data4b_new_values = data4b.values()
    data4b_new_variances = data4b.variances()
    data4b_new_values[0:4] = data4b_nTagJets.values()[4:8]
    data4b_new_variances[0:4] = data4b_nTagJets.variances()[4:8]
    data4b.view().value = data4b_new_values
    data4b.view().variance = data4b_new_variances

    # Do the same for tt4b
    tt4b_new_values = tt4b.values()
    tt4b_new_variances = tt4b.variances()
    tt4b_new_values[0:4] = tt4b_nTagJets.values()[4:8]
    tt4b_new_variances[0:4] = tt4b_nTagJets.variances()[4:8]
    tt4b.view().value = tt4b_new_values
    tt4b.view().variance = tt4b_new_variances
