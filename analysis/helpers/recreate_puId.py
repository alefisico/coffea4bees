"""
This module provides a function to create a correctionlib object for the PUId tight working point (WP) values for Run2 UL.

The function `create_puId_correctionlib` generates a JSON file containing the PUId tight WP values for different Run2 UL datasets (UL16, UL17, UL18). These values are used for pileup jet identification in high-energy physics analyses.

Dependencies:
- `logging`: For debugging and logging messages.
- `hist`: For creating histograms to represent the PUId values.
- `correctionlib` and `correctionlib.convert`: For converting histograms into correctionlib objects and saving them as JSON files.

The generated JSON file is saved in the `data/puId/` directory.
"""

import logging
import hist
import correctionlib
import correctionlib.convert

def create_puId_correctionlib():
    """
    Create a correctionlib object with the PUId tight working point (WP) values for Run2 UL.

    This function defines the eta and pt ranges, along with the PUId tight WP values for UL16, UL17, and UL18 datasets.
    It creates a histogram to represent these values and converts it into a correctionlib object. The resulting object
    is saved as a JSON file in the `data/puId/` directory.

    The PUId tight WP values are sourced from the CMS Twiki:
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL

    Returns:
    --------
    None
    """

    # Define the eta and pt ranges
    eta_bins = [0, 2.5, 2.75, 3.0, 10.0]
    pt_bins = [0., 10., 20., 30., 40., 50., 100000.]

    # Define the table values
    puid_WP_tight = {
        "UL16" : [
                    [0.71, -0.32, -0.30, -0.22],
                    [0.71, -0.32, -0.30, -0.22],
                    [0.87, -0.08, -0.16, -0.12],
                    [0.94, 0.24, 0.05, 0.10],
                    [0.97, 0.48, 0.26, 0.29],
                    [1.00, 1.00, 1.00, 1.00],
                ],
        "UL17" : [
                    [0.77, 0.38, -0.31, -0.21 ],
                    [0.77, 0.38, -0.31, -0.21 ],
                    [0.90, 0.60, -0.12, -0.13],
                    [0.96, 0.82, 0.20, 0.09],
                    [0.98, 0.92, 0.47, 0.29],
                    [1.00, 1.00, 1.00, 1.00],
                ],
        "UL18" : [
                    [0.77, 0.38, -0.31, -0.21 ],
                    [0.77, 0.38, -0.31, -0.21 ],
                    [0.90, 0.60, -0.12, -0.13],
                    [0.96, 0.82, 0.20, 0.09],
                    [0.98, 0.92, 0.47, 0.29],
                    [1.00, 1.00, 1.00, 1.00],
                ]
    }

    # Create the histogram
    h = hist.Hist(
        hist.axis.Variable(pt_bins, name="pt", label="pT [GeV]"),
        hist.axis.Variable(eta_bins, name="eta", label="eta"),
        hist.axis.StrCategory(puid_WP_tight.keys(), name="category", label="Category")
    )

    # Fill the histogram with the table values
    for cat in puid_WP_tight.keys():
        table = puid_WP_tight[cat]
        for i, pt_bin in enumerate(pt_bins[:-1]):
            for j, eta_bin in enumerate(eta_bins[:-1]):
                logging.debug(f"pt_bin: {pt_bin}, eta_bin: {eta_bin}, cat: {cat}, value: {table[i][j]}")
                h.fill(pt=pt_bins[i], eta=eta_bins[j], category=cat, weight=table[i][j])

    h.name = 'PUID'
    h.label = 'out'

    new_puid = correctionlib.convert.from_histogram(h)

    cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        description="puId tight working point",
        corrections=[ new_puid ],
    )

    with open(f'data/puId/puid_tightWP.json', 'w') as f:
        f.write(cset.model_dump_json(indent=4))