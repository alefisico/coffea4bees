import argparse
import ROOT
import array
from convert_json_to_root import create_root_file

def compute_variable_binning(multijet_hist, signal_hist, threshold):
    """
       multijet_hist and signal_hist are ROOT TH1 histograms
    """

    # Sum the last elements of multijet bkg until the sum is greater than the threshold
    total_nbins = multijet_hist.GetNbinsX()
    multijet_bin = 0
    for ibin in range(total_nbins, 0, -1):
        multijet_bin = ibin
        if multijet_hist.Integral(multijet_bin, total_nbins+1) > threshold:
            break
    
    ## Check the binning of the signal asuming multijet binning threshold
    signal_binning = signal_hist.Integral(multijet_bin, total_nbins+1)
    variable_binning = [1]
    higher_bin = total_nbins+1
    for ibin in range(total_nbins, 0, -1):
        if signal_hist.Integral(ibin, higher_bin) > signal_binning:
            variable_binning.append(signal_hist.GetBinLowEdge(ibin))
            higher_bin = ibin
            continue
    variable_binning.append(signal_hist.GetBinLowEdge(1))
    variable_binning = array.array('d', variable_binning[::-1])
    print(f"Variable binning: {variable_binning}")

    return variable_binning

def rebin_histogram(hist, variable_binning):
    """
       hist is a ROOT TH1 histogram
    """
    return hist.Rebin(len(variable_binning) - 1, hist.GetName()+'_rebin', array.array('d', variable_binning))

def make_variable_binning(input_file, hist_name, threshold, output_file):
    
    # Open the ROOT file
    file = ROOT.TFile.Open(input_file)
    if not file or file.IsZombie():
        print(f"It is not a ROOT file, creating a ROOT file '{input_file.replace('.json', '.root')}'")
        create_root_file(input_file, hist_name, '/'.join(input_file.split("/")[:-1]))
        file = ROOT.TFile.Open(input_file.replace(".json", ".root"))
        
    # Retrieve the histograms
    hist_name = hist_name.replace(".", "_")
    multijet_hist = file.Get(f"{hist_name}_data_UL16_preVFP_threeTag_SR")
    signal_hist = file.Get(f"{hist_name}_GluGluToHHTo4B_cHHH1_UL16_preVFP_fourTag_SR")
    for iy in [ 'UL16_postVFP', 'UL17', 'UL18']:
        multijet_hist.Add(file.Get(f"{hist_name}_data_{iy}_threeTag_SR"))
        signal_hist.Add(file.Get(f"{hist_name}_GluGluToHHTo4B_cHHH1_{iy}_fourTag_SR"))
    
    variable_binning = compute_variable_binning(multijet_hist, signal_hist, threshold)

    # Create a new ROOT file to save the rebinned histograms
    output = ROOT.TFile(output_file, "RECREATE")

    # Rebin all histograms in the file using variable_binning
    for key in file.GetListOfKeys():
        hist = key.ReadObj()
        if isinstance(hist, ROOT.TH1) and (hist_name in hist.GetName()):
            rebinned_hist = rebin_histogram( hist, variable_binning)
            rebinned_hist.Write()
            print(f"Rebinned histogram '{hist.GetName()}'")

    # Close the ROOT file
    file.Close()
    output.Close()
    return variable_binning



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make variable binning study",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_file", default = "histsAll.root", type=str, help="Path to the input ROOT file")
    parser.add_argument("-o", "--output_file", default = "histAll_rebinned.root", type=str, help="Path to the output ROOT file")
    parser.add_argument("-n", "--hist_name", default = "SvB_MA.ps_hh_fine", type=str, help="Name of the histogram to rebin")
    parser.add_argument('-t', '--threshold', type=float, default=10.0, help='Threshold value')
    

    args = parser.parse_args()

    make_variable_binning(args.input_file, args.hist_name, args.threshold, args.output_file)