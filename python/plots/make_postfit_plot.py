
import yaml
import argparse
import logging
import ROOT
from array import array
import cmsstyle as CMS
import numpy as np
import sys
import os
ROOT.gROOT.SetBatch(True)

def convert_tgraph_to_th1(tgraph, name="hist"):
    # Create a histogram with the same binning as the TGraphAsymmErrors
    n_points = tgraph.GetN()
    x_values = [tgraph.GetX()[i] for i in range(n_points)]
    x_values.append(x_values[-1] + (x_values[-1] - x_values[-2]))  # Add an extra bin edge
    hist = ROOT.TH1F(f"hist{name}", f"hist{name}", n_points, array('d', x_values))

    # Fill the histogram with the values from the TGraphAsymmErrors
    for i in range(n_points):
        hist.SetBinContent(i+1, tgraph.GetY()[i])
        hist.SetBinError(i+1, tgraph.GetErrorY(i))

    return hist

def filter_th2_by_labels(th2, label):
    # Find the bins that match the label
    x_bins = [x_bin for x_bin in range(1, th2.GetNbinsX() + 1) if label in th2.GetXaxis().GetBinLabel(x_bin)]
    y_bins = [y_bin for y_bin in range(1, th2.GetNbinsY() + 1) if label in th2.GetYaxis().GetBinLabel(y_bin)]

    # Create a new TH2 histogram with the filtered binning
    new_th2 = ROOT.TH2F(f"{th2.GetName()}_filtered", f"{th2.GetTitle()}_filtered",
                        len(x_bins), 0, len(x_bins),
                        len(y_bins), 0, len(y_bins))

    # Set the bin labels for the new histogram
    for i, x_bin in enumerate(x_bins):
        new_th2.GetXaxis().SetBinLabel(i + 1, th2.GetXaxis().GetBinLabel(x_bin))
    for j, y_bin in enumerate(y_bins):
        new_th2.GetYaxis().SetBinLabel(j + 1, th2.GetYaxis().GetBinLabel(y_bin))

    # Copy the content and errors of the matching bins
    for i, x_bin in enumerate(x_bins):
        for j, y_bin in enumerate(y_bins):
            new_th2.SetBinContent(i + 1, j + 1, th2.GetBinContent(x_bin, y_bin))
            new_th2.SetBinError(i + 1, j + 1, th2.GetBinError(x_bin, y_bin))

    return new_th2, len(x_bins)

if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert json hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', dest="output",
                        default="output/stat_plots/", help='Output directory.')
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default='fitDiagnostics.root', help="Root file after fitDiagnostics")
    parser.add_argument('-s', '--signal', dest='signal',
                        default='GluGluToHHTo4B_cHHH1', help="Signal to plot")
    parser.add_argument('-m', '--metadata', dest='metadata',
                        default='stats_analysis/metadata/HH4b.yml', help="Metadata file")
    parser.add_argument('-t', '--type_of_fit', dest='type_of_fit', choices=['prefit', 'fit_b', 'fit_s'],
                        default='prefit', help="Type of fit to plot, choices: prefit, fit_b, fit_s")
    parser.add_argument('--make_bkg_covariance', dest='make_bkg_covariance', action='store_true', 
                        default=False, help="Flag to make background covariance matrix")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")
    
    logging.info(f"Reading {args.metadata}")
    metadata = yaml.safe_load(open(args.metadata, 'r'))

    hists = { }
    channels = metadata['bin'] 
    mj = metadata['processes']['background']['multijet']['label']
    tt = metadata['processes']['background']['tt']['label']
    signal = metadata['processes']['signal'][args.signal]['label']
    infile = ROOT.TFile.Open(args.input_file)

    if args.make_bkg_covariance:
        CMS.SetExtraText("Preliminary")
        CMS.SetLumi("")
        CMS.SetEnergy("13")
        CMS.ResetAdditionalInfo()
        new_cov, nbins = filter_th2_by_labels(infile.Get("covariance_fit_s"), "datadriven")
        canv = CMS.cmsCanvas( "cov", 0,
            nbins,
            0,
            nbins,
            "",
            "",
            square=CMS.kSquare,
            extraSpace=0.01,
            iPos=0,
            with_z_axis=True,
        )
        pad = canv.GetPad(0)
        pad.SetLeftMargin(0.3)
        pad.SetBottomMargin(0.3)
        new_cov.Draw("colz")
        for i in range(1, new_cov.GetNbinsX() + 1):
            for j in range(1, new_cov.GetNbinsY() + 1):
                value = new_cov.GetBinContent(i, j)
                text = ROOT.TText()
                text.SetTextSize(0.02)
                text.SetTextAlign(22)  # Center alignment
                text.DrawText(i - 0.5, j - 0.5, f"{value:.2f}")
        new_cov.GetXaxis().LabelsOption("v")  # Set labels to be vertical
        # Set a new palette
        CMS.SetAlternative2DColor(new_cov, CMS.cmsStyle)
        # Allow to adjust palette position
        CMS.UpdatePalettePosition(new_cov, canv)
        output_file = f"{args.output}/bkg_covariance"
        CMS.SaveCanvas(canv, f"{output_file}.pdf", close=False)
        CMS.SaveCanvas(canv, f"{output_file}.png", close=False)
        CMS.SaveCanvas(canv, f"{output_file}.C")


    for i, ichannel in enumerate(channels):
        tmp_folder = f'shapes_{args.type_of_fit}/{ichannel}'
        if i==0:
            hists['data'] = convert_tgraph_to_th1(infile.Get(f'{tmp_folder}/data'), f'data{ichannel}')
            hists[mj] = infile.Get(f'{tmp_folder}/{mj}')
            hists[tt] = infile.Get(f'{tmp_folder}/{tt}')
            hists['TotalBkg'] = infile.Get(f'{tmp_folder}/total_background')
            hists[signal] = infile.Get(f'{tmp_folder}/{signal}')
            hists['cov_matrix'] = infile.Get(f'{tmp_folder}/total_covar')
        else: 
            hists['data'].Add( convert_tgraph_to_th1(infile.Get(f'{tmp_folder}/data'), f'data{ichannel}') )
            hists[mj].Add( infile.Get(f'{tmp_folder}/{mj}') )
            hists[tt].Add( infile.Get(f'{tmp_folder}/{tt}') )
            hists['TotalBkg'].Add( infile.Get(f'{tmp_folder}/total_background') )
            hists[signal].Add( infile.Get(f'{tmp_folder}/{signal}') )
            hists['cov_matrix'].Add( infile.Get(f'{tmp_folder}/total_covar') )

    ## Rescaling histogram
    for _, ih in hists.items():
        # ih.Rebin(2)
        ax = ih.GetXaxis()
        ax.Set( ax.GetNbins(), 0, 1.0 )
        ih.ResetStats()
    print(f"NUmber of bkg events in last bin: {hists['TotalBkg'].GetBinContent(hists['TotalBkg'].GetNbinsX())}")
    
    # Remove data points in hists['data'] that are higher than 0.5 in X
    # for bin_idx in range(1, hists['data'].GetNbinsX() + 1):
    #     if hists['data'].GetBinCenter(bin_idx) > 0.12:
    #         hists['data'].SetBinContent(bin_idx, 0)
    #         hists['data'].SetBinError(bin_idx, 0)
    
    xmax = hists['TotalBkg'].GetXaxis().GetXmax()
    ymax = hists['TotalBkg'].GetMaximum()*1.2
    # Styling
    CMS.SetExtraText("Preliminary")
    iPos = 0
    CMS.SetLumi("")
    CMS.SetEnergy("13")
    CMS.ResetAdditionalInfo()
    nominal_can = CMS.cmsDiCanvas('nominal_can',0,xmax,0.1,ymax,0.5,1.5,
                                  "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                  "Events", 'Data/Pred.',
                                  square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
    nominal_can.cd(1)
    leg = CMS.cmsLeg(0.70, 0.89 - 0.05 * 4, 0.99, 0.89, textSize=0.04)

    stack = ROOT.THStack()
    CMS.cmsDrawStack(stack, leg, {'ttbar': hists[tt], 'Multijet': hists[mj] }, data= hists['data'], palette=['#85D1FBff', '#FFDF7Fff'] )
    if 'mixed' in args.input_file: 
        leg.Clear()
        leg.AddEntry( hists[mj], 'Multijet', 'f' )
        leg.AddEntry( hists[tt], 'ttbar', 'f' )
        leg.AddEntry( hists['data'], 'Mixed-Data', 'lp' )
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).GetYaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).GetYaxis().SetTitleSize(0.05)
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).Draw('AXISSAME')

    hsignal = hists[signal].Clone("hsignal")
    hsignal.Scale( 100 )
    leg.AddEntry( hsignal, 'HH4b (x100)', 'lp' )
    CMS.cmsDraw( hsignal, 'histsame', fstyle=0, marker=1, alpha=1, lcolor=ROOT.TColor.GetColor("#e42536" ), fcolor=ROOT.TColor.GetColor("#e42536"))
    nominal_can.cd(1).SetLogy(True)

    nominal_can.cd(2)

    bkg_syst = hists['TotalBkg'].Clone("bkg_syst")
    bkg_syst.Reset()
    for ibin in range(1, bkg_syst.GetXaxis().GetNbins()+1):
        bkg_syst.SetBinContent(ibin, 1.0)
        bkg_syst.SetBinError(ibin, np.sqrt(hists['cov_matrix'].GetBinContent(ibin, ibin)) / hists['TotalBkg'].GetBinContent(ibin))
    CMS.cmsDraw( bkg_syst, 'E2', fstyle=3004, fcolor=ROOT.kBlack, marker=0 )

    print(hists[signal].GetBinContent(hists[signal].GetNbinsX()), hists['TotalBkg'].GetBinContent(hists['TotalBkg'].GetNbinsX()))
    ratio = hists['data'].Clone()
    denom = hists['TotalBkg'].Clone("denom")
    if args.type_of_fit == 'fit_s': denom.Add(hists[signal].Clone("signal"))
    print(f"Data: {ratio.GetBinContent(ratio.GetNbinsX())}, denom: {denom.GetBinContent(denom.GetNbinsX())}, ")
    ratio.Divide( denom )
    print(f"Ratio: {ratio.GetBinContent(ratio.GetNbinsX())}, ratio.GetBinError(ratio.GetNbinsX()): {ratio.GetBinError(ratio.GetNbinsX())}")
    # CMS.cmsDraw( ratio, 'PE same', mcolor=ROOT.kBlack )
    ratio.Draw("PE same")
    oldSize = ratio.GetMarkerSize()
    ratio.SetMarkerSize(0)
    ratio.DrawCopy("same e0")
    ratio.SetMarkerSize(oldSize)
    ratio.Draw("PE same")

    
    ref_line = ROOT.TLine(0, 1, 1, 1)
    CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetXaxis().SetTitleSize(0.095)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetYaxis().SetTitleSize(0.09)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetXaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetYaxis().SetTitleOffset(0.8)

    output_file = f"{args.output}/SvB_MA_postfitplots_{args.type_of_fit}"
    CMS.SaveCanvas(nominal_can, f"{output_file}.pdf", close=False )
    CMS.SaveCanvas(nominal_can, f"{output_file}.png", close=False )
    CMS.SaveCanvas(nominal_can, f"{output_file}.C" )