
import yaml
import argparse
import logging
import ROOT
from array import array
import cmsstyle as CMS
import numpy as np
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



if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert json hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', dest="output",
                        default="SvB_postfit.png", help='Output file and directory.')
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default='fitDiagnostics.root', help="Root file after fitDiagnostics")
    parser.add_argument('-s', '--signal', dest='signal',
                        default='GluGluToHHTo4B_cHHH1', help="Signal to plot")
    parser.add_argument('-m', '--metadata', dest='metadata',
                        default='stats_analysis/metadata/HH4b.yml', help="Metadata file")
    parser.add_argument('-t', '--type_of_fit', dest='type_of_fit', choices=['prefit', 'fit_b', 'fit_s'],
                        default='prefit', help="Type of fit to plot, choices: prefit, fit_b, fit_s")
    args = parser.parse_args()

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
    
    xmax = hists['TotalBkg'].GetXaxis().GetXmax()
    ymax = hists['TotalBkg'].GetMaximum()*1.2
    # Styling
    CMS.SetExtraText("Preliminary")
    iPos = 0
    CMS.SetLumi("")
    CMS.SetEnergy("13")
    CMS.ResetAdditionalInfo()
    nominal_can = CMS.cmsDiCanvas('nominal_can',0,xmax,0.1,ymax,0.8,1.2,
                                  "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                  "Events", 'Data/Pred.',
                                  square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
    nominal_can.cd(1)
    leg = CMS.cmsLeg(0.70, 0.89 - 0.05 * 4, 0.99, 0.89, textSize=0.04)

    stack = ROOT.THStack()
    CMS.cmsDrawStack(stack, leg, {'ttbar': hists[tt].Clone(), 'Multijet': hists[mj].Clone() }, data= hists['data'], palette=['#85D1FBff', '#FFDF7Fff'] )
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).GetYaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).GetYaxis().SetTitleSize(0.05)
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).Draw('AXISSAME')

    hists[signal].Scale( 100 )
    leg.AddEntry( hists[signal], 'HH4b (x100)', 'lp' )
    CMS.cmsDraw( hists[signal], 'histsame', fstyle=0, marker=1, alpha=1, lcolor=ROOT.TColor.GetColor("#e42536" ), fcolor=ROOT.TColor.GetColor("#e42536"))
    nominal_can.cd(1).SetLogy(True)

    nominal_can.cd(2)
    
    ratio = hists['data'].Clone()
    ratio.Divide( hists['TotalBkg'].Clone() )
    CMS.cmsDraw( ratio, 'P', mcolor=ROOT.kBlack )
    
    # bkg_syst = ROOT.TGraphAsymmErrors()
    # bkg_syst.Divide( hists['TotalBkg'].Clone(), hists['TotalBkg'].Clone(), 'pois' )
    bkg_syst = hists['TotalBkg'].Clone("bkg_syst")
    bkg_syst.Reset()
    for ibin in range(1, bkg_syst.GetXaxis().GetNbins()+1):
        bkg_syst.SetBinContent( ibin, 1.0 )
        bkg_syst.SetBinError( ibin, np.sqrt( hists['cov_matrix'].GetBinContent(ibin, ibin) )/hists['TotalBkg'].GetBinContent(ibin) )
    CMS.cmsDraw( bkg_syst, 'E2', fstyle=3004, fcolor=ROOT.kBlack, marker=0 )
    
    ref_line = ROOT.TLine(0, 1, 1, 1)
    CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetXaxis().SetTitleSize(0.095)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetYaxis().SetTitleSize(0.09)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetXaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetYaxis().SetTitleOffset(0.8)

    CMS.SaveCanvas( nominal_can, f"{args.output}" )