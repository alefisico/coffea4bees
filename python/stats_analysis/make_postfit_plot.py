
import yaml
import argparse
import logging
import ROOT
import cmsstyle as CMS
ROOT.gROOT.SetBatch(True)

if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert json hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', dest="output",
                        default="SvB_postfit.png", help='Output file and directory.')
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default='post_fit.root', help="Root file after PostFitShapesFromWorkspace")
    parser.add_argument('-s', '--signal', dest='signal',
                        default='GluGluToHHTo4B_cHHH1', help="Signal to plot")
    parser.add_argument('-m', '--metadata', dest='metadata',
                        default='stats_analysis/metadata/HH4b.yml', help="Metadata file")
    parser.add_argument('--do_postfit', dest='do_postfit',
                        default=False, help="Do postfit plots. If False, does prefit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")

    label = 'postfit' if args.do_postfit else 'prefit'
    
    logging.info(f"Reading {args.metadata}")
    metadata = yaml.safe_load(open(args.metadata, 'r'))

    hists = { }
    channels = metadata['bin'] #[ 'hh_UL16', 'hh_UL17', 'hh_UL18' ]
    # channels = [ 'hh6', 'hh7', 'hh8' ]
    # channels = [ 'ch1', 'ch2', 'ch3' ]
    mj = metadata['processes']['background']['multijet']['label']
    tt = metadata['processes']['background']['tt']['label']
    signal = metadata['processes']['signal'][args.signal]['label']
    infile = ROOT.TFile.Open(args.input_file)
    for i, ichannel in enumerate(channels):
        if i==0:
            hists['data'] = infile.Get(f'{ichannel}_{label}/data_obs')
            hists[mj] = infile.Get(f'{ichannel}_{label}/{mj}')
            hists[tt] = infile.Get(f'{ichannel}_{label}/{tt}')
            hists['TotalBkg'] = infile.Get(f'{ichannel}_{label}/TotalBkg')
            hists[signal] = infile.Get(f'{ichannel}_{label}/{signal}')
        else: 
            hists['data'].Add( infile.Get(f'{ichannel}_{label}/data_obs') )
            hists[mj].Add( infile.Get(f'{ichannel}_{label}/{mj}') )
            hists[tt].Add( infile.Get(f'{ichannel}_{label}/{tt}') )
            hists['TotalBkg'].Add( infile.Get(f'{ichannel}_{label}/TotalBkg') )
            hists[signal].Add( infile.Get(f'{ichannel}_{label}/{signal}') )

    ## Rescaling histogram
    for _, ih in hists.items():
        # ih.Rebin(2)
        ax = ih.GetXaxis()
        ax.Set( ax.GetNbins(), 0, 1.0 )
        ih.ResetStats()
    for i in range(hists['TotalBkg'].GetNbinsX()):
        print(hists['TotalBkg'].GetBinContent(i))
    
    xmax = hists['TotalBkg'].GetXaxis().GetXmax()
    ymax = hists['TotalBkg'].GetMaximum()*1.2
    # Styling
    CMS.SetExtraText("Preliminary")
    iPos = 0
    CMS.SetLumi("")
    CMS.SetEnergy("13")
    CMS.ResetAdditionalInfo()
    nominal_can = CMS.cmsDiCanvas('nominal_can',0,xmax,0,ymax,0.8,1.2,
                                  "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                  "Events", 'Data/Pred.',
                                  square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
    nominal_can.cd(1)
    leg = CMS.cmsLeg(0.70, 0.89 - 0.05 * 4, 0.99, 0.89, textSize=0.04)

    stack = ROOT.THStack()
    CMS.cmsDrawStack(stack, leg, {'ttbar': hists[tt].Clone(), 'Multijet': hists[mj].Clone() }, data= hists['data'], palette=['#85D1FBff', '#FFDF7Fff'] )
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).GetYaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).GetYaxis().SetTitleSize(0.05)
    CMS.fixOverlay()
    hists[signal].Scale( 1000 )
    leg.AddEntry( hists[signal], 'HH4b (x1000)', 'lp' )
    CMS.cmsDraw( hists[signal], 'hist', fstyle=0, marker=1, alpha=1, lcolor=ROOT.TColor.GetColor("#e42536" ), fcolor=ROOT.TColor.GetColor("#e42536"))

    nominal_can.cd(2)

    bkg_syst= ROOT.TGraphAsymmErrors()
    bkg_syst.Divide( hists['TotalBkg'].Clone(), hists['TotalBkg'].Clone(), 'pois' )
    CMS.cmsDraw( bkg_syst, 'F3', fstyle=3004, lcolor=ROOT.kBlack, fcolor=ROOT.kBlack  )

    bkg = hists['TotalBkg'].Clone()
    ratio = ROOT.TGraphAsymmErrors()
    ratio.Divide( hists['data'].Clone(), bkg, 'pois' )
    CMS.cmsDraw( ratio, 'P', mcolor=ROOT.kBlack )

    ref_line = ROOT.TLine(0, 1, 1, 1)
    CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetXaxis().SetTitleSize(0.095)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetYaxis().SetTitleSize(0.09)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetXaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetYaxis().SetTitleOffset(0.8)

    CMS.SaveCanvas( nominal_can, f"{args.output}" )