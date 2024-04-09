import os, sys
import ROOT
import argparse
import logging
import yaml
import numpy as np
ROOT.gROOT.SetBatch(True)


def yml_to_TH1( coffea_hist, iname, rebin ):
    """docstring for hist_to_root"""

    edges     = coffea_hist['edges']
    centers   = coffea_hist['centers']
    values    = coffea_hist['values']
    variances = coffea_hist['variances']

    rHist = ROOT.TH1F(iname, iname, len(centers), edges[0], edges[-1])
    rHist.Sumw2()

    for ibin in range(1, len(centers) ):
        rHist.SetBinContent(ibin, values[ibin-1])
        rHist.SetBinError(ibin, ROOT.TMath.Sqrt(variances[ibin-1]))

    rHist.Rebin( rebin )

    return rHist

def create_root_file(file_to_convert, histos, output_dir):

    coffea_hists = yaml.safe_load(open(file_to_convert, 'r'))

    root_hists = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = output_dir + "/" + (file_to_convert.split("/")
                                 [-1].replace(".yml", "")) + ".root"

    root_file = ROOT.TFile(output, 'recreate')

    for ih in coffea_hists.keys():
        for iprocess in coffea_hists[ih].keys():
            for iy in coffea_hists[ih][iprocess].keys():
                for itag in coffea_hists[ih][iprocess][iy].keys():
                    for iregion in coffea_hists[ih][iprocess][iy][itag].keys():
                        this_hist = yml_to_TH1(
                            coffea_hists[ih][iprocess][iy][itag][iregion],
                            ih.replace(".", "_") + "_" + iprocess + "_" + iy + "_" + itag + "_" + iregion,
                            1)
                        this_hist.Write()

    root_file.Close()
    logging.info("\n File " + output + " created.")


def create_combine_root_file( file_to_convert, rebin, classifier, output_dir, plot, merge_2016=False ):

    coffea_hists = yaml.safe_load(open(file_to_convert, 'r'))

    root_hists = {}
    for iclass in classifier:

        root_hists[iclass] = {}
        for channel in rebin.keys():

            ih = iclass+'.ps_'+channel
            full_histo = coffea_hists[ih]
            for iyear in coffea_hists[ih]['data'].keys():
                root_hists[iclass][channel+"_"+iyear] = {}

                ### For multijets

                root_hists[iclass][channel+"_"+iyear]['mj'] = yml_to_TH1(
                    coffea_hists[ih]['data'][iyear]['nominal']['threeTag']['SR'], "mj_"+ih.replace('.', '_')+"_passPreSel_SR_"+iyear, rebin[channel] )

                ### SR 4b
                for iprocess in coffea_hists[ih].keys():
                    root_hists[iclass][channel+"_"+iyear][iprocess] = yml_to_TH1(
                        coffea_hists[ih][iprocess][iyear]['nominal']['fourTag']['SR'], iprocess+"_"+ih.replace('.', '_')+"_passPreSel_SR_"+iyear, rebin[channel] )

        for iy in root_hists[iclass].keys():
            root_hists[iclass][iy]['tt'] = root_hists[iclass][iy]['data'].Clone()
            root_hists[iclass][iy]['tt'].SetName('tt')
            root_hists[iclass][iy]['tt'].SetTitle('tt_'+iclass+'_ps_'+iy+'_passPreSel_SR')
            root_hists[iclass][iy]['tt'].Reset()
            for ip, _ in list(root_hists[iclass][iy].items()):
                if 'TTTo' in ip:
                    root_hists[iclass][iy]['tt'].Add( root_hists[iclass][iy][ip] )
                    del root_hists[iclass][iy][ip]
                elif 'data' in ip:
                    root_hists[iclass][iy]['data_obs'] = root_hists[iclass][iy][ip]
                    root_hists[iclass][iy]['data_obs'].SetName("data_obs")
                    root_hists[iclass][iy]['data_obs'].SetTitle("data_obs_"+iclass+"_ps_"+iy+"_passPreSel_SR")
                    del root_hists[iclass][iy][ip]
                elif '4b' in ip:
                    root_hists[iclass][iy][ip].SetName( ip.split("4b")[0]  )
                else:
                    root_hists[iclass][iy][ip].SetName( ip.split("_")[0]  )

        if merge_2016:
            logging.info("\n Merging UL16_preVFP and UL16_postVFP")
            for iy in list(root_hists[iclass].keys()):
                if 'UL16_preVFP' in iy:
                    for ip, _ in list(root_hists[iclass][iy].items()):
                        root_hists[iclass][iy][ip].Add( root_hists[iclass][iy.replace('pre', 'post')][ip] )
                    del root_hists[iclass][iy.replace('pre', 'post')]
                    root_hists[iclass]['_'.join(iy.split('_')[:-1])] = root_hists[iclass].pop(iy)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output = output_dir+"/hists_"+iclass+".root"

        root_file = ROOT.TFile(output, 'recreate')

        for channel in root_hists[iclass].keys():
            root_file.cd()
            try:
                directory = root_file.Get(channel)
                directory.IsZombie()
            except ReferenceError:
                directory = root_file.mkdir(channel)

            root_file.cd(channel)
            for _, ih in root_hists[iclass][channel].items():
                ih.Write()
        root_file.Close()

        logging.info("\n File "+output+" created.")

        #### minimal plot test
        test_hists = root_hists[iclass]
        if plot:

            can = ROOT.TCanvas('can', 'can', 800,500)

            list_era = ['UL16', 'UL17'] if merge_2016 else ['UL16_preVFP', 'UL16_postVFP', 'UL17']
            for iy in list_era:
                test_hists['hh_UL18']['data_obs'].Add( test_hists['hh_'+iy]['data_obs'] )
                test_hists['hh_UL18']['mj'].Add( test_hists['hh_'+iy]['mj'] )
                test_hists['hh_UL18']['tt'].Add( test_hists['hh_'+iy]['tt'] )
                test_hists['hh_UL18']['HH4b'].Add( test_hists['hh_'+iy]['HH4b'] )

            stack = ROOT.THStack()
            stack.Add(test_hists['hh_UL18']['tt'])
            stack.Add(test_hists['hh_UL18']['mj'])

            stack.Draw("histe")
            test_hists['hh_UL18']['data_obs'].Draw("histe same")
            test_hists['hh_UL18']['HH4b'].Scale( 100 )
            test_hists['hh_UL18']['HH4b'].Draw("histe same")

            test_hists['hh_UL18']['data_obs'].SetLineColor(ROOT.kRed)
            can.SaveAs(output_dir+"/test_plot_"+iclass+"_hh.png")



if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Convert yml hist to root TH1F', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        default="./datacards/", help='Output directory.')
    parser.add_argument('--histos', dest="histos", nargs="+",
                        default=[ ], help='List of histograms to convert')
    parser.add_argument('--classifier', dest="classifier", nargs="+",
                        default=["SvB_MA", "SvB"], help='Classifier to make histograms.')
    parser.add_argument('-f', '--file', dest='file_to_convert',
                        default="histos/histAll.yml", help="File with coffea hists")
    parser.add_argument('--make_combine_inputs', dest='make_combine_inputs', action="store_true",
                        default=False, help="Make a combine output root files")
    parser.add_argument('--plot', dest='plot', action="store_true",
                        default=False, help="Make a test plot with root objects")
    parser.add_argument('--merge2016', dest='merge_2016', action="store_true",
                        default=False, help="(Temporary. Merge 2016 datasets)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("\nRunning with these parameters: ")
    logging.info(args)

    if args.make_combine_inputs:
        rebin = {'zz': 4, 'zh': 5, 'hh': 10}  # temp
        logging.info("Creating root files for combine")
        create_combine_root_file(
            args.file_to_convert,
            rebin,
            args.classifier,
            args.output_dir,
            args.plot,
            args.merge_2016)

    else:
        logging.info("Creating root files from yml")
        create_root_file(args.file_to_convert, args.histos, args.output_dir)
