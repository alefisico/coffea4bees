import os, sys
import ROOT
import argparse
import logging
import json
import yaml
import numpy as np
from copy import copy
from convert_json_to_root import json_to_TH1
ROOT.gROOT.SetBatch(True)

class Channel:
    def __init__(self, SR, era, obs=-1):
        self.name = f"{SR}_{era}"
        self.SR   = SR
        self.era  =    era
        self.obs  = '%3d'%obs
    def space(self, s='  '):
        self.name = self.name+s
        self.obs  = self.obs +s


class Process:
    def __init__(self, name, index, rate=-1):
        self.name  = name
        self.title = '%3s'%name
        self.index = '%3d'%index
        self.rate  = '%3d'%rate
    def space(self, s='  '):
        self.title = self.title+s
        self.index = self.index+s
        self.rate  = self.rate +s


class Column:
    def __init__(self, channel, process, closureSysts, mcSysts):
        self.channel = copy(channel)
        self.process = copy(process)
        self.name = channel.name
        self.closure_systs = closureSysts
        self.mc_systs = mcSysts
        # self.lumi = '%s'%(str(uncert_lumi[self.channel.era]) if int(process.index)<1 else '-') # only signals have lumi uncertainty

        self.closureSysts = {}
        for nuisance in self.closure_systs:
            self.closureSysts[nuisance] = '%3s'%('1' if self.channel.SR in nuisance and self.process.name == 'mj' else '-')

        self.mcSysts = {}
        for nuisance in self.mc_systs:
            self.mcSysts[nuisance] = '%3s'%'-'
            if int(self.process.index)>0: continue
            if self.process.name == 'HH' and 'VFP'     in nuisance:                       continue # only apply 2016_*VFP systematics to ZZ and ZH
            if self.process.name != 'HH' and 'VFP' not in nuisance and   '6' in nuisance: continue # only apply 2016 systematics to HH
            if 'prefire' in nuisance and '8' in self.channel.era: continue # no prefire in 2018
            if '201' in nuisance: # years are uncorrelated
                if self.channel.era not in nuisance: continue
            self.mcSysts[nuisance] = '%3s'%'1'

        uncert_lumi_corr = {'6': '1.006', '7': '1.009', '8': '1.020'}
        uncert_lumi_1718 = {              '7': '1.006', '8': '1.002'}
        uncert_lumi_2016 = {'6': '1.010'                            }
        uncert_lumi_2017 = {              '7': '1.020'              }
        uncert_lumi_2018 = {                            '8': '1.015'}
        uncert_br = {'ZH': '1.013', 'HH': '1.025'} # https://gitlab.cern.ch/hh/naming-conventions https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR?rev=22#Higgs_2_fermions
        uncert_pdf_HH = {'HH': '1.030'} #https://gitlab.cern.ch/hh/naming-conventions
        uncert_pdf_ZH = {'ZH': '1.013'}
        uncert_pdf_ZZ = {'ZZ': '1.001'} #https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV?rev=27
        uncert_scale_ZZ = {'ZZ': '1.002'} #https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV?rev=27
        uncert_scale_ZH = {'ZH': '0.97/1.038'} #https://gitlab.cern.ch/hh/naming-conventions
        uncert_scale_HH = {'HH': '0.95/1.022'}
        uncert_alpha_s  = {'ZH': '1.009'} #https://gitlab.cern.ch/hh/naming-conventions
# all three signal processes have different production modes and so do not have shared pdf or scale nuisance parameters so they can be combined into a single parameter
        uncert_xs = {'ZZ': '1.002', 'ZH': '0.966/1.041', 'HH': '0.942/1.037'}

        self.br = uncert_br.get(self.process.name, '-')
        self.xs = uncert_xs.get(self.process.name, '-')
        self.lumi_corr = uncert_lumi_corr.get(self.channel.era, '-') if int(process.index)<1 else '-' # only signals have lumi uncertainty
        self.lumi_1718 = uncert_lumi_1718.get(self.channel.era, '-') if int(process.index)<1 else '-'
        self.lumi_2016 = uncert_lumi_2016.get(self.channel.era, '-') if int(process.index)<1 else '-'
        self.lumi_2017 = uncert_lumi_2017.get(self.channel.era, '-') if int(process.index)<1 else '-'
        self.lumi_2018 = uncert_lumi_2018.get(self.channel.era, '-') if int(process.index)<1 else '-'

        if self.process.name == 'tt': # add space for easier legibility
            self.space('  ')
            if self.channel.SR == 'hh': # add extra space for easier legibility
                self.space('  ')
                self.lumi_corr = self.lumi_corr + '    '
                self.lumi_1718 = self.lumi_1718 + '    '
                self.lumi_2016 = self.lumi_2016 + '    '
                self.lumi_2017 = self.lumi_2017 + '    '
                self.lumi_2018 = self.lumi_2018 + '    '
                self.br   = self.br   + '    '
                self.xs   = self.xs   + '    '

    def space(self, s='  '):
        self.channel.space(s)
        self.name = self.channel.name
        self.process.space(s)
        for nuisance in self.closure_systs:
            self.closureSysts[nuisance] = self.closureSysts[nuisance]+s
        for nuisance in self.mc_systs:
            self.mcSysts[nuisance] = self.mcSysts[nuisance]+s

def create_combine_root_file( file_to_convert, rebin, classifier, output_dir, plot, systematics_file, add_old_bkg=False, merge_2016=False, stat_only=False ):

    logging.info(f"Reading {file_to_convert}")
    coffea_hists = json.load(open(file_to_convert, 'r'))
    if systematics_file:
        logging.info(f"Reading {systematics_file}")
        coffea_hists_syst = json.load(open(systematics_file, 'r'))

    for iclass in classifier:

        root_hists = {}
        for channel in rebin.keys():

            ih = iclass+'.ps_'+channel
            full_histo = coffea_hists[ih]
            for iyear in coffea_hists[ih]['data'].keys():
                root_hists[channel+"_"+iyear] = {}

                ### For multijets

                root_hists[channel+"_"+iyear]['mj'] = {}
                root_hists[channel+"_"+iyear]['mj']['nominal'] = json_to_TH1(
                    coffea_hists[ih]['data'][iyear]['threeTag']['SR'], "mj_"+iyear, rebin[channel] )

                ### SR 4b
                for iprocess in coffea_hists[ih].keys():
                    if systematics_file and 'HH4b' in iprocess:
                        root_hists[channel+"_"+iyear][iprocess] = {}
                        for ivar in coffea_hists_syst[ih][iprocess][iyear].keys():
                            root_hists[channel+"_"+iyear][iprocess][ivar] = json_to_TH1(
                                coffea_hists_syst[ih][iprocess][iyear][ivar]['fourTag']['SR'], iprocess.split('4b')[0] +"_"+ivar+"_"+iyear, rebin[channel] )
                    else:
                        root_hists[channel+"_"+iyear][iprocess] = json_to_TH1(
                            coffea_hists[ih][iprocess][iyear]['fourTag']['SR'], iprocess.split('4b')[0]+"_"+iyear, rebin[channel] )


        for iy in root_hists.keys():
            root_hists[iy]['tt'] = root_hists[iy]['data'].Clone()
            root_hists[iy]['tt'].SetName('tt')
            root_hists[iy]['tt'].SetTitle('tt_'+iy)
            root_hists[iy]['tt'].Reset()
            for ip, _ in list(root_hists[iy].items()):
                if 'TTTo' in ip:
                    root_hists[iy]['tt'].Add( root_hists[iy][ip] )
                    del root_hists[iy][ip]
                elif 'data' in ip:
                    root_hists[iy]['data_obs'] = root_hists[iy][ip]
                    root_hists[iy]['data_obs'].SetName("data_obs")
                    root_hists[iy]['data_obs'].SetTitle("data_obs_"+iy)
                    del root_hists[iy][ip]
                else:
                    if isinstance(root_hists[iy][ip], ROOT.TH1F):
                        root_hists[iy][ip].SetName(ip.split("_UL")[0].split('4b')[0])
                    else:
                        for ivar, _ in root_hists[iy][ip].items():
                            if 'nominal' in ivar: root_hists[iy][ip][ivar].SetName(ip.split('4b')[0])
                            else: root_hists[iy][ip][ivar].SetName(ip.split('4b')[0]+'_'+ivar)

        if merge_2016:
            logging.info("\n Merging UL16_preVFP and UL16_postVFP")
            for iy in list(root_hists.keys()):
                if 'UL16_preVFP' in iy:
                    for ip, _ in list(root_hists[iy].items()):
                        if 'mj' in ip:
                            root_hists[iy][ip]['nominal'].Add( root_hists[iy.replace('pre', 'post')][ip]['nominal'] )
                        elif not 'HH4b' in ip:   ### AGE: temporary, HH4b is preUL
                            root_hists[iy][ip].Add( root_hists[iy.replace('pre', 'post')][ip] )
                    del root_hists[iy.replace('pre', 'post')]
                    root_hists['_'.join(iy.split('_')[:-1])] = root_hists.pop(iy)

        if add_old_bkg:
            old_bkg_file = ROOT.TFile(f'HIG-20-011/hist_{iclass}.root', 'read' )
            for channel in rebin.keys():
                for iy in ['UL16', 'UL17', 'UL18']:
                    root_hists[f"{channel}_{iy}"]['mj']['nominal'] = old_bkg_file.Get(f"hh{iy[-1]}/mj")
                    for i in ['0', '1', '2']:
                        for ivar in ['Up', 'Down']:
                            root_hists[f"{channel}_{iy}"]['mj'][f'basis{i}_bias_hh{ivar}'] = old_bkg_file.Get(f"hh{iy[-1]}/mj_basis{i}_bias_hh{ivar}")
                            root_hists[f"{channel}_{iy}"]['mj'][f'basis{i}_vari_hh{ivar}'] = old_bkg_file.Get(f"hh{iy[-1]}/mj_basis{i}_vari_hh{ivar}")

                    root_hists[f"{channel}_{iy}"]['data_obs'] = old_bkg_file.Get(f"hh{iy[-1]}/data_obs")
                    root_hists[f"{channel}_{iy}"]['tt'] = old_bkg_file.Get(f"hh{iy[-1]}/tt")


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output = output_dir+"/hists_"+iclass+('_oldbkg' if add_old_bkg else '' )+".root"

        root_file = ROOT.TFile(output, 'recreate')

        for channel in root_hists.keys():
            root_file.cd()
            try:
                directory = root_file.Get(channel)
                directory.IsZombie()
            except ReferenceError:
                directory = root_file.mkdir(channel)

            root_file.cd(channel)
            for ih_name, ih in root_hists[channel].items():
                if isinstance(ih, dict):
                    for _, ih2 in root_hists[channel][ih_name].items():
                        ih2.Write()
                else:
                    ih.Write()
        root_file.Close()

        logging.info("\n File "+output+" created.")

        #### minimal plot test
        test_hists = root_hists
        if plot:

            can = ROOT.TCanvas('can', 'can', 800,500)

            list_era = ['UL16', 'UL17'] if merge_2016 else ['UL16_preVFP', 'UL16_postVFP', 'UL17']
            for iy in list_era:
                test_hists['hh_UL18']['data_obs'].Add( test_hists['hh_'+iy]['data_obs'] )
                test_hists['hh_UL18']['mj']['nominal'].Add( test_hists['hh_'+iy]['mj']['nominal'] )
                test_hists['hh_UL18']['tt'].Add( test_hists['hh_'+iy]['tt'] )
                test_hists['hh_UL18']['HH4b'].Add( test_hists['hh_'+iy]['HH4b'] )

            stack = ROOT.THStack()
            stack.Add(test_hists['hh_UL18']['tt'])
            stack.Add(test_hists['hh_UL18']['mj']['nominal'])

            stack.Draw("histe")
            test_hists['hh_UL18']['data_obs'].Draw("histe same")
            test_hists['hh_UL18']['HH4b'].Scale( 100 )
            test_hists['hh_UL18']['HH4b'].Draw("histe same")

            test_hists['hh_UL18']['data_obs'].SetLineColor(ROOT.kRed)
            can.SaveAs(output_dir+"/test_plot_"+iclass+"_hh.png")



        #### make datacard
        metadata = yaml.safe_load(open('metadata/HH4b.yml', 'r'))

        closureSysts = [ i.replace('Up', '') for i in root_hists[next(iter(root_hists))]['mj'].keys() if 'Up' in i ]

        btagSysts = []
        juncSysts = []
        mcSysts = []
#        if mcSystsfile:
#            print('btag, trigger, JEC systematics from', mcSystsfile, classifier)
#            with open(mcSystsfile,'rb') as sfile:
#                mcSysts = pickle.load(sfile)[classifier]
#                keys = []
#                for systs in mcSysts.values():
#                    keys += systs.keys()
#                mcSysts = sorted(set(keys))
#                mcSysts = [s.replace('Up', '') for s in mcSysts if 'Up' in s and 'Total' not in s]
#                for s in mcSysts:
#                    if 'btag' in s: btagSysts.append(s)
#                    if 'junc' in s: juncSysts.append(s)

        channels = []
        for era in metadata['eras']:
            for SR in metadata['SR']:
                channels.append( Channel(SR, era) )

        processes = [Process('ZZ', -2), Process('ZH', -1), Process('HH', 0), Process('mj', 1), Process('tt', 2)]

        columns = []
        for channel in channels:
            for process in processes:
                columns.append( Column(channel, process, closureSysts, mcSysts) )

        hline = len(columns)*(4+1)+35+5+1+1
        hline = '-'*hline
        lines = []
        lines.append('imax %d number of channels'%(len(metadata['SR'])*len(metadata['eras'])))
        lines.append('jmax 4 number of processes minus one') # zz, zh, hh, mj, tt is five processes, so jmax is 4
        lines.append('kmax * number of systematics')
        lines.append(hline)
        lines.append('shapes * * '+output+' $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC')
        lines.append(hline)
        lines.append('%-35s %5s %s'%('bin',         '', ' '.join([channel.name for channel in channels])))
        lines.append('%-35s %5s %s'%('observation', '', ' '.join([channel.obs  for channel in channels])))
        lines.append(hline)
        lines.append('%-35s %5s %s'%('bin',         '', ' '.join([column.name for column in columns])))
        lines.append('%-35s %5s %s'%('process',     '', ' '.join([column.process.title for column in columns])))
        lines.append('%-35s %5s %s'%('process',     '', ' '.join([column.process.index for column in columns])))
        lines.append('%-35s %5s %s'%('rate',        '', ' '.join([column.process.rate  for column in columns])))
        lines.append(hline)
        for nuisance in closureSysts:
            lines.append('%-35s %5s %s'%(nuisance, 'shape', ' '.join([column.closureSysts[nuisance] for column in columns])))
#    for nuisance in mcSysts:
#        lines.append('%-35s %5s %s'%(nuisance, 'shape', ' '.join([column.     mcSysts[nuisance] for column in columns])))
        lines.append('%-35s %5s %s'%('BR_hbb',   'lnN', ' '.join([column.br        for column in columns])))
        lines.append('%-35s %5s %s'%('xs',       'lnN', ' '.join([column.xs        for column in columns])))
#        lines.append('%-35s %5s %s'%('lumi_corr','lnN', ' '.join([column.lumi_corr for column in columns])))
#        lines.append('%-35s %5s %s'%('lumi_1718','lnN', ' '.join([column.lumi_1718 for column in columns])))
#        lines.append('%-35s %5s %s'%('lumi_2016','lnN', ' '.join([column.lumi_2016 for column in columns])))
#        lines.append('%-35s %5s %s'%('lumi_2017','lnN', ' '.join([column.lumi_2017 for column in columns])))
#        lines.append('%-35s %5s %s'%('lumi_2018','lnN', ' '.join([column.lumi_2018 for column in columns])))
#        if not stat_only:
#            lines.append(hline)
#            lines.append('* autoMCStats 0 1 1')
#        lines.append(hline)
        if closureSysts:
            lines.append('multijet group = %s'%(' '.join(closureSysts)))
##    if mcSysts:
##        lines.append('btag     group = %s'%(' '.join(   btagSysts)))
##        lines.append('junc     group = %s'%(' '.join(   juncSysts)))
##        lines.append('trig     group = trigger_emulation')
#        lines.append('lumi     group = lumi_corr lumi_1718 lumi_2016 lumi_2017 lumi_2018')
#        lines.append('theory   group = BR_hbb xs')
#        if not stat_only:
#            lines.append('others   group = trigger_emulation lumi_corr lumi_1718 lumi_2016 lumi_2017 lumi_2018 BR_hbb xs pileup prefire %s'%(' '.join(juncSysts)))


        with open(output.replace('hists', 'combine').replace('root', 'txt'), 'w') as ofile:
            for line in lines:
                print(line)
                ofile.write(line+'\n')




if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Convert json hist to root TH1F', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        default="./datacards/", help='Output directory.')
    parser.add_argument('--histos', dest="histos", nargs="+",
                        default=[ ], help='List of histograms to convert')
    parser.add_argument('--classifier', dest="classifier", nargs="+",
                        default=["SvB_MA", "SvB"], help='Classifier to make histograms.')
    parser.add_argument('-f', '--file', dest='file_to_convert',
                        default="histos/histAll.json", help="File with coffea hists")
    parser.add_argument('--make_combine_inputs', dest='make_combine_inputs', action="store_true",
                        default=False, help="Make a combine output root files")
    parser.add_argument('--plot', dest='plot', action="store_true",
                        default=False, help="Make a test plot with root objects")
    parser.add_argument('-s', '--syst_file', dest='systematics_file',
                        default='', help="File contain systematic variations")
    parser.add_argument('--merge2016', dest='merge_2016', action="store_true",
                        default=False, help="(Temporary. Merge 2016 datasets)")
    parser.add_argument('--add_old_bkg', dest='add_old_bkg', action="store_true",
                        default=False, help="(Temporary. Add Bkgs from HIG-22-011)")
    parser.add_argument('--stat_only', dest='stat_only', action="store_true",
                        default=False, help="Create stat only inputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("\nRunning with these parameters: ")
    logging.info(args)

    rebin = { 'hh':20 }  #{'zz': 4, 'zh': 5, 'hh': 10}  # temp
    logging.info("Creating root files for combine")
    create_combine_root_file(
        args.file_to_convert,
        rebin,
        args.classifier,
        args.output_dir,
        args.plot,
        args.systematics_file,
        add_old_bkg=args.add_old_bkg,
        merge_2016=args.merge_2016,
        stat_only=args.stat_only
    )

