import os, sys
import ROOT
import cmsstyle as CMS
import argparse
import logging
import json
import yaml
import numpy as np
import pickle
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

        self.closureSysts = {}
        for nuisance in self.closure_systs:
            self.closureSysts[nuisance] = '%3s'%('1' if self.channel.SR in nuisance and self.process.name == 'mj' else '-')

        self.mcSysts = {}
        for nuisance in self.mc_systs:
            self.mcSysts[nuisance] = '%3s'%'-'
            if int(self.process.index)>0: continue
            if self.process.name == 'HH' and 'VFP'     in nuisance:                       continue # only apply 2016_*VFP systematics to ZZ and ZH
            if self.process.name != 'HH' and 'VFP' not in nuisance and   '6' in nuisance: continue # only apply 2016 systematics to HH
            if '201' in nuisance: # years are uncorrelated
                if self.channel.era[-1] not in nuisance: continue
            self.mcSysts[nuisance] = '%3s'%'1'

        uncert_lumi_corr = {'UL16': '1.006', 'UL17': '1.009', 'UL18': '1.020'}
        uncert_lumi_1718 = {              'UL17': '1.006', 'UL18': '1.002'}
        uncert_lumi_2016 = {'UL16': '1.010'                            }
        uncert_lumi_2017 = {              'UL17': '1.020'              }
        uncert_lumi_2018 = {                            'UL18': '1.015'}
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


def make_trigger_syst( json_input, root_output, iyear, rebin ):

    hData = json_input['nominal']['fourTag']['SR']
    hMC = json_input['CMS_bbbb_resolved_ggf_triggerEffSFUp']['fourTag']['SR']
    nominal = json_input['CMS_bbbb_resolved_ggf_triggerEffSFDown']['fourTag']['SR']

    for num, denom, ivar in [ (nominal, hData, 'Up'), (nominal, hMC, 'Down')]:
        n_sumw, n_sumw2 = np.array(num['values']), np.array(num['variances'])
        d_sumw, d_sumw2 = np.array(denom['values']), np.array(denom['variances'])
        ratio = np.divide(n_sumw, d_sumw, out=np.ones(len(n_sumw)), where=d_sumw!=0)
        htrig = copy(hData)
        htrig['values'] *= ratio
        htrig['variances'] *= ratio*ratio
        root_output[f'CMS_bbbb_resolved_ggf_triggerEffSF{ivar}'] = json_to_TH1( htrig, "HH_"+ivar+"_"+iyear, rebin )

#    can = ROOT.TCanvas('test', 'test', 500, 500)
#    root_output['nominal'].Draw()
#    root_output['nominal'].SetLineColor(ROOT.kRed)
#    root_output[f'CMS_bbbb_resolved_ggf_triggerEffSFUp'].Draw('same')
#    root_output['CMS_bbbb_resolved_ggf_triggerEffSFUp'].SetLineColor(ROOT.kBlue)
#    root_output[f'CMS_bbbb_resolved_ggf_triggerEffSFDown'].Draw('same')
#    root_output['CMS_bbbb_resolved_ggf_triggerEffSFDown'].SetLineColor(ROOT.kMagenta)
#    can.SaveAs("test.png")


def create_combine_root_file( file_to_convert,
                             rebin,
                             classifier,
                             output_dir,
                             systematics_file,
                             bkg_systematics_file,
                             make_syst_plots=False,
                             use_preUL=False,
                             add_old_bkg=False,
                             merge_2016=False,
                             stat_only=False ):

    logging.info(f"Reading {file_to_convert}")
    coffea_hists = json.load(open(file_to_convert, 'r'))
    if systematics_file:
        logging.info(f"Reading {systematics_file}")
        coffea_hists_syst = json.load(open(systematics_file, 'r'))
    if bkg_systematics_file and not stat_only:
        logging.info(f"Reading {bkg_systematics_file}")
        bkg_syst_file = pickle.load(open(bkg_systematics_file, 'rb'))
    else:
        logging.info("Running bkg systematics needs bkg file. It is missing")

    for iclass in classifier:

        root_hists = {}
        for channel in rebin.keys():

            ih = iclass+'.ps_'+channel#+'_fine'
            full_histo = coffea_hists[ih]
            for iyear in coffea_hists[ih]['data'].keys():
                root_hists[channel+"_"+iyear] = {}

                ### For multijets

                root_hists[channel+"_"+iyear]['mj'] = {}
                root_hists[channel+"_"+iyear]['mj']['nominal'] = json_to_TH1(
                    coffea_hists[ih]['data'][iyear]['threeTag']['SR'], "mj_"+iyear, rebin[channel] )

                ### For ZH ZZ
                for iprocess in coffea_hists[ih].keys():
                    root_hists[channel+"_"+iyear][('HH' if iprocess.startswith('GluGlu') else iprocess)] = json_to_TH1(
                        coffea_hists[ih][iprocess][iyear]['fourTag']['SR'], iprocess.split('4b')[0]+"_"+iyear, rebin[channel] )

                if systematics_file and not use_preUL:
                    iprocess = 'GluGluToHHTo4B_cHHH1'
                    root_hists[channel+"_"+iyear]['HH'] = {}
                    for ivar in coffea_hists_syst[ih][iprocess][iyear].keys():
                        tmpvar = ''.join(ivar.replace('_Up', '').replace('Up','').replace('_Down','').replace('Down', '')[-2:])
                        if tmpvar.isdigit() and int(tmpvar) != int(iyear[2:4]): continue
                        if 'triggerEffSFUp' in ivar:
                            make_trigger_syst(coffea_hists_syst[ih][iprocess][iyear],
                                              root_hists[channel+"_"+iyear]['HH'],
                                              iyear, rebin[channel])
                        elif 'triggerEffSFDown' in ivar: continue
                        root_hists[channel+"_"+iyear]['HH'][ivar.replace('_Up', 'Up').replace('_Down', 'Down')] = json_to_TH1(
                            coffea_hists_syst[ih][iprocess][iyear][ivar]['fourTag']['SR'], "HH_"+ivar+"_"+iyear, rebin[channel] )


            if systematics_file and use_preUL:
                iprocess = 'HH4b'
                for iyear in coffea_hists_syst[ih]['HH4b'].keys():
                    tmpname=channel+"_"+iyear.replace('20', 'UL') + ('_preVFP' if '16' in iyear else '')
                    root_hists[tmpname][iprocess] = {}
                    for ivar in coffea_hists_syst[ih][iprocess][iyear].keys():
                        root_hists[tmpname][iprocess][ivar] = json_to_TH1(
                            coffea_hists_syst[ih][iprocess][iyear][ivar]['fourTag']['SR'], iprocess+"_"+ivar+"_"+iyear, rebin[channel] )


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
                        if ip.startswith(('mj', 'HH')) and isinstance(root_hists[iy][ip], dict):
                            for iv, _ in list(root_hists[iy][ip].items()):
                                root_hists[iy][ip][iv].Add( root_hists[iy.replace('pre', 'post')][ip][iv] )
                        elif 'HH4b' in ip:   ### AGE: temporary, HH4b is preUL
                            continue
                        else:
                            root_hists[iy][ip].Add( root_hists[iy.replace('pre', 'post')][ip] )
                    del root_hists[iy.replace('pre', 'post')]
                    root_hists['_'.join(iy.split('_')[:-1])] = root_hists.pop(iy)


        if not stat_only:
            if add_old_bkg:
                old_bkg_file = ROOT.TFile(f'HIG-20-011/hist_{iclass}.root', 'read' )
                for channel in rebin.keys():
                    for iy in ['UL16', 'UL17', 'UL18']:
                        for i in ['0', '1', '2']:
                            for ivar in ['Up', 'Down']:
                                root_hists[f"{channel}_{iy}"]['mj'][f'basis{i}_bias_hh{ivar}'] = old_bkg_file.Get(f"hh{iy[-1]}/mj_basis{i}_bias_hh{ivar}")
                                root_hists[f"{channel}_{iy}"]['mj'][f'basis{i}_vari_hh{ivar}'] = old_bkg_file.Get(f"hh{iy[-1]}/mj_basis{i}_vari_hh{ivar}")
            else:
                for channel in rebin.keys():
                    for iy in ['UL16', 'UL17', 'UL18']:
                        for ibin, ivalues in bkg_syst_file.items():
                            root_hists[f"{channel}_{iy}"]['mj'][ibin] = root_hists[f"{channel}_{iy}"]['mj']['nominal'].Clone()
                            root_hists[f"{channel}_{iy}"]['mj'][ibin].SetName(f'mj_{ibin}')
                            for i in range(len(ivalues)):
                                nom_val = root_hists[f"{channel}_{iy}"]['mj'][ibin].GetBinContent( i+1 )
                                root_hists[f"{channel}_{iy}"]['mj'][ibin].SetBinContent( i+1, nom_val*ivalues[i]  )



        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output = output_dir+"/hists_"+iclass+".root"

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


        #### make datacard
        metadata = yaml.safe_load(open('metadata/HH4b.yml', 'r'))

        closureSysts, mcSysts, juncSysts, btagSyst = [], [], [], []
        if not stat_only:
            closureSysts = [ i.replace('Up', '') for i in root_hists[next(iter(root_hists))]['mj'].keys() if i.endswith('Up') ]
            mcSysts = list(set([ s.replace('_Up', '').replace('Up', '') for c in root_hists.keys() for s in root_hists[c]['HH'].keys() if s.endswith('Up') ]))
            juncSysts = [ s for s in mcSysts if s.startswith('CMS_scale_j') ]
            btagSysts = [ s for s in mcSysts if s.startswith('CMS_btag') ]

        channels = []
        for era in metadata['eras']:
            for SR in metadata['SR']:
                channels.append( Channel(SR, era) )

        processes = [Process('HH', 0), Process('mj', 1), Process('tt', 2)]
        #processes = [Process('ZZ', -2), Process('ZH', -1), Process('HH', 0), Process('mj', 1), Process('tt', 2)]

        columns = []
        for channel in channels:
            for process in processes:
                columns.append( Column(channel, process, closureSysts, mcSysts) )

        hline = len(columns)*(4+1)+35+5+1+1
        hline = '-'*hline
        lines = []
        lines.append('imax %d number of channels'%(len(metadata['SR'])*len(metadata['eras'])))
        lines.append(f'jmax {len(processes)-1} number of processes minus one') # zz, zh, hh, mj, tt is five processes, so jmax is 4
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
        for nuisance in mcSysts:
            lines.append('%-35s %5s %s'%(nuisance, 'shape', ' '.join([column.     mcSysts[nuisance] for column in columns])))
        lines.append('%-35s %5s %s'%('BR_hbb',   'lnN', ' '.join([column.br        for column in columns])))
        lines.append('%-35s %5s %s'%('xs',       'lnN', ' '.join([column.xs        for column in columns])))
        lines.append('%-35s %5s %s'%('lumi_13TeV_corr','lnN', ' '.join([column.lumi_corr for column in columns])))
        lines.append('%-35s %5s %s'%('lumi_13TeV_1718','lnN', ' '.join([column.lumi_1718 for column in columns])))
        lines.append('%-35s %5s %s'%('lumi_13TeV_2016','lnN', ' '.join([column.lumi_2016 for column in columns])))
        lines.append('%-35s %5s %s'%('lumi_13TeV_2017','lnN', ' '.join([column.lumi_2017 for column in columns])))
        lines.append('%-35s %5s %s'%('lumi_13TeV_2018','lnN', ' '.join([column.lumi_2018 for column in columns])))
        if not stat_only:
            lines.append(hline)
            lines.append('* autoMCStats 0 1 1')
        lines.append(hline)
        if closureSysts:
            lines.append('multijet group = %s'%(' '.join(closureSysts)))
        if mcSysts:
            lines.append('btag     group = %s'%(' '.join(   btagSysts)))
            lines.append('junc     group = %s'%(' '.join(   juncSysts)))
            lines.append('trig     group = CMS_bbbb_resolved_ggf_triggerEffSF')
        lines.append('lumi     group = lumi_13TeV_corr lumi_13TeV_1718 lumi_13TeV_2016 lumi_13TeV_2017 lumi_13TeV_2018')
        lines.append('theory   group = BR_hbb xs')
        if not stat_only:
            lines.append('others   group = lumi_13TeV_corr lumi_13TeV_1718 lumi_13TeV_2016 lumi_13TeV_2017 lumi_13TeV_2018 BR_hbb xs CMS_pileup_2016 CMS_pileup_2017 CMS_pileup_2018 CMS_prefire_2016 CMS_prefire_2017 CMS_prefire_2018 %s'%(' '.join(juncSysts)))


        with open(output.replace('hists', 'combine').replace('root', 'txt'), 'w') as ofile:
            for line in lines:
                print(line)
                ofile.write(line+'\n')

        if make_syst_plots:

            if not systematics_file:
                logging.info(f'For make_syst_plots it is require to provide syst_file.')
                sys.exit(0)
            if not os.path.exists(f"{output_dir}/plots/"):
                os.makedirs(f"{output_dir}/plots/")

            # Styling
            CMS.SetExtraText("Preliminary")
            iPos = 0
            CMS.SetLumi("")
            CMS.SetEnergy("13")
            CMS.ResetAdditionalInfo()
            nominal_can = CMS.cmsDiCanvas('nominal_can',0,1,0,2500,0.8,1.2,
                                        "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                        "Events", 'Data/Pred.',
                                          square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
            nominal_can.cd(1)
            leg = CMS.cmsLeg(0.81, 0.89 - 0.05 * 7, 0.99, 0.89, textSize=0.04)

            nom_data = root_hists[next(iter(root_hists))]['data_obs'].Clone('data_obs')
            nom_data.Reset()
            nom_tt = nom_data.Clone('tt')
            nom_mj = nom_data.Clone('mj')
            nom_signal = nom_data.Clone('signal')
            for ichannel in root_hists.keys():
                nom_data.Add( root_hists[ichannel]['data_obs'] )
                nom_tt.Add( root_hists[ichannel]['tt'] )
                nom_mj.Add( root_hists[ichannel]['mj']['nominal'] )
                nom_signal.Add( root_hists[ichannel]['HH']['nominal'] )
            nom_signal.Scale( 100 )

            stack = ROOT.THStack()
            CMS.cmsDrawStack(stack, leg, {'ttbar': nom_tt, 'Multijet': nom_mj }, data= nom_data )
            #CMS.GetcmsCanvasHist(nominal_can).GetYaxis().SetTitleOffset(1.6)
            CMS.fixOverlay()

            nominal_can.cd(2)

            bkg = nom_mj.Clone()
            bkg.Add( nom_tt )
            ratio = ROOT.TGraphAsymmErrors()
            ratio.Divide( nom_data, bkg, 'pois' )
            CMS.cmsDraw( ratio, 'P', mcolor=ROOT.kBlack )

            ref_line = ROOT.TLine(0, 1, 1, 1)
            CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)

            CMS.SaveCanvas( nominal_can, f"{output_dir}/plots/{iclass}_nominal.pdf" )

            for ichannel in root_hists.keys():

                for isyst in closureSysts:
                    logging.info(f"Plotting {ichannel} {isyst}")

                    CMS.SetExtraText("Simulation Preliminary")
                    iPos = 0
                    CMS.SetLumi("")
                    CMS.SetEnergy("13")
                    CMS.ResetAdditionalInfo()
                    bkg_syst_can = CMS.cmsDiCanvas('bkg_syst_can',0,1,0,1000,0.8,1.2,
                                                "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                                "Events", 'Var/Nom',
                                                  square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
                    bkg_syst_can.cd(1)
                    leg = CMS.cmsLeg(0.55, 0.89 - 0.05 * 3, 0.99, 0.89, textSize=0.04)

                    mj_nominal = root_hists[ichannel]['mj']['nominal'].Clone()
                    mj_var_up = root_hists[ichannel]['mj'][f"{isyst}Up"]
                    mj_var_dn = root_hists[ichannel]['mj'][f"{isyst}Down"]

                    leg.AddEntry( mj_nominal, 'Nominal Multijet', 'lp' )
                    CMS.cmsDraw( mj_nominal, 'P', mcolor=ROOT.kBlack )
                    leg.AddEntry( mj_var_up, f'{isyst} Up', 'lp' )
                    CMS.cmsDraw( mj_var_up, 'hist', fstyle=0, marker=1, alpha=1, lcolor=ROOT.kBlue, fcolor=ROOT.kBlue )
                    leg.AddEntry( mj_var_dn, f'{isyst} Down', 'lp' )
                    CMS.cmsDraw( mj_var_dn, 'hist', fstyle=0,  marker=1, alpha=1, lcolor=ROOT.kRed, fcolor=ROOT.kRed )
                    CMS.fixOverlay()

                    bkg_syst_can.cd(2)

                    ratio_up = ROOT.TGraphAsymmErrors()
                    ratio_up.Divide( mj_nominal, mj_var_up, 'pois' )
                    CMS.cmsDraw( ratio_up, 'hist', fstyle=0, marker=1, alpha=1, lcolor=ROOT.kBlue, fcolor=ROOT.kBlue )
                    ratio_dn = ROOT.TGraphAsymmErrors()
                    ratio_dn.Divide( mj_nominal, mj_var_dn, 'pois' )
                    CMS.cmsDraw( ratio_dn, 'hist', fstyle=0,  marker=1, alpha=1, lcolor=ROOT.kRed, fcolor=ROOT.kRed )

                    ref_line = ROOT.TLine(0, 1, 1, 1)
                    CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)

                    CMS.SaveCanvas( bkg_syst_can, f"{output_dir}/plots/{iclass}_{isyst}_{ichannel}.pdf" )

                for isyst in root_hists[ichannel]['HH'].keys():
                    if ('nominal' in isyst) or ('Down' in isyst): continue
                    isyst = isyst.replace('Up', '')
                    logging.info(f"Plotting {ichannel} {isyst}")

                    CMS.SetExtraText("Simulation Preliminary")
                    iPos = 0
                    CMS.SetLumi("")
                    CMS.SetEnergy("13")
                    CMS.ResetAdditionalInfo()
                    mc_syst_can = CMS.cmsDiCanvas('bkg_syst_can',0,1,0,3.,0.9,1.1,
                                                "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                                "Events", 'Var/Nom',
                                                  square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
                    mc_syst_can.cd(1)
                    leg = CMS.cmsLeg(0.2, 0.89 - 0.05 * 3, 0.4, 0.89, textSize=0.04)

                    HH_nominal = root_hists[ichannel]['HH']['nominal'].Clone()
                    HH_var_up = root_hists[ichannel]['HH'][f"{isyst}Up"]
                    HH_var_dn = root_hists[ichannel]['HH'][f"{isyst}Down"]

                    leg.AddEntry( HH_nominal, 'Nominal HH', 'lp' )
                    CMS.cmsDraw( HH_nominal, 'P', mcolor=ROOT.kBlack )
                    leg.AddEntry( HH_var_up, f'{isyst} Up', 'lp' )
                    CMS.cmsDraw( HH_var_up, 'hist', fstyle=0, marker=1, alpha=1, lcolor=ROOT.kBlue, fcolor=ROOT.kBlue )
                    leg.AddEntry( HH_var_dn, f'{isyst} Down', 'lp' )
                    CMS.cmsDraw( HH_var_dn, 'hist', fstyle=0,  marker=1, alpha=1, lcolor=ROOT.kRed, fcolor=ROOT.kRed )
                    CMS.fixOverlay()

                    mc_syst_can.cd(2)

                    ratio_up = ROOT.TGraphAsymmErrors()
                    ratio_up.Divide( HH_nominal, HH_var_up, 'pois' )
                    CMS.cmsDraw( ratio_up, 'hist', fstyle=0, marker=1, alpha=1, lcolor=ROOT.kBlue )
                    ratio_dn = ROOT.TGraphAsymmErrors()
                    ratio_dn.Divide( HH_nominal, HH_var_dn, 'pois' )
                    CMS.cmsDraw( ratio_dn, 'hist', fstyle=0,  marker=1, alpha=1, lcolor=ROOT.kRed )

                    ref_line = ROOT.TLine(0, 1, 1, 1)
                    CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)

                    CMS.SaveCanvas( mc_syst_can, f"{output_dir}/plots/{iclass}_{isyst}_{ichannel}.pdf" )

                    del HH_nominal, HH_var_up, HH_var_dn, ratio_up, ratio_dn



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
    parser.add_argument('--make_syst_plots', dest='make_syst_plots', action="store_true",
                        default=False, help="Make a plots for systematics with root objects")
    parser.add_argument('-r', '--rebin', dest='rebin', type=int,
                        default=15, help="Rebin")
    parser.add_argument('-s', '--syst_file', dest='systematics_file',
                        default='', help="File contain systematic variations")
    parser.add_argument('-b', '--bkg_syst_file', dest='bkg_systematics_file',
                        default='', help="File contain background systematic variations")
    parser.add_argument('--merge2016', dest='merge_2016', action="store_true",
                        default=False, help="(Temporary. Merge 2016 datasets)")
    parser.add_argument('--use_preUL', dest='use_preUL', action="store_true",
                        default=False, help="(Temporary. Use preUL samples)")
    parser.add_argument('--add_old_bkg', dest='add_old_bkg', action="store_true",
                        default=False, help="(Temporary. Add Bkgs from HIG-22-011)")
    parser.add_argument('--stat_only', dest='stat_only', action="store_true",
                        default=False, help="Create stat only inputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("\nRunning with these parameters: ")
    logging.info(args)

    rebin = { 'hh': args.rebin }  #{'zz': 4, 'zh': 5, 'hh': 10}  # temp
    logging.info("Creating root files for combine")
    create_combine_root_file(
        args.file_to_convert,
        rebin,
        args.classifier,
        args.output_dir,
        args.systematics_file,
        args.bkg_systematics_file,
        make_syst_plots=args.make_syst_plots,
        use_preUL=args.use_preUL,
        add_old_bkg=args.add_old_bkg,
        merge_2016=args.merge_2016,
        stat_only=args.stat_only
    )

