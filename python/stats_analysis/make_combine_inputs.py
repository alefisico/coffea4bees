from curses.ascii import isdigit
import os, sys
from typing import OrderedDict
import ROOT
import argparse
import logging
import json
import yaml
import numpy as np
import pickle
import pandas as pd
from copy import copy, deepcopy
from convert_json_to_root import json_to_TH1
ROOT.gROOT.SetBatch(True)

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
                             metadata_file='metadata/HH4b.yml',
                             make_syst_plots=False,
                             use_preUL=False,
                             add_old_bkg=False,
                             stat_only=False ):

    logging.info(f"Reading {metadata_file}")
    metadata = yaml.safe_load(open(metadata_file, 'r'))
    metadata['processes']['all'] = { **metadata['processes']['signal'], **metadata['processes']['background'] }
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
        mcSysts, closureSysts = [], []
        ih = iclass+'.ps_hh'
        for iyear in coffea_hists[ih]['data'].keys():
            root_hists[iyear] = {}

            ### For multijets

            root_hists[iyear]['multijet'] = {}
            root_hists[iyear]['multijet']['nominal'] = json_to_TH1(
                coffea_hists[ih]['data'][iyear]['threeTag']['SR'], 'multijet_'+iyear+iclass, rebin )

            ### For ZH ZZ
            for iprocess in coffea_hists[ih].keys():
                if iprocess not in metadata['processes']['signal']:
                    root_hists[iyear][iprocess] = json_to_TH1( coffea_hists[ih][iprocess][iyear]['fourTag']['SR'], 
                                                              f'{iprocess.split("4b")[0]}_{iyear}', rebin )

            if systematics_file and not use_preUL:
                for iprocess in metadata['processes']['signal']:

                    root_hists[iyear][iprocess] = {}
                    for ivar in coffea_hists_syst[ih][iprocess][iyear].keys():

                        ## renaming syst
                        if 'prefire' in ivar: namevar = ivar.replace("CMS_prefire", 'CMS_l1_ecal_prefiring')
                        else: namevar = ivar
                        namevar = namevar.replace('_Up', 'Up').replace('_Down', 'Down')

                        ### check for dedicated JESUnc per year, if not conitnue
                        tmpvar = namevar.replace('Up','').replace('Down', '')
                        if tmpvar not in mcSysts and not 'nominal' in tmpvar: mcSysts.append( tmpvar )
                        tmpvar = ''.join(tmpvar[-2:])
                        if tmpvar.isdigit() and int(tmpvar) != int(iyear[2:4]): continue

                        ### trigger efficiency
                        if 'triggerEffSFUp' in namevar:
                            make_trigger_syst(coffea_hists_syst[ih][iprocess][iyear],
                                              root_hists[iyear][iprocess],
                                              iyear, rebin)
                        elif 'triggerEffSFDown' in namevar: continue

                        root_hists[iyear][iprocess][namevar] = json_to_TH1(
                                                        coffea_hists_syst[ih][iprocess][iyear][ivar]['fourTag']['SR'], 
                                                        f'{iprocess}_{ivar}_{iyear}', rebin )

        if systematics_file and use_preUL:
            iprocess = 'HH4b'
            for iyear in coffea_hists_syst[ih]['HH4b'].keys():
                tmpname=iyear.replace('20', 'UL') + ('_preVFP' if '16' in iyear else '')
                root_hists[tmpname][iprocess] = {}
                for ivar in coffea_hists_syst[ih][iprocess][iyear].keys():
                    root_hists[tmpname][iprocess][ivar] = json_to_TH1(
                        coffea_hists_syst[ih][iprocess][iyear][ivar]['fourTag']['SR'], iprocess+"_"+ivar+"_"+iyear, rebin )


        if "UL16_preVFP" not in metadata['bin']:
            logging.info("\n Merging UL16_preVFP and UL16_postVFP")
            for iy in list(root_hists.keys()):
                if 'UL16_preVFP' in iy:
                    for ip, _ in list(root_hists[iy].items()):
                        if isinstance(root_hists[iy][ip], dict):
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
                old_bkg_file = ROOT.TFile(f'HIG-22-011/hist_{iclass}.root', 'read' )
                for channel in metadata['bin']:
                    for iy in ['UL16', 'UL17', 'UL18']:
                        for i in ['0', '1', '2']:
                            for ivar in ['Up', 'Down']:
                                root_hists[f"{channel}_{iy}"]['multijet'][f'basis{i}_bias_hh{ivar}'] = old_bkg_file.Get(f"hh{iy[-1]}/mj_basis{i}_bias_hh{ivar}")
                                root_hists[f"{channel}_{iy}"]['multijet'][f'basis{i}_vari_hh{ivar}'] = old_bkg_file.Get(f"hh{iy[-1]}/mj_basis{i}_vari_hh{ivar}")
            else:
                for channel in metadata['bin']:
                    for ibin, ivalues in bkg_syst_file.items():
                        bkg_name_syst = f"CMS_bbbb_resolved_bkg_datadriven_{ibin.replace('_hh', '').replace('vari', 'variance')}"
                        root_hists[channel]['multijet'][bkg_name_syst] = root_hists[channel]['multijet']['nominal'].Clone()
                        root_hists[channel]['multijet'][bkg_name_syst].SetName(f'multijet_{bkg_name_syst}')
                        for i in range(len(ivalues)):
                            nom_val = root_hists[channel]['multijet'][bkg_name_syst].GetBinContent( i+1 )
                            root_hists[channel]['multijet'][bkg_name_syst].SetBinContent( i+1, nom_val*ivalues[i]  )

            closureSysts = [ i.replace('Up', '') for i in root_hists[next(iter(root_hists))]['multijet'].keys() if i.endswith('Up') ]

        ### renaming histos for final combine inputs
        for channel in root_hists.keys():
            tt_label = metadata['processes']['background']['tt']['label']
            root_hists[channel][tt_label] = root_hists[channel]['data'].Clone()
            root_hists[channel][tt_label].SetName(tt_label)
            root_hists[channel][tt_label].SetTitle('tt_'+channel)
            root_hists[channel][tt_label].Reset()
            for ip, _ in list(root_hists[channel].items()):
                if 'TTTo' in ip:
                    root_hists[channel][tt_label].Add( root_hists[channel][ip] )
                    del root_hists[channel][ip]
                elif 'data' in ip:
                    root_hists[channel]['data_obs'] = root_hists[channel][ip]
                    root_hists[channel]['data_obs'].SetName("data_obs")
                    root_hists[channel]['data_obs'].SetTitle("data_obs_"+channel)
                    del root_hists[channel][ip]
                elif ip in metadata['processes']['all'].keys():
                    label = metadata['processes']['signal'][ip]['label'] if ip in metadata['processes']['signal'].keys() else metadata['processes']['background'][ip]['label']
                    root_hists[channel][label] = deepcopy(root_hists[channel][ip])
                    if isinstance(root_hists[channel][label], ROOT.TH1F):
                        root_hists[channel][label].SetName(label)
                        root_hists[channel][label].SetTitle(f'{label}_{channel}')
                    else:
                        for ivar, _ in root_hists[channel][label].items():
                            if 'nominal' in ivar: 
                                root_hists[channel][label][ivar].SetName(label)
                                root_hists[channel][label][ivar].SetTitle(f'{label}_{channel}')
                            else: 
                                root_hists[channel][label][ivar].SetName(f'{label}_{ivar}')
                                root_hists[channel][label][ivar].SetTitle(f'{label}_{ivar}_{channel}')
                    if not ip.startswith(label): del root_hists[channel][ip]
                else: 
                    logging.info(f"{ip} not in metadata processes, removing from root file.")
                    del root_hists[channel][ip]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = "hists_"+iclass+".root"
        output = output_dir+"/"+output_file

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
        juncSysts, btagSyst = [], []
        if not stat_only:
            juncSysts = [ s for s in mcSysts if s.startswith('CMS_scale_j') ]
            btagSysts = [ s for s in mcSysts if s.startswith('CMS_btag') ]

        total_process = len( metadata["processes"]["all"].keys() ) 

        hline = '-'*30
        lines = []
        lines.append('imax %d number of channels'%(len(metadata['processes']['signal'])*len(metadata['bin'])))
        lines.append(f'jmax {total_process-1} number of processes minus one') # zz, zh, hh, mj, tt is five processes, so jmax is 4
        lines.append('kmax * number of systematics')
        lines.append(hline)
        lines.append('shapes * * '+output_file+' $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC')
        lines.append(hline)

        def make_lalign_formatter(df, cols=None):
            if cols is None:
                cols = df.columns[df.dtypes == 'object']

            return {col: f'{{:<{df[col].str.len().max()}s}}'.format for col in cols}

        size_datacard = len(metadata['bin'])*len(metadata['processes']['all'])
        datacard = pd.DataFrame( columns=[f'col_{i}' for i in range( size_datacard + 2) ] )
        datacard.loc[0] = [ 'bin', '' ] + metadata['bin'] + [ '' for ibin in metadata['bin'] for _ in range(len(metadata['bin'])-1) ] 
        datacard.loc[1] = [ 'observation', '' ] + [ '-1' for _ in metadata['bin'] ] + [ '' for ibin in metadata['bin'] for _ in range(len(metadata['bin'])-1) ] 
        datacard.loc[2] = [ 'bin', '' ] + [ ibin for ibin in metadata['bin'] for _ in range(len(metadata['bin'])) ] 
        datacard.loc[3] = [ 'process', '' ] + [ metadata['processes']['all'][ibin]['label'] for ibin in metadata['processes']['all'] ] * len(metadata['bin'])
        datacard.loc[4] = [ 'process', '' ] + [ metadata['processes']['all'][ibin]['process'] for ibin in metadata['processes']['all'] ] * len(metadata['bin'])
        datacard.loc[5] = [ 'rate', '' ] + [ '-1' ] * len(metadata['bin']) * len(metadata['processes']['all'])

        is_multijet = (datacard.loc[3,:] == metadata['processes']['background']['multijet']['label']).to_numpy()
        for nuisance in closureSysts:
            row = pd.Series([ nuisance, 'shape'] + ['-'] * size_datacard, index=datacard.columns )
            new_row = row.where(~is_multijet, '1')
            datacard = datacard.append(new_row, ignore_index=True)
        
        for isignal in metadata['processes']['signal']:

            is_signal = (datacard.loc[3,:] == metadata['processes']['signal'][isignal]['label']).to_numpy()
            for nuisance in mcSysts:
                iy = ''.join(nuisance[-2:])
                row = pd.Series([ nuisance, 'shape'] + ['-'] * size_datacard, index=datacard.columns )
                if iy.isdigit():
                    is_year = (datacard.loc[2,:] == f'UL{iy}' ).to_numpy()
                    new_row = row.where(~(is_signal & is_year), '1')
                else:
                    new_row = row.where(~is_signal, '1')
                datacard = datacard.append(new_row, ignore_index=True)

            for isyst in metadata['uncertainty']:
                row = pd.Series([ isyst, metadata['uncertainty'][isyst]['type']] + ['-'] * size_datacard, index=datacard.columns )
                for iy in metadata['uncertainty'][isyst]['years']:
                    is_year = (datacard.loc[2,:] == iy ).to_numpy()
                    row = row.where(~(is_signal & is_year), metadata['uncertainty'][isyst]['years'][iy])
                datacard = datacard.append(row, ignore_index=True)

        lines.append( datacard.to_string(justify='left', header=False, index=False ) )
        if not stat_only:
            lines.append(hline)
            lines.append('* autoMCStats 0 1 1')
        lines.append(hline)
        # if closureSysts:
        #     lines.append('multijet group = %s'%(' '.join(closureSysts)))
        # if mcSysts:
        #     lines.append('btag     group = %s'%(' '.join(   btagSysts)))
        #     lines.append('junc     group = %s'%(' '.join(   juncSysts)))
        #     lines.append('trig     group = CMS_bbbb_resolved_ggf_triggerEffSF')
        # lines.append(f'lumi     group = {" ".join([ isyst for isyst in metadata["uncertainty"] if "lumi" in isyst ])}')
        # lines.append('theory   group = BR_hbb xs_hbbhbb') #mtop_ggH
        # if not stat_only:
        #     lines.append(f'others   group = lumi_13TeV_correlated lumi_13TeV_1718 lumi_2016 lumi_2017 lumi_2018 BR_hbb xs_hbbhbb CMS_pileup_2016 CMS_pileup_2017 CMS_pileup_2018 CMS_l1_ecal_prefiring_2016 CMS_l1_ecal_prefiring_2017 CMS_l1_ecal_prefiring_2018 %s'%(' '.join(juncSysts)))


        with open(output.replace('hists', 'combine').replace('root', 'txt'), 'w') as ofile:
            for line in lines:
                print(line)
                ofile.write(line+'\n')

        if make_syst_plots:

            import cmsstyle as CMS

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
            nom_mj = nom_data.Clone('multijet')
            nom_signal = nom_data.Clone('signal')
            for ichannel in root_hists.keys():
                nom_data.Add( root_hists[ichannel]['data_obs'] )
                nom_tt.Add( root_hists[ichannel]['tt'] )
                nom_mj.Add( root_hists[ichannel]['multijet']['nominal'] )
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

                    mj_nominal = root_hists[ichannel]['multijet']['nominal'].Clone()
                    mj_var_up = root_hists[ichannel]['multijet'][f"{isyst}Up"]
                    mj_var_dn = root_hists[ichannel]['multijet'][f"{isyst}Down"]

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
    parser.add_argument('-m', '--metadata', dest='metadata',
                        default='metadata/HH4b.yml', help="File contain systematic variations")
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

    logging.info("Creating root files for combine")
    create_combine_root_file(
        args.file_to_convert,
        args.rebin,
        args.classifier,
        args.output_dir,
        args.systematics_file,
        args.bkg_systematics_file,
        metadata_file=args.metadata,
        make_syst_plots=args.make_syst_plots,
        use_preUL=args.use_preUL,
        add_old_bkg=args.add_old_bkg,
        stat_only=args.stat_only
    )

