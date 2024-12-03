import os, sys
import ROOT
import argparse
import logging
import json
import yaml
import numpy as np
import pickle
from copy import copy, deepcopy
from convert_json_to_root import json_to_TH1
from make_variable_binning import make_variable_binning

import CombineHarvester.CombineTools.ch as ch
ROOT.gROOT.SetBatch(True)

def make_trigger_syst( json_input, root_output, name, rebin ):

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
        root_output[f'CMS_bbbb_resolved_ggf_triggerEffSF{ivar}'] = json_to_TH1( htrig, name+ivar, rebin )

def create_combine_root_file( file_to_convert,
                             rebin,
                             var,
                             output_dir,
                             systematics_file,
                             bkg_systematics_file,
                             metadata_file='metadata/HH4b.yml',
                             make_syst_plots=False,
                             use_preUL=False,
                             add_old_bkg=False,
                             variable_binning=False,
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


    root_hists = {}
    mcSysts, closureSysts = [], []
    for iyear in coffea_hists[var]['data'].keys():
        root_hists[iyear] = {}

        ### For multijets

        root_hists[iyear]['multijet'] = {}
        root_hists[iyear]['multijet']['nominal'] = json_to_TH1(
            coffea_hists[var]['data'][iyear]['threeTag']['SR'], 'multijet_'+iyear+var, rebin )

        ### signals
        for iprocess in coffea_hists[var].keys():
            if iprocess not in metadata['processes']['signal']:
                root_hists[iyear][iprocess] = json_to_TH1( coffea_hists[var][iprocess][iyear]['fourTag']['SR'], 
                                                            f'{iprocess.split("4b")[0]}_{iyear}', rebin )
            else:
                root_hists[iyear][iprocess] = {}
                root_hists[iyear][iprocess]['nominal'] = json_to_TH1(
                    coffea_hists[var][iprocess][iyear]['fourTag']['SR'], iprocess+'_'+iyear, rebin )


        if systematics_file and not use_preUL:
            for iprocess in metadata['processes']['signal']:

                root_hists[iyear][iprocess] = {}
                for ivar in coffea_hists_syst[var][iprocess][iyear].keys():
                    
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
                        make_trigger_syst(coffea_hists_syst[var][iprocess][iyear],
                                            root_hists[iyear][iprocess],
                                            f'{iprocess}_{ivar}_{iyear}', rebin)
                    elif 'triggerEffSFDown' in namevar: continue
                    else:
                        root_hists[iyear][iprocess][namevar] = json_to_TH1(
                                                        coffea_hists_syst[var][iprocess][iyear][ivar]['fourTag']['SR'], 
                                                        f'{iprocess}_{ivar}_{iyear}', rebin )
    
    if systematics_file and use_preUL:
        iprocess = 'HH4b'
        for iyear in coffea_hists_syst[var]['HH4b'].keys():
            tmpname=iyear.replace('20', 'UL') + ('_preVFP' if '16' in iyear else '')
            root_hists[tmpname][iprocess] = {}
            for ivar in coffea_hists_syst[var][iprocess][iyear].keys():
                root_hists[tmpname][iprocess][ivar] = json_to_TH1(
                    coffea_hists_syst[var][iprocess][iyear][ivar]['fourTag']['SR'], iprocess+"_"+ivar+"_"+iyear, rebin )


    # if "UL16_preVFP" not in metadata['bin']:
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

    for iy in list(root_hists.keys()):
        for jy in metadata['bin']:
            if ''.join(iy[-2:]) == ''.join(jy[-2:]):
                root_hists[jy] = root_hists.pop(iy)


    if not stat_only:
        if add_old_bkg:
            old_bkg_file = ROOT.TFile(f"HIG-22-011/hist_{var.replace('.', '_')}.root", 'read' )
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
        root_hists[channel][tt_label].SetTitle(f"{tt_label}_{channel}")
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

    output_file = "shapes.root" 
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
    for i, ibin in enumerate(metadata['bin']):

        cb = ch.CombineHarvester()
        cb.SetVerbosity(3)

        cats = [(i, ibin)]
        cb.AddObservations(['*'], [''], ['13TeV'], ['*'], cats)
        cb.AddProcesses(['*'], [''], ['13TeV'], ['*'], [ metadata['processes']['all'][ibin]['label'] for ibin in metadata['processes']['background'] ], cats, False)
        signals = [ metadata['processes']['all'][ibin]['label'] for ibin in metadata['processes']['signal'] ]
        cb.AddProcesses(['*'], [''], ['13TeV'], ['*'], signals, cats, True)

        if stat_only:
            cb.cp().backgrounds().ExtractShapes( output, '$BIN/$PROCESS', '')
            cb.cp().signals().ExtractShapes( output, '$BIN/$PROCESS', '')
            cb.PrintAll()
            cb.WriteDatacard(f"{output_dir}/datacard_{ibin}.txt", f"{output_dir}/{ibin}_{output_file}")

        else:
            for nuisance in closureSysts:
                cb.cp().process(["multijet"]).AddSyst(cb, nuisance, 'shape', ch.SystMap()(1.0))
            
            for nuisance in mcSysts:
                if ('2016' in nuisance):
                    if ('2016' in ibin):
                        cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap('bin')(['HHbb_2016'],1.0))
                elif ('2017' in nuisance):
                    if('2017' in ibin):
                        cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap('bin')(['HHbb_2017'],1.0))
                elif ('2018' in nuisance): 
                    if ('2018' in ibin):
                        cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap('bin')(['HHbb_2018'],1.0))
                else:
                    cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap()(1.0))

            for isyst in metadata['uncertainty']:
                if ('2016' in isyst):
                    if ('2016' in ibin):
                        cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')(['HHbb_2016'],metadata['uncertainty'][isyst]['years']['HHbb_2016']))
                elif ('2017' in isyst):
                    if ('2017' in ibin):
                        cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')(['HHbb_2017'],metadata['uncertainty'][isyst]['years']['HHbb_2017']))
                elif ('2018' in isyst):
                    if ('2018' in ibin):
                        cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')(['HHbb_2018'],metadata['uncertainty'][isyst]['years']['HHbb_2018']))
                elif ('1718' in isyst):
                    if '2017' in ibin or '2018' in ibin:
                        cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')
                                            ([ibin],metadata['uncertainty'][isyst]['years'][ibin]))
                else:
                    cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')
                                            ([ibin], metadata['uncertainty'][isyst]['years'][ibin])
                                            )

            cb.cp().backgrounds().ExtractShapes(
                output, '$BIN/$PROCESS', '$BIN/$PROCESS_$SYSTEMATIC')
            cb.cp().signals().ExtractShapes(
                output, '$BIN/$PROCESS', '$BIN/$PROCESS_$SYSTEMATIC')
            
            cb.cp().SetAutoMCStats(cb, 0, 1, 1)

            cb.PrintAll()
            cb.WriteDatacard(f"{output_dir}/datacard_{ibin}.txt", f"{output_dir}/{ibin}_{output_file}")



if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Convert json hist to root TH1F', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        default="./datacards/", help='Output directory.')
    parser.add_argument('--var', dest="variable", 
                        default="SvB_MA.ps_hh_fine", help='Variable to make histograms.')
    parser.add_argument('-f', '--file', dest='file_to_convert',
                        default="histos/histAll.json", help="File with coffea hists")
    parser.add_argument('-r', '--rebin', dest='rebin', type=int,
                        default=15, help="Rebin")
    parser.add_argument('--variable_binning', dest='variable_binning', action="store_true",
                        default=False, help="Make variable binning based on the amount of signal. (ran make_variable_binning.py)")
    parser.add_argument('-s', '--syst_file', dest='systematics_file',
                        default='', help="File contain systematic variations")
    parser.add_argument('-b', '--bkg_syst_file', dest='bkg_systematics_file',
                        default='', help="File contain background systematic variations")
    parser.add_argument('-m', '--metadata', dest='metadata',
                        default='stats_analysis/metadata/HH4b.yml', help="File contain systematic variations")
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

    if args.variable_binning:
        logging.info(f"Creating variable binning and using {args.rebin} as threshold for data and signal.")
        args.rebin = list(make_variable_binning(args.file_to_convert, args.variable, args.rebin, None ))
    
    logging.info("Creating root files for combine")
    create_combine_root_file(
        args.file_to_convert,
        args.rebin,
        args.variable,
        args.output_dir,
        args.systematics_file,
        args.bkg_systematics_file,
        metadata_file=args.metadata,
        make_syst_plots=args.make_syst_plots,
        use_preUL=args.use_preUL,
        add_old_bkg=args.add_old_bkg,
        stat_only=args.stat_only,
    )