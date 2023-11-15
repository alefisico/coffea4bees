import pickle, os, time, gc, argparse
import numpy as np
import uproot
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
NanoAODSchema.warn_missing_crossrefs = False
import warnings
warnings.filterwarnings("ignore")
from coffea import processor
import cachetools

from helpers.MultiClassifierSchema import MultiClassifierSchema
from functools import partial
from multiprocessing import Pool
from helpers.networks import HCREnsemble
from helpers.jetCombinatoricModel import jetCombinatoricModel
import yaml

from processors.processor_HH4b import analysis
from helpers.correctionFunctions import btagSF_norm, btagVariations, juncVariations


if __name__ == '__main__':

    ##### Loading metadata
    fullmetadata = yaml.safe_load(open('metadata/HH4b.yml', 'r'))
    correctionsMetadata = yaml.safe_load(open('metadata/corrections.yml', 'r'))

    ###### input parameters
    parser = argparse.ArgumentParser(description='coffea_analysis')
    parser.add_argument('-t','--test', dest="test", action="store_true", default=False, help='Run as a test with few files')
    parser.add_argument('-o','--output', dest="output_file", default="hists.pkl", help='Output file. Default: hists.pkl')
    parser.add_argument('-op','--outputPath', dest="output_path", default="hists/", help='Output path, if you want to save file somewhere else. Default: hists/')
    parser.add_argument('-y', '--year', nargs='+', dest='years', default=['2018'], choices=['2016', '2017', '2018', 'RunII'], help="Year of data to run. Example if more than one: --year 2016 2017")
    parser.add_argument('-d', '--datasets', nargs='+', dest='datasets', default=['HH4b', 'ZZ4b', 'ZH4b'], choices=fullmetadata.keys(), help="Name of dataset to run. Example if more than one: -d HH4b ZZ4b")
    parser.add_argument('--debug', dest="debug", action="store_true", default=False, help='Run in debug mode')
    parser.add_argument('--condor', dest="condor", action="store_true", default=False, help='Run in condor')
    args = parser.parse_args()
    print("Running with these parameters:", args)

    #### Metadata
    metadata = {}
    fileset = {}
    for year in args.years:
        for dataset in args.datasets:
            VFP = '_'+dataset.split('_')[-1] if 'VFP' in dataset else ''   #### AGE: I dont think we need it, maybe remove later
            era = f'{20 if "HH4b" in dataset else "UL"}{year[2:]+VFP}'
            jercCorrections = [ correctionsMetadata[era]["JERC"][0].replace('STEP', istep) for istep in ['L1FastJet', 'L2Relative', 'L2L3Residual', 'L3Absolute'] ] + correctionsMetadata[era]["JERC"][1:]

            metadata[dataset] = {'isMC'  : True,
                                 'xs'    : eval(fullmetadata[dataset]['xs']),
                                 'lumi'  : float(fullmetadata['data'][year]['lumi']),
                                 'year'  : year,
                                 'btagSF': correctionsMetadata[era]['btagSF'],
                                 'btagSF_norm': btagSF_norm(dataset),
                                 'juncWS': jercCorrections,
                                 'puWeight': correctionsMetadata[era]['PU'],
            }
            fileset[dataset] = {'files': [ f'root://cmseos.fnal.gov/{fullmetadata[dataset][year]["picoAOD"]}' ],
                                'metadata': metadata[dataset]}

            print(f'Dataset {dataset} with {len(fileset[dataset]["files"])} files')


    #### analysis arguments
    analysis_args = {'debug': args.debug,
                     'JCM': 'weights/dataRunII/jetCombinatoricModel_SB_00-00-02.txt',
                     'btagVariations': btagVariations(systematics=True),
                     'juncVariations': juncVariations(systematics=False),
                     'threeTag': False,
                     'apply_puWeight':True,
                     'apply_prefire' :True,
                     # 'SvB'   : 'ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_8_np753_seed0_lr0.01_epochs20_offset*_epoch20.pkl',
                     # 'SvB_MA': 'ZZ4b/nTupleAnalysis/pytorchModels/SvB_MA_HCR+attention_8_np1061_seed0_lr0.01_epochs20_offset*_epoch20.pkl',
    }

    #### IF run in condor
    if args.condor:

        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        transfer_input_files = ['helpers/', 'metadata/', 'processors/', 'data/', 'pytorchModels/' ]

        cluster_args = {'transfer_input_files': transfer_input_files,
                        'shared_temp_directory': '/tmp',
                        'cores': 2,
                        'memory': '4GB',
                        'ship_env': False}

        cluster = LPCCondorCluster(**cluster_args)
        cluster.adapt(minimum=1, maximum=200)
        client = Client(cluster)
        # client = Client()

        print('Waiting for at least one worker...')
        client.wait_for_workers(1)


        executor_args = {
            'client': client,
            'savemetrics': True,
            'schema': NanoAODSchema,
            'align_clusters': False,
        }
    else:
        executor_args = {'schema': NanoAODSchema, 'workers': 6}

    #### Run processor
    tstart = time.time()
    output = processor.run_uproot_job(
        fileset,
        treename = 'Events',
        processor_instance = analysis(**analysis_args),
        executor = processor.dask_executor if args.condor else processor.futures_executor,
        executor_args = executor_args,
        chunksize = 100 if args.test else 10_000,
        maxchunks = 1 if args.test else None,
    )
    elapsed = time.time() - tstart
    nEvent = sum([output['nEvent'][dataset] for dataset in output['nEvent'].keys()])
    print(f'{nEvent/elapsed:,.0f} events/s total ({nEvent}/{elapsed})')

    if not os.path.exists(args.output_path): os.makedirs(args.output_path)
    with open(f'{args.output_path}/{args.output_file}', 'wb') as hfile:
        print(f'pickle.dump(output, {args.output_path}/{args.output_file})')
        pickle.dump(output, hfile)

