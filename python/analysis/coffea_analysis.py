import pickle, os, time, gc, argparse, sys
import numpy as np
import uproot
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
NanoAODSchema.warn_missing_crossrefs = False
import warnings
warnings.filterwarnings("ignore")
from coffea import processor
from coffea.util import save
import cachetools
import importlib
import logging

from helpers.MultiClassifierSchema import MultiClassifierSchema
from functools import partial
from multiprocessing import Pool
from helpers.networks import HCREnsemble
from helpers.jetCombinatoricModel import jetCombinatoricModel
import yaml

from helpers.correctionFunctions import btagSF_norm


if __name__ == '__main__':

    ##### Loading metadata
    fullmetadata = yaml.safe_load(open('metadata/HH4b.yml', 'r'))

    ###### input parameters
    parser = argparse.ArgumentParser(description='coffea_analysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t','--test', dest="test", action="store_true", default=False, help='Run as a test with few files')
    parser.add_argument('-o','--output', dest="output_file", default="hists.coffea", help='Output file.')
    parser.add_argument('-p','--processor', dest="processor", default="processors/processor_HH4b.py", help='Processor file.')
    parser.add_argument('-op','--outputPath', dest="output_path", default="hists/", help='Output path, if you want to save file somewhere else.')
    parser.add_argument('-y', '--year', nargs='+', dest='years', default=['2018'], choices=['2016_postVFP', '2016_preVFP', '2017', '2018'], help="Year of data to run. Example if more than one: --year 2016 2017")
    parser.add_argument('-d', '--datasets', nargs='+', dest='datasets', default=['HH4b', 'ZZ4b', 'ZH4b'], choices=fullmetadata['datasets'].keys(), help="Name of dataset to run. Example if more than one: -d HH4b ZZ4b")
    parser.add_argument('--condor', dest="condor", action="store_true", default=False, help='Run in condor')
    parser.add_argument( '--debug', help="Print lots of debugging statements", action="store_true", dest="debug", default=False)
    args = parser.parse_args()
    logging.basicConfig(level= logging.DEBUG if args.debug else logging.INFO )

    if args.test:
        #args.datasets=['HH4b']
        args.output_file='test.coffea'
    logging.info(f"\nRunning with these parameters: {args}")

    #### Metadata
    metadata = {}
    fileset = {}
    for year in args.years:
        for dataset in args.datasets:
            VFP = '_'+dataset.split('_')[-1] if 'VFP' in dataset else ''   #### AGE: I dont think we need it, maybe remove later
            era = f'{20 if "HH4b" in dataset else "UL"}{year[2:]+VFP}'
            jercCorrections = [ correctionsMetadata[era]["JERC"][0].replace('STEP', istep) for istep in ['L1FastJet', 'L2Relative', 'L2L3Residual', 'L3Absolute'] ] + correctionsMetadata[era]["JERC"][1:]

            metadata[dataset] = {
                                 'xs'    : 1. if 'data' else (fullmetadata['datasets'][dataset]['xs'] if isinstance(fullmetadata[dataset]['xs'], float) else eval(fullmetadata[dataset]['xs']) ),
                                 'lumi'  : float(fullmetadata['datasets']['data'][year]['lumi']),
                                 'year'  : year,
                                 'btagSF': correctionsMetadata[era]['btagSF'],
                                 'btagSF_norm': btagSF_norm(dataset),
                                 'juncWS': jercCorrections,
                                 'puWeight': correctionsMetadata[era]['PU'],
            }
            fileset[dataset] = {'files': [ f'root://cmseos.fnal.gov/{fullmetadata["datasets"][dataset][year]["picoAOD"]}' ],
                                'metadata': metadata[dataset]}

            logging.info(f'\nDataset {dataset} with {len(fileset[dataset]["files"])} files')


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
        logging.info("\nCluster arguments: ", cluster_args)

        cluster = LPCCondorCluster(**cluster_args)
        cluster.adapt(minimum=1, maximum=200)
        client = Client(cluster)
        # client = Client()

        logging.info('\nWaiting for at least one worker...')
        client.wait_for_workers(1)


        executor_args = {
            'client': client,
            'savemetrics': True,
            'schema': NanoAODSchema,
            'align_clusters': False,
        }
    else:
        executor_args = {'schema': NanoAODSchema, 'workers': 6, 'savemetrics':True}
    logging.info( f"\nExecutor arguments: {executor_args}")

    #### Run processor
    try:  analysis = getattr( importlib.import_module(args.processor.split('.')[0].replace("/",'.')), 'analysis' )
    except (ModuleNotFoundError, NameError) as e:
        logging.error("No processor included. Check the --processor options and remember to call the processor class as: analysis")
        sys.exit(0)

    tstart = time.time()
    output, metrics = processor.run_uproot_job(
        fileset,
        treename = 'Events',
        processor_instance = analysis(**fullmetadata['config']),
        executor = processor.dask_executor if args.condor else processor.futures_executor,
        executor_args = executor_args,
        chunksize = 100 if args.test else 100_000,
        maxchunks = 1 if args.test else None,
    )
    elapsed = time.time() - tstart
    if args.condor:
        nEvent = metrics['entries']
        processtime = metrics['processtime']
        logging.info(f'\n{nEvent/elapsed:,.0f} events/s total ({nEvent}/{elapsed}, processtime {processtime})')
    else:
        nEvent = sum([output['nEvent'][dataset] for dataset in output['nEvent'].keys()])
        logging.info(f'\n{nEvent/elapsed:,.0f} events/s total ({nEvent}/{elapsed})')

    ##### Saving file
    if not os.path.exists(args.output_path): os.makedirs(args.output_path)
    hfile = f'{args.output_path}/{args.output_file}'
    logging.info(f'\nSaving file {hfile}')
    save(output, hfile)

