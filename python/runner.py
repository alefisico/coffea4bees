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
from functools import partial
from multiprocessing import Pool
import yaml


if __name__ == '__main__':

    ###### input parameters
    parser = argparse.ArgumentParser(description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t','--test', dest="test", action="store_true", default=False, help='Run as a test with few files')
    parser.add_argument('-o','--output', dest="output_file", default="hists.coffea", help='Output file.')
    parser.add_argument('-p','--processor', dest="processor", default="analysis/processors/processor_HH4b.py", help='Processor file.')
    parser.add_argument('-m','--metadata', dest="metadata", default="analysis/metadata/HH4b.yml", help='Metadata file.')
    parser.add_argument('-op','--outputPath', dest="output_path", default="hists/", help='Output path, if you want to save file somewhere else.')
    parser.add_argument('-y', '--year', nargs='+', dest='years', default=['UL18'], choices=['UL16_postVFP', 'UL16_preVFP', 'UL17', 'UL18'], help="Year of data to run. Example if more than one: --year UL17 UL18")
    parser.add_argument('-d', '--datasets', nargs='+', dest='datasets', default=['HH4b', 'ZZ4b', 'ZH4b'], help="Name of dataset to run. Example if more than one: -d HH4b ZZ4b")
    parser.add_argument('--condor', dest="condor", action="store_true", default=False, help='Run in condor')
    parser.add_argument( '--debug', help="Print lots of debugging statements", action="store_true", dest="debug", default=False)
    args = parser.parse_args()
    logging.basicConfig(level= logging.DEBUG if args.debug else logging.INFO )

    if args.test:
        #args.datasets=['HH4b']
        args.output_file='test.coffea'
    logging.info(f"\nRunning with these parameters: {args}")

    #### Metadata
    metadata = yaml.safe_load(open(args.metadata, 'r'))   #### AGE: to be modified
    metadata['config']['year'] = args.years[0]  ### check later for more years

    metadata_dataset = {}
    fileset = {}
    for year in args.years:
        logging.info(f"\nconfig year: {year}")
        for dataset in args.datasets:
            logging.info(f"\nconfig dataset: {dataset}")
            if dataset not in metadata['datasets'].keys():
                logging.error(f"{dataset} name not in metadatafile")
                sys.exit(0)

                
            if not year in metadata['datasets'][dataset]: 
                logging.warning(f"{year} name not in metadatafile for {dataset}")
                continue


            metadata_dataset[dataset] = {
                'xs'    : 1. if dataset in ['data','mixeddata'] else (metadata['datasets'][dataset]['xs'] if isinstance(metadata['datasets'][dataset]['xs'], float) else eval(metadata['datasets'][dataset]['xs']) ),
                'lumi'  : float(metadata['datasets']['data'][year]['lumi']),
                'year'  : year,
            }

            if isinstance( metadata['datasets'][dataset][year]["picoAOD"], dict):

                for iera, ifile in metadata['datasets'][dataset][year]["picoAOD"].items():
                    idataset = f'{dataset}{year}{iera}'
                    metadata_dataset[idataset] = metadata_dataset[dataset]
                    metadata_dataset[idataset]['era'] = iera
                    fileset[dataset+"_"+year] = {'files': [ f'root://cmseos.fnal.gov/{ifile}' ],
                                                  'metadata': metadata_dataset[idataset]}
                    logging.info(f'\nDataset {dataset+"_"+year} with {len(fileset[dataset+"_"+year]["files"])} files')

            else:
                fileset[dataset+"_"+year] = {'files': [ f'root://cmseos.fnal.gov/{metadata["datasets"][dataset][year]["picoAOD"]}' ],
                                             'metadata': metadata_dataset[dataset]}

                logging.info(f'\nDataset {dataset+"_"+year} with {len(fileset[dataset+"_"+year]["files"])} files')


    #### IF run in condor
    if args.condor:

        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        transfer_input_files = [ 'analysis/', 'base_class/', 'data/', 'skimmer/' ]

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
    processorName = args.processor.split('.')[0].replace("/",'.')
    try:
        analysis = getattr( importlib.import_module(processorName), 'analysis' )
        logging.info(f"\nRunning processsor: {processorName}")
    except (ModuleNotFoundError, NameError) as e:
        logging.error(f"{args.processor} No processor included. Check the --processor options and remember to call the processor class as: analysis")
        sys.exit(0)

    tstart = time.time()
    print(f"fileset is {fileset}")
    output, metrics = processor.run_uproot_job(
        fileset,
        treename = 'Events',
        processor_instance = analysis(**metadata['config']),
        executor = processor.dask_executor if args.condor else processor.futures_executor,
        executor_args = executor_args,
        chunksize = 1000 if args.test else 100_000,
        maxchunks = 10 if args.test else None,
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

