from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import dask
import uproot
import fsspec
import psutil
import yaml
from base_class.addhash import get_git_diff, get_git_revision_hash
from coffea import processor
from coffea.dataset_tools import rucio_utils
from coffea.nanoevents import NanoAODSchema, PFNanoAODSchema
from coffea.util import save
from dask.distributed import performance_report
from distributed.diagnostics.plugin import WorkerPlugin
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from skimmer.processor.picoaod import fetch_metadata, integrity_check, resize

if TYPE_CHECKING:
    from base_class.root import Friend

dask.config.set({'logging.distributed': 'error'})

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")

@dataclass
class WorkerInitializer(WorkerPlugin):
    uproot_xrootd_retry_delays: list[float] = None

    def setup(self, worker=None):
        if delays := self.uproot_xrootd_retry_delays:
            from base_class.root.patch import uproot_XRootD_retry
            uproot_XRootD_retry(len(delays) + 1, delays)

def checking_input_files(outfiles):
    '''Check if the input files are corrupted'''

    good_files = []
    for outfile in outfiles:
            try:
                # Attempt to open the file with uproot to check for corruption
                uproot.open(outfile + ":Events")
                good_files.append(outfile)
            except Exception as e:
                logging.error(f"Error opening file {outfile}: {e}")
                logging.error(f"Skipping corrupted file {outfile}")
    return good_files


def list_of_files(ifile,
                  allowlist_sites: list =['T3_US_FNALLPC'],
                  blocklist_sites: list =[''],
                  rucio_regex_sites: str ='T[23]',
                  test: bool = False,
                  test_files: int = 5,
                  check_input_files: bool = False
                  ):
    '''Check if ifile is root file or dataset to check in rucio'''

    if isinstance(ifile, list):
        ifile = checking_input_files(ifile) if check_input_files else ifile
        return ifile[:(test_files if test else None)]
    elif ifile.endswith('.txt'):
        file_list = [
            jfile.rstrip() if jfile.startswith(('root','file')) else f'root://cmseos.fnal.gov/{jfile.rstrip()}' for jfile in open(ifile).readlines()]
        file_list = checking_input_files(file_list) if check_input_files else file_list
        return file_list[:(test_files if test else None)]
    else:
        rucio_client = rucio_utils.get_rucio_client()
        outfiles, outsite, sites_counts = rucio_utils.get_dataset_files_replicas(
            ifile, client=rucio_client, regex_sites=fr"{rucio_regex_sites}", mode="first", allowlist_sites=allowlist_sites, blocklist_sites=blocklist_sites)
        good_files = checking_input_files(outfiles) if check_input_files else outfiles
        return good_files[:(test_files if test else None)]


def _friend_merge_name(path1: str, path0: str, name: str, **_):
    return f'{path1}/{path0.replace("picoAOD", name)}'

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
def profile(func):
    def wrapper(*args, **kwargs):

        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        logging.info("{}:consumed memory (before, after, diff): {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))

        return result
    return wrapper


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--test', dest="test", action="store_true",
                        default=False, help='Run as a test with few files')
    parser.add_argument('-o', '--output', dest="output_file",
                        default="hists.coffea", help='Output file.')
    parser.add_argument('-p', '--processor', dest="processor",
                        default="analysis/processors/processor_HH4b.py", help='Processor file.')
    parser.add_argument('-c', '--configs', dest="configs",
                        default="analysis/metadata/HH4b.yml", help='Config file.')
    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="metadata/datasets_HH4b.yml", help='Metadata datasets file.')
    parser.add_argument('-op', '--outputPath', dest="output_path", default="hists/",
                        help='Output path, if you want to save file somewhere else.')
    parser.add_argument('-y', '--year', nargs='+', dest='years', default=['UL18'], choices=[
                        '2016', '2017', '2018', 'UL16_postVFP', 'UL16_preVFP', 'UL17', 'UL18', '2022_preEE', '2022_EE', '2023_preBPix', '2023_BPix'],
                        help="Year of data to run. Example if more than one: --year UL17 UL18")
    parser.add_argument('-d', '--datasets', nargs='+', dest='datasets', default=[
                        'HH4b', 'ZZ4b', 'ZH4b'], help="Name of dataset to run. Example if more than one: -d HH4b ZZ4b")
    parser.add_argument('-e', '--era', nargs='+', dest='era', default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                        help="For data only. To run only on one data era.")
    parser.add_argument('-s', '--skimming', dest="skimming", action="store_true",
                        default=False, help='Run skimming instead of analysis')
    parser.add_argument('--dask', dest="run_dask",
                        action="store_true", default=False, help='Run with dask')
    parser.add_argument('--condor', dest="condor",
                        action="store_true", default=False, help='Run in condor')
    parser.add_argument('--debug', help="Print lots of debugging statements",
                        action="store_true", dest="debug", default=False)
    parser.add_argument('--githash', dest="githash",
                        default="", help='Overwrite git hash for reproducible')
    parser.add_argument('--gitdiff', dest="gitdiff",
                        default="", help='Overwrite git diff for reproducible')
    parser.add_argument('--check_input_files', dest="check_input_files",
                        action="store_true", default=False, help='Check input files for corruption')
    args = parser.parse_args()
    # configure default logger
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        handlers=[RichHandler(level=logging_level, markup=True)],
        format="%(message)s",
    )
    # disable numba debug warnings
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger("lpcjobqueue").setLevel(logging.WARNING)

    logging.info(f"Running with these parameters: {args}")

    #
    # Metadata
    #
    configs = yaml.safe_load(open(args.configs, 'r'))
    metadata = yaml.safe_load(open(args.metadata, 'r'))

    config_runner = configs['runner'] if 'runner' in configs.keys() else {}
    config_runner.setdefault('data_tier', 'picoAOD')
    config_runner.setdefault('chunksize', (1_000 if args.test else 100_000))
    config_runner.setdefault('maxchunks', (1 if args.test else None))
    config_runner.setdefault('schema', NanoAODSchema)
    config_runner.setdefault('test_files', 5)
    config_runner.setdefault('allowlist_sites', ['T3_US_FNALLPC'])
    config_runner.setdefault('blocklist_sites', [''])
    config_runner.setdefault('rucio_regex_sites', "T[23]")
    config_runner.setdefault('class_name', 'analysis')
    config_runner.setdefault('condor_cores', 2)
    config_runner.setdefault('condor_memory', '4GB')
    config_runner.setdefault('condor_transfer_input_files', [
                             'analysis/', 'base_class/', 'classifier/', 'data/', 'skimmer/'])
    config_runner.setdefault('min_workers', 1)
    config_runner.setdefault('max_workers', 100)
    config_runner.setdefault('workers', 2)
    config_runner.setdefault('skipbadfiles', False)
    config_runner.setdefault('dashboard_address', 10200)
    config_runner.setdefault('friend_base', None)
    config_runner.setdefault('friend_base_argname', "make_classifier_input")
    config_runner.setdefault('friend_metafile', 'friends')
    config_runner.setdefault('friend_merge_step', 100_000)
    config_runner.setdefault('write_coffea_output', True)
    config_runner.setdefault('override_top_reconstruction', None)
    config_runner.setdefault('uproot_xrootd_retry_delays', [5, 15, 45])

    if isinstance(config_runner['schema'], str):
        if config_runner['schema'] == "NanoAODSchema":
            config_runner['schema'] = NanoAODSchema
        elif config_runner['schema'] == "PFNanoAODSchema":
            config_runner['schema'] = PFNanoAODSchema
        else:
            raise ValueError(f"Unknown schema: {config_runner['schema']}")

    if 'all' in args.datasets:
        metadata['datasets'].pop("mixeddata")
        metadata['datasets'].pop("datamixed")
        metadata['datasets'].pop("synthetic_data")
        metadata['datasets'].pop("data_3b_for_mixed")
        args.datasets = metadata['datasets'].keys()

    metadata_dataset = {}
    fileset = {}
    for year in args.years:
        logging.info(f"config year: {year}")
        for dataset in args.datasets:
            logging.info(f"config dataset: {dataset}")
            if dataset not in metadata['datasets'].keys():
                logging.error(f"{dataset} name not in metadatafile")
                continue

            if year not in metadata['datasets'][dataset]:
                logging.warning(
                    f"{year} name not in metadatafile for {dataset}")
                continue

            if dataset in ['data', 'mixeddata', 'datamixed', 'data_3b_for_mixed', 'synthetic_data'] or not ('xs' in metadata['datasets'][dataset].keys()):
                xsec = 1.
            elif isinstance(metadata['datasets'][dataset]['xs'], float):
                xsec = metadata['datasets'][dataset]['xs']
            else:
                xsec = eval(metadata['datasets'][dataset]['xs'])

            top_reconstruction = config_runner["override_top_reconstruction"] or (
                metadata['datasets'][dataset]['top_reconstruction']
                if "top_reconstruction" in metadata['datasets'][dataset]
                else None)
            logging.info(f"top construction configured as {top_reconstruction} ")

            metadata_dataset[dataset] = {'year': year,
                                         'processName': dataset,
                                         'xs': xsec,
                                         'lumi': float(metadata['datasets']['data'][year]['lumi']),
                                         'trigger':  metadata['datasets']['data'][year]['trigger'],
                                         'top_reconstruction':  top_reconstruction
                                         }
            isData = (dataset == 'data')
            isMixedData = (dataset == 'mixeddata')
            isDataMixed = (dataset == 'datamixed')
            isSyntheticData = (dataset == 'synthetic_data')
            isDataForMix = (dataset == 'data_3b_for_mixed')
            isTTForMixed = (dataset in ['TTToHadronic_for_mixed', 'TTToSemiLeptonic_for_mixed', 'TTTo2L2Nu_for_mixed'])

            if not ( isData or isSyntheticData or isMixedData or isDataMixed or isDataForMix or isTTForMixed):
                logging.info("Config MC")
                if config_runner['data_tier'].startswith('pico'):
                    if 'data' not in dataset:
                        metadata_dataset[dataset]['genEventSumw'] = metadata['datasets'][dataset][year][config_runner['data_tier']]['sumw']
                    meta_files = metadata['datasets'][dataset][year][config_runner['data_tier']]['files']
                # if not dataset.endswith('data'):
                #     if config_runner['data_tier'].startswith('pico'):
                #         metadata_dataset[dataset]['genEventSumw'] = metadata['datasets'][dataset][year][config_runner['data_tier']]['sumw']
                #         meta_files = metadata['datasets'][dataset][year][config_runner['data_tier']]['files']
                else:
                    metadata_dataset[dataset]['genEventSumw'] = 1
                    meta_files = metadata['datasets'][dataset][year][config_runner['data_tier']]

                fileset[dataset + "_" + year] = {'files': list_of_files(meta_files,
                                                                        test=args.test,
                                                                        test_files=config_runner['test_files'],
                                                                        allowlist_sites=config_runner['allowlist_sites'],
                                                                        blocklist_sites=config_runner['blocklist_sites'],
                                                                        rucio_regex_sites=config_runner['rucio_regex_sites']),
                                                 'metadata': metadata_dataset[dataset]}

                logging.info(f'Dataset {dataset+"_"+year} with '
                             f'{len(fileset[dataset+"_"+year]["files"])} files')

            elif isMixedData:
                logging.info("Config Mixed Data ")

                nMixedSamples = metadata['datasets'][dataset]["nSamples"]
                mixed_config = metadata['datasets'][dataset][year][config_runner['data_tier']]
                logging.info(f"Number of mixed samples is {nMixedSamples}")
                for v in range(nMixedSamples):

                    mixed_name = f"mix_v{v}"
                    idataset = f'{mixed_name}_{year}'

                    metadata_dataset[idataset] = copy(metadata_dataset[dataset])
                    metadata_dataset[idataset]['processName'] = mixed_name
                    metadata_dataset[idataset]['FvT_name'] = mixed_config['FvT_name_template'].replace("XXX",str(v))
                    metadata_dataset[idataset]['FvT_file'] = mixed_config['FvT_file_template'].replace("XXX",str(v))
                    mixed_files = [f.replace("XXX",str(v)) for f in mixed_config['files_template']]
                    fileset[idataset] = {'files': list_of_files(mixed_files,
                                                                test=args.test, test_files=config_runner['test_files'],
                                                                allowlist_sites=config_runner['allowlist_sites'],
                                                                blocklist_sites=config_runner['blocklist_sites'],
                                                                rucio_regex_sites=config_runner['rucio_regex_sites']),
                                         'metadata': metadata_dataset[idataset]}

                    logging.info(
                        f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')

            elif isDataMixed:
                logging.info("Config Data Mixed ")

                nMixedSamples = metadata['datasets'][dataset]["nSamples"]
                mixed_config = metadata['datasets'][dataset][year][config_runner['data_tier']]
                logging.info(f"Number of mixed samples is {nMixedSamples}")
                for v in range(nMixedSamples):

                    mixed_name = f"mix_v{v}"
                    idataset = f'{mixed_name}_{year}'

                    metadata_dataset[idataset] = copy(metadata_dataset[dataset])
                    metadata_dataset[idataset]['processName'] = mixed_name
                    mixed_files = [f.replace("XXX",str(v)) for f in mixed_config['files_template']]
                    fileset[idataset] = {'files': list_of_files(mixed_files,
                                                                test=args.test, test_files=config_runner['test_files'],
                                                                allowlist_sites=config_runner['allowlist_sites'],
                                                                blocklist_sites=config_runner['blocklist_sites'],
                                                                rucio_regex_sites=config_runner['rucio_regex_sites']),
                                         'metadata': metadata_dataset[idataset]}

                    logging.info(
                        f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')


            elif isSyntheticData:
                logging.info("Config Synthetic Data ")

                nSyntheticSamples = metadata['datasets'][dataset]["nSamples"]
                synthetic_config = metadata['datasets'][dataset][year][config_runner['data_tier']]
                logging.info(f"Number of synthetic samples is {nSyntheticSamples}")
                for v in range(nSyntheticSamples):

                    synthetic_name = f"syn_v{v}"
                    idataset = f'{synthetic_name}_{year}'

                    metadata_dataset[idataset] = copy(metadata_dataset[dataset])
                    metadata_dataset[idataset]['processName'] = synthetic_name
                    # metadata_dataset[idataset]['FvT_name'] = synthetic_config['FvT_name_template'].replace("XXX",str(v))
                    # metadata_dataset[idataset]['FvT_file'] = synthetic_config['FvT_file_template'].replace("XXX",str(v))
                    synthetic_files = [f.replace("XXX",str(v)) for f in synthetic_config['files_template']]
                    fileset[idataset] = {'files': list_of_files(synthetic_files,
                                                                test=args.test, test_files=config_runner['test_files'],
                                                                allowlist_sites=config_runner['allowlist_sites'],
                                                                blocklist_sites=config_runner['blocklist_sites'],
                                                                rucio_regex_sites=config_runner['rucio_regex_sites']),
                                         'metadata': metadata_dataset[idataset]}

                    logging.info(
                        f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')

            elif isDataForMix:
                logging.info("Config Data for Mixed ")

                nMixedSamples = metadata['datasets'][dataset]["nSamples"]
                use_kfold = config_runner.get("use_kfold", False)
                use_ZZinSB = config_runner.get("use_ZZinSB", False)
                use_ZZandZHinSB = config_runner.get("use_ZZandZHinSB", False)
                data_3b_mix_config = metadata['datasets'][dataset][year][config_runner['data_tier']]
                logging.info(f"Number of mixed samples is {nMixedSamples}")
                logging.info(f"Using kfolding? {use_kfold}")
                logging.info(f"Using ZZinSB? {use_ZZinSB}")
                logging.info(f"Using ZZandZHinSB? {use_ZZandZHinSB}")

                idataset = f'{dataset}_{year}'

                metadata_dataset[idataset] = copy(metadata_dataset[dataset])
                metadata_dataset[idataset]['JCM_loads'] = [data_3b_mix_config['JCM_load_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                if use_kfold:
                    metadata_dataset[idataset]['FvT_files'] = [data_3b_mix_config['FvT_file_kfold_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                    metadata_dataset[idataset]['FvT_names'] = [data_3b_mix_config['FvT_name_kfold_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                elif use_ZZinSB:
                    metadata_dataset[idataset]['FvT_files'] = [data_3b_mix_config['FvT_file_ZZinSB_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                    metadata_dataset[idataset]['FvT_names'] = [data_3b_mix_config['FvT_name_ZZinSB_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                elif use_ZZandZHinSB:
                    metadata_dataset[idataset]['FvT_files'] = [data_3b_mix_config['FvT_file_ZZandZHinSB_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                    metadata_dataset[idataset]['FvT_names'] = [data_3b_mix_config['FvT_name_ZZandZHinSB_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                else:
                    metadata_dataset[idataset]['FvT_files'] = [data_3b_mix_config['FvT_file_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                    metadata_dataset[idataset]['FvT_names'] = [data_3b_mix_config['FvT_name_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]

                fileset[idataset] = {'files': list_of_files(data_3b_mix_config['files'],
                                                            test=args.test, test_files=config_runner['test_files'],
                                                            blocklist_sites=config_runner['blocklist_sites'],
                                                            allowlist_sites=config_runner['allowlist_sites'], rucio_regex_sites=config_runner['rucio_regex_sites']),
                                     'metadata': metadata_dataset[idataset]}

                logging.info(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')

            elif isTTForMixed:
                logging.info("Config TT for Mixed ")

                nMixedSamples = metadata['datasets'][dataset]["nSamples"]
                TT_3b_mix_config = metadata['datasets'][dataset][year][config_runner['data_tier']]
                logging.info(f"Number of mixed samples is {nMixedSamples}")

                idataset = f'{dataset}_{year}'

                metadata_dataset[idataset] = copy(metadata_dataset[dataset])
                metadata_dataset[idataset]['FvT_files'] = [TT_3b_mix_config['FvT_file_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                metadata_dataset[idataset]['FvT_names'] = [TT_3b_mix_config['FvT_name_template'].replace("XXX",str(v)) for v in range(nMixedSamples)]
                metadata_dataset[idataset]['genEventSumw'] = TT_3b_mix_config['sumw']

                fileset[idataset] = {'files': list_of_files(TT_3b_mix_config['files'],
                                                            test=args.test, test_files=config_runner['test_files'],
                                                            allowlist_sites=config_runner['allowlist_sites'],
                                                            blocklist_sites=config_runner['blocklist_sites'],
                                                            rucio_regex_sites=config_runner['rucio_regex_sites']),
                                     'metadata': metadata_dataset[idataset]}

                logging.info(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')

            # isData
            else:

                for iera, ifile in metadata['datasets'][dataset][year][config_runner['data_tier']].items():
                    if iera in args.era:
                        idataset = f'{dataset}_{year}{iera}'
                        metadata_dataset[idataset] = metadata_dataset[dataset]
                        metadata_dataset[idataset]['era'] = iera
                        fileset[idataset] = {'files': list_of_files((ifile['files'] if config_runner['data_tier'].startswith('pico') else ifile),
                                                                    test=args.test,
                                                                    test_files=config_runner['test_files'],
                                                                    allowlist_sites=config_runner['allowlist_sites'],
                                                                    blocklist_sites=config_runner['blocklist_sites'],
                                                                    rucio_regex_sites=config_runner['rucio_regex_sites']),
                                            'metadata': metadata_dataset[idataset]}

                        logging.info(
                            f'\nDataset {idataset} with {len(fileset[idataset]["files"])} files')

    client = None
    pool = None
    #
    # IF run in condor
    #
    if args.condor:

        args.run_dask = True
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster

        transfer_input_files = config_runner['condor_transfer_input_files']

        cluster_args = {'transfer_input_files': transfer_input_files,
                        'shared_temp_directory': '/tmp',
                        'cores': config_runner['condor_cores'],
                        'memory': config_runner['condor_memory'],
                        'ship_env': False,
                        'scheduler_options': {'dashboard_address': config_runner['dashboard_address']},
                        'worker_extra_args':[
                            f"--worker-port 10000:10100",
                            f"--nanny-port 10100:10200",
                        ]}
        logging.info("Cluster arguments: ")
        logging.info(pretty_repr(cluster_args))

        cluster = LPCCondorCluster(**cluster_args)
        cluster.adapt(
            minimum=config_runner['min_workers'], maximum=config_runner['max_workers'])
        client = Client(cluster)

        logging.info('Waiting for at least one worker...')
        client.wait_for_workers(1)

    else:
        if args.run_dask:
            from dask.distributed import Client, LocalCluster
            if args.skimming:
                cluster_args = {
                    'n_workers': 4,
                    'memory_limit': config_runner['condor_memory'],
                    'threads_per_worker': 1,
                    'dashboard_address': config_runner['dashboard_address'],
                }
            else:
                cluster_args = {
                    'n_workers': 6,
                    'memory_limit': config_runner['condor_memory'],
                    'threads_per_worker': 1,
                    'dashboard_address': config_runner['dashboard_address'],
                }
            cluster = LocalCluster(**cluster_args)
            client = Client(cluster)

    worker_initializer = WorkerInitializer(uproot_xrootd_retry_delays=config_runner['uproot_xrootd_retry_delays'])
    if client is not None:
        client.register_plugin(worker_initializer)

    executor_args = {
        'schema': config_runner['schema'],
        'savemetrics': True,
        'skipbadfiles': config_runner['skipbadfiles'],
        'xrootdtimeout': 600
    }
    if args.debug:
        logging.info(f"Running iterative executor in debug mode")
        executor = processor.iterative_executor
    elif args.condor or args.run_dask:
        executor_args["client"] = client
        executor_args["align_clusters"] = False
        # disable the progressbar when using Dask which will mess up the logging. use Dask's Dashboard instead
        executor_args["status"] = False
        executor = processor.dask_executor
    else:
        logging.info(f"Running futures executor")
        # to run with processor futures_executor ()
        n_workers = config_runner['workers']
        pool = ProcessPoolExecutor(max_workers=n_workers, initializer=worker_initializer.setup)
        executor_args["pool"] = pool
        executor_args["workers"] = n_workers
        executor = processor.futures_executor

    logging.info(f"Executor arguments:")
    logging.info(pretty_repr(executor_args))
    #
    # Run processor
    #
    processorName = args.processor.split('.')[0].replace("/", '.')
    analysis = getattr(importlib.import_module(
        processorName), config_runner['class_name'])
    logging.info(f"Running processsor: {processorName}")

    tstart = time.time()
    logging.info(f"fileset keys are:")
    logging.info(pretty_repr(fileset.keys()))
    logging.debug(f"fileset is")
    logging.debug(pretty_repr(fileset))

    #
    # Running the job
    #
    @profile
    def run_job():
        output, metrics = processor.run_uproot_job(
            fileset,
            treename='Events',
            processor_instance=analysis(**configs['config']),
            executor=executor,
            executor_args=executor_args,
            chunksize=config_runner['chunksize'],
            maxchunks=config_runner['maxchunks'],
        )
        elapsed = time.time() - tstart
        nEvent = metrics['entries']
        processtime = metrics['processtime']
        logging.info(f'Metrics:')
        logging.info(pretty_repr(metrics))
        logging.info(f'{nEvent/elapsed:,.0f} events/s total '
                     f'({nEvent}/{elapsed})')

        #
        # Saving the output
        #
        if args.skimming:
            # check integrity of the output
            output, complete = integrity_check(fileset, output)
            if not complete and (config_runner["maxchunks"] is None) and not args.test:
                # not merge for incomplete jobs except for testing
                logging.error("The jobs above failed. Merging is skipped.")
            else:
                # merge output into new chunks each have `chunksize` events
                kwargs = dict(
                    base_path=configs["config"]["base_path"],
                    output=output,
                    step=config_runner.get("basketsize", configs["config"]["step"]),
                    chunk_size=config_runner.get(
                        "picosize", config_runner["chunksize"]
                    ),
                )

                if (pico_base_name := configs["config"].get("pico_base_name", None)) is not None:
                    kwargs["pico_base_name"] = pico_base_name

                if "declustering_rand_seed" in configs["config"]:
                    kwargs["pico_base_name"] = f'picoAOD_seed{configs["config"]["declustering_rand_seed"]}'

                if configs['runner'].get("class_name", None) == "SubSampler":
                    kwargs["pico_base_name"] = f'picoAOD_PSData'

                if configs['runner'].get("class_name", None) == "Skimmer" and configs["config"].get("skim4b", False):
                    kwargs["pico_base_name"] = f'picoAOD_fourTag'

                if client is not None:
                    output = client.compute(resize(**kwargs), sync=True)
                else:
                    output = resize(**kwargs, dask=False)
            # only keep file name for each chunk
            for dataset, chunks in output.items():
                chunks['files'] = [str(f.path) for f in chunks['files']]

            elapsed = time.time() - tstart
            nEvent = metrics['entries']
            processtime = metrics['processtime']
            logging.info(f'{nEvent/elapsed:,.0f} events/s total '
                         f'({nEvent}/{elapsed})')

            if client is not None:
                metadata = client.compute(fetch_metadata(fileset, dask=True), sync=True)
            else:
                metadata = fetch_metadata(fileset, dask=False)
            metadata = processor.accumulate(metadata)

            for ikey in metadata:
                if ikey in output:
                    metadata[ikey].update(output[ikey])
                    metadata[ikey]['reproducible'] = {
                        'date': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                        'hash': args.githash if args.githash else get_git_revision_hash(),
                        'args': str(args),
                        'diff': args.gitdiff if args.gitdiff else str(get_git_diff()),
                    }

                    if config_runner["data_tier"] in ['picoAOD'] and "genEventSumw" in fileset[ikey]["metadata"]:
                        metadata[ikey]["sumw"] = fileset[ikey]["metadata"]["genEventSumw"]

            args.output_file = 'picoaod_datasets.yml' if args.output_file.endswith(
                'coffea') else args.output_file
            dfile = f'{args.output_path}/{args.output_file}'
            yaml.dump(metadata, open(dfile, 'w'), default_flow_style=False)
            logging.info(f'Saving metadata file {dfile}')

        else:
            #
            # Adding reproducible info
            #
            output['reproducible'] = {
                args.output_file: {
                    'date': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                    'hash': args.githash if args.githash else get_git_revision_hash(),
                    'args': args,
                    'diff': args.gitdiff if args.gitdiff else get_git_diff(),
                }
            }

            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)

            #
            # Save friend tree metadata if exists
            #
            friend_base = config_runner["friend_base"] or configs["config"].get(
                config_runner["friend_base_argname"], None
            )
            friends: dict[str, Friend] = output.get("friends", None)
            if friend_base is not None and friends is not None:
                if args.run_dask:
                    merged_friends = client.compute(
                        {
                            k: friends[k].merge(
                                step=config_runner["friend_merge_step"],
                                base_path=friend_base,
                                naming=_friend_merge_name,
                                clean=False,
                                dask=True,
                            )
                            for k in friends
                        },
                        sync=True,
                    )
                    for v in friends.values():
                        v.reset(confirm=False)
                    friends = merged_friends
                else:
                    for k, v in friends.items():
                        friends[k] = v.merge(
                            step=config_runner["friend_merge_step"],
                            base_path=friend_base,
                            naming=_friend_merge_name,
                        )
                from base_class.system.eos import EOS
                from base_class.utils.json import DefaultEncoder
                metafile = f'{args.output_path}/{args.output_file.replace("coffea", "json")}'
                # metafile = EOS(friend_base) / f'{config_runner["friend_metafile"]}.json'
                with fsspec.open(metafile, "wt") as f:
                    json.dump(friends, f, cls=DefaultEncoder)
                logging.info("The following frends trees are created:")
                logging.info(pretty_repr([*friends.keys()]))
                logging.info(f"Saved friend tree metadata to {metafile}")

            #
            #  Saving file
            #
            if config_runner['write_coffea_output']:
                hfile = f'{args.output_path}/{args.output_file}'
                logging.info(f'Saving file {hfile}')
                save(output, hfile)

    #
    # Run dask performance only in dask jobs
    #
    if args.run_dask:
        dask_report_file = f'/tmp/coffea4bees-dask-report-{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.html'
        with performance_report(filename=dask_report_file):
            run_job()
        try:
            cluster.close()
        except RuntimeError:
            ...
        try:
            client.close()
        except RuntimeError:
            ...
        logging.info(f'Dask performace report saved in {dask_report_file}')
    else:
        run_job()

    if pool is not None:
        pool.shutdown(wait=True)
