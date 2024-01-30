# FIXME maybe move to the same file as `runner.py`?

import json

import dask
from coffea.nanoevents import NanoAODSchema
from coffea.processor import dask_executor, run_uproot_job
from dask.distributed import Client
from skimmer.picoaod import PicoAOD, fetch_metadata, resize


class Skimmer(PicoAOD):
    def select(self, events):
        return events['event'] % 100 == 0  # FIXME define selection


if __name__ == '__main__':
    fileset = ...  # FIXME construct fileset

    # fetch metadata, save to a separate file
    metadata = fetch_metadata(fileset)

    base = ...  # FIXME setup base path e.g. root://cmseos.fnal.gov//store/user/.../HH4b/
    cluster = ...  # FIXME init cluster
    step = 80_000  # FIXME setup io step size, depends on memory usage
    client = Client(cluster)
    chunksize = 200_000  # FIXME setup chunksize, depends on memory usage
    output = run_uproot_job(
        fileset,
        treename='Events',
        processor_instance=Skimmer(
            base_path=base,
            step=step,
            # FIXME collections to skip. e.g. 'Tau' will skip all 'Tau_.*' and 'nTau'
            skip_collections=['CaloMET', 'FatJet', 'FsrPhoton',
                              'Photon', 'SubJet', 'Tau', 'boostedTau'],
            # FIXME branches to skip, use regex
            skip_branches=['btagWeight_.*'],
        ),
        executor=dask_executor,
        executor_args={
            "schema": NanoAODSchema,
            "client": client,
        },
        chunksize=chunksize,
    )
    # merge output into new chunks each have `chunksize` events
    # FIXME can use a different chunksize
    output = dask.compute(resize(base, output, step, chunksize))[0]
    # only keep file name for each chunk
    for dataset, chunks in output.items():
        chunks['files'] = [str(f.path) for f in chunks['files']]

    # FIXME write `output` and `metadata` to a file
    json.dump(output, open('picoAOD.json', 'w'))
    json.dump(metadata, open('metadata.json', 'w'))


# example `output`
# {
#     '{dataset1}': {
#         'total_events': 100000,
#         'saved_events': 100,
#         'files': [
#             'root://cmseos.fnal.gov//store/user/{username}/HH4b/{dataset1}/picoAOD.chunk0.root',
#             'root://cmseos.fnal.gov//store/user/{username}/HH4b/{dataset1}/picoAOD.chunk1.root',
#         ]
#     },
#     '{dataset2}': {
#         'total_events': 200000,
#         'saved_events': 200,
#         'files': [
#             'root://cmseos.fnal.gov//store/user/{username}/HH4b/{dataset2}/picoAOD.chunk0.root',
#             'root://cmseos.fnal.gov//store/user/{username}/HH4b/{dataset2}/picoAOD.chunk1.root',
#             'root://cmseos.fnal.gov//store/user/{username}/HH4b/{dataset2}/picoAOD.chunk2.root',
#             'root://cmseos.fnal.gov//store/user/{username}/HH4b/{dataset2}/picoAOD.chunk3.root',
#         ]
#     }
# }


# example `metadata`
# {
#     '{dataset1}': {'count': 100000, 'sumw': 1000, 'sumw2': 10},
#     '{dataset2}': {'count': 200000, 'sumw': 2000, 'sumw2': 20}
# }
