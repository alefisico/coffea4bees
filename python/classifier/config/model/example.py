# TODO SvB, FvT, DvT3, DvT4
# TODO for test only, will be removed or rewritten
# TODO work in progress, match new skimmed data
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypedDict

if TYPE_CHECKING:
    import torch


class features:
    candidate_jet = ['pt', 'eta', 'phi', 'mass']
    candidate_jet_max = 4
    other_jet = ['pt', 'eta', 'phi', 'mass', 'isSelJet']
    other_jet_max = 4
    ancillary = ['nSelJets']
    ancillary += ['year']
    # ancillary += ['xW', 'xbW']


class TrainingBatch(TypedDict):
    event_offset: Annotated[
        torch.ByteTensor,
        ('*',)]
    label_index: Annotated[
        torch.ByteTensor,
        ('*',)]
    region_index: Annotated[
        torch.ByteTensor,
        ('*',)]
    weight: Annotated[
        torch.FloatTensor,
        ('*',)]
    ancillary: Annotated[
        torch.FloatTensor,
        ('*', len(features.ancillary))]
    candidate_jet: Annotated[
        torch.FloatTensor,
        ('*', len(features.candidate_jet)*features.candidate_jet_max)]
    other_jet: Annotated[
        torch.FloatTensor,
        ('*', len(features.other_jet)*features.other_jet_max)]
