import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
NanoAODSchema.warn_missing_crossrefs = False
import warnings
warnings.filterwarnings("ignore")
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
from coffea import processor, util
# import hist as shh # https://hist.readthedocs.io/en/latest/
# import hist
import correctionlib
import cachetools
import logging
import copy

# following example here: https://github.com/CoffeaTeam/coffea/blob/master/tests/test_jetmet_tools.py#L529
def init_jet_factory(weight_sets):
    from coffea.lookup_tools import extractor
    extract = extractor()
    extract.add_weight_sets(weight_sets)
    extract.finalize()
    evaluator = extract.make_evaluator()

    from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack, JetResolution, JetResolutionScaleFactor
    jec_stack_names = []
    for key in evaluator.keys():
        jec_stack_names.append(key)
        if 'UncertaintySources' in key:
            jec_stack_names.append(key)

    # print(jec_stack_names)
    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    # print('jec_inputs:')
    # print(jec_inputs)
    jec_stack = JECStack(jec_inputs)
    # print('jec_stack')
    # print(jec_stack.__dict__)
    name_map = jec_stack.blank_name_map
    name_map["JetPt"]    = "pt"
    name_map["JetMass"]  = "mass"
    name_map["JetEta"]   = "eta"
    name_map["JetA"]     = "area"
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw']    = 'pt_raw'
    name_map['massRaw']  = 'mass_raw'
    name_map['Rho']      = 'rho'
    # print(name_map)

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    uncertainties = jet_factory.uncertainties()
    if uncertainties:
        for unc in uncertainties:
            logging.debug(unc)
    else:
        logging.warning('WARNING: No uncertainties were loaded in the jet factory')

    return jet_factory

def jetCorrection( uncorrJets, jercFile="/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jet_jerc.json.gz", campaign="Summer19UL18_V5_MC", jectype="L1L2L3Res", jettype='AK4PFchs' ):

    JECFile = correctionlib.CorrectionSet.from_file(jercFile)
    # preparing jets
    uncorrJets['pt_raw'] = (1 - uncorrJets['rawFactor']) * uncorrJets['pt']
    uncorrJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, uncorrJets.pt)[0]
    j, nj = ak.flatten(uncorrJets), ak.num(uncorrJets)

    if 'L1' in jectype:
        corr = JECFile.compound[f'{campaign}_{jectype}_{jettype}'] if 'L1L2L3' in jectype else JECFile[f'{campaign}_{jectype}_{jettype}']
        flat_jec = corr.evaluate( j['area'], j['eta'], j['pt_raw'], j['rho']  )
    else:
        corr = JECFile[f'{campaign}_{jectype}_{jettype}']
        flat_jec = corr.evaluate( j['eta'], j['pt_raw']  )
    jec = ak.unflatten(flat_jec, nj)

    correctP4Jets = uncorrJets * jec
    correctJets = copy.deepcopy(uncorrJets)
    correctJets['pt'] = correctP4Jets.pt
    correctJets['mass'] = correctP4Jets.mass

    return correctJets

