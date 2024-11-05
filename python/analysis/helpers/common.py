import awkward as ak
import numpy as np
from coffea.nanoevents import NanoAODSchema

NanoAODSchema.warn_missing_crossrefs = False
import warnings

warnings.filterwarnings("ignore")
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)
import logging
import pickle
import correctionlib
import tarfile
import os

def extract_jetmet_tar_files(tar_file_name: str=None,
                            jet_type: str='AK4PFchs'
                            ):
  """Extracts a tar.gz file to a specified path and returns a list of extracted files with their locations.

  Args:
    tar_file_name: The name of the tar.gz file.
    jet_type: The type of jet to apply correction

  Returns:
    A list of tuples, where each tuple contains the file name and its full path.
  """

  extracted_files = []
  extract_path = f"/tmp/{os.getenv('USER', 'default_user')}/coffea4bees/"

  with tarfile.open(tar_file_name, "r:gz") as tar:
    for member in tar.getmembers():
      if member.isfile():
        # Extract the file to the specified path
        member.name = os.path.basename(member.name)  # Remove any directory structure from the archive

        new_file_name = member.name
        if 'Puppi' in jet_type and jet_type in member.name:  ### this is only for Run3, temporary fix
          new_file_name = member.name.replace('_', '', 1)
          member.name = new_file_name

        new_file_path = os.path.join(extract_path, new_file_name)

        if not os.path.isfile(new_file_path):
          tar.extract(member, path=extract_path)

          old_file_path = os.path.join(extract_path, member.name)
          #new_file_name = member.name.replace('_', '', 1)
          #new_file_path = os.path.join(extract_path, new_file_name)
          #print(f"new_file_name is {new_file_name}\n")
          #print(f"member.name is {member.name}\n")
          # Rename the file if the name was changed
          if old_file_path != new_file_path:
            os.rename(old_file_path, new_file_path)


        # Get the full path of the extracted file
        file_path = os.path.join(extract_path, member.name)
        if jet_type in member.name:
            extracted_files.append(f"* * {file_path}")

  return extracted_files

# following example here: https://github.com/CoffeaTeam/coffea/blob/master/tests/test_jetmet_tools.py#L529
def apply_jerc_corrections( event,
                    corrections_metadata: dict = {},
                    run_systematics: bool = False,
                    isMC: bool = False,
                    dataset: str = None,
                    jec_levels: list = ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
                    jer_levels: list = ["PtResolution", "SF"],
                    ):

    logging.info(f"Applying JEC/JER corrections for {dataset}")

    jet_type = 'AK4PFchs' if '202' not in dataset else 'AK4PFPuppiPNetRegressionPlusNeutrino'
    jec_file = corrections_metadata['JEC_MC'] if isMC else corrections_metadata['JEC_DATA'][dataset[-1]]
    extracted_files = extract_jetmet_tar_files(jec_file, jet_type=jet_type)
    if run_systematics: jec_levels.append("RegroupedV2")
    weight_sets = list(set([file for level in jec_levels for file in extracted_files if level in file]))  ## list(set()) to remove duplicates

    if isMC and ('202' not in dataset):
        jer_file = corrections_metadata["JER_MC"]
        extracted_files = extract_jetmet_tar_files(jer_file, jet_type=jet_type)
        weight_sets += [file for level in jer_levels for file in extracted_files if level in file]

    logging.debug(f"For {dataset}, applying these corrections: {weight_sets}")

    event['Jet', 'pt_raw']    = (1 - event.Jet.rawFactor) * event.Jet.pt
    event['Jet', 'mass_raw']  = (1 - event.Jet.rawFactor) * event.Jet.mass
    nominal_jet = event.Jet
    if isMC: nominal_jet['pt_gen'] = ak.values_astype(ak.fill_none(nominal_jet.matched_gen.pt, 0), np.float32)

    nominal_jet['rho'] = ak.broadcast_arrays((event.Rho.fixedGridRhoFastjetAll if 'Rho' in event.fields else event.fixedGridRhoFastjetAll), nominal_jet.pt)[0]

    from coffea.lookup_tools import extractor
    extract = extractor()
    extract.add_weight_sets(weight_sets)
    extract.finalize()
    evaluator = extract.make_evaluator()

    from base_class.jetmet_tools import CorrectedJetsFactory
    from coffea.jetmet_tools import JECStack
    jec_stack_names = []
    for key in evaluator.keys():
        jec_stack_names.append(key)
        if 'UncertaintySources' in key:
            jec_stack_names.append(key)

    logging.debug(jec_stack_names)
    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    logging.debug('jec_inputs:')
    logging.debug(jec_inputs)
    jec_stack = JECStack(jec_inputs)
    logging.debug('jec_stack')
    logging.debug(jec_stack.__dict__)
    name_map = jec_stack.blank_name_map
    name_map["JetPt"]    = "pt"
    name_map["JetMass"]  = "mass"
    name_map["JetEta"]   = "eta"
    name_map["JetA"]     = "area"
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw']    = 'pt_raw'
    name_map['massRaw']  = 'mass_raw'
    name_map['Rho']      = 'rho'
    logging.debug(name_map)

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    uncertainties = jet_factory.uncertainties()
    if uncertainties:
        for unc in uncertainties:
            logging.debug(unc)
    else:
        logging.warning('WARNING: No uncertainties were loaded in the jet factory')

    jet_variations = jet_factory.build(nominal_jet, event.event)

    return jet_variations

##### JER needs to be included
def jet_corrections( uncorr_jets,
                    fixedGridRhoFastjetAll,
                    isMC,
                    jercFile="/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jet_jerc.json.gz",
                    data_campaign="Summer19UL18",
                    jec_campaign='V5_MC',
                    jer_campaign='JRV2_MC',
                    jec_type=["L1L2L3Res"],  ####, "PtResolution", 'ScaleFactor'
                    jettype='AK4PFchs',
                    variation='nom'
                    ):

    JECFile = correctionlib.CorrectionSet.from_file(jercFile)
    # preparing jets
    uncorr_jets['pt_raw'] = (1 - uncorr_jets['rawFactor']) * uncorr_jets['pt']
    uncorr_jets['mass_raw'] = (1 - uncorr_jets['rawFactor']) * uncorr_jets['mass']
    uncorr_jets['rho'] = ak.broadcast_arrays(fixedGridRhoFastjetAll, uncorr_jets.pt)[0]
    j, nj = ak.flatten(uncorr_jets), ak.num(uncorr_jets)

    jec_campaign = f'{jec_campaign}_{"MC" if isMC else "DATA"}'

    total_flat_jec = np.ones( len(j), dtype="float32" )
    for ijec in jec_type:

        if 'L1' in ijec:
            corr = JECFile.compound[f'{data_campaign}_{jec_campaign}_{ijec}_{jettype}'] if 'L1L2L3' in ijec else JECFile[f'{data_campaign}_{jec_campaign}_{ijec}_{jettype}']
            flat_jec = corr.evaluate( j['area'], j['eta'], j['pt_raw'], j['rho']  )
        else:
            corr = JECFile[f'{data_campaign}_{jec_campaign}_{ijec}_{jettype}']
            flat_jec = corr.evaluate( j['eta'], j['pt_raw']  )
        total_flat_jec *= flat_jec
    jec = ak.unflatten(total_flat_jec, nj)

    corr_jets = uncorr_jets
    corr_jets['jet_energy_correction'] = jec
    corr_jets['pt'] = corr_jets.pt_raw * jec
    corr_jets['mass'] = corr_jets.mass_raw * jec

    return corr_jets

def apply_jet_veto_maps( corrections_metadata, jets ):
    '''
    taken from https://github.com/PocketCoffea/PocketCoffea/blob/main/pocket_coffea/lib/cut_functions.py#L65
    '''

    mask_for_VetoMap = (
        ((jets.jetId & 2)==2) # Must fulfill tight jetId
        & (abs(jets.eta) < 5.19) # Must be within HCal acceptance
        & ((jets["neEmEF"]+jets["chEmEF"])<0.9) # Energy fraction not dominated by ECal
    )
    if 'muonSubtrFactor' in jets.fields:  ### AGE: this should be temporary
        mask_for_VetoMap = mask_for_VetoMap & (jets.muonSubtrFactor < 0.8) # May no be Muons misreconstructed as jets
    masked_jets = jets[mask_for_VetoMap]

    corr = correctionlib.CorrectionSet.from_file(corrections_metadata['file'])[corrections_metadata['tag']]

    etaFlat, phiFlat, etaCounts = ak.flatten(masked_jets.eta), ak.flatten(masked_jets.phi), ak.num(masked_jets.eta)
    phiFlat = np.clip(phiFlat, -3.14159, 3.14159) # Needed since no overflow included in phi binning
    weight = ak.unflatten(
        corr.evaluate("jetvetomap", etaFlat, phiFlat),
        counts=etaCounts,
    )
    eventMask = ak.sum(weight, axis=-1)==0 # if at least one jet is vetoed, reject it event
    return ak.where(ak.is_none(eventMask), False, eventMask)


def mask_event_decision(event, decision='OR', branch='HLT', list_to_mask=[''], list_to_skip=['']):
    '''
    Takes event.branch and passes an boolean array mask with the decisions of all the list_to_mask
    '''

    tmp_list = []
    if branch in event.fields:
        for i in list_to_mask:
            if i in event[branch].fields:
                tmp_list.append( event[branch][i] )
            elif i in list_to_skip: continue
            else: logging.warning(f'\n{i} branch not in {branch} for event.')
    else: logging.warning(f'\n{branch} branch not in event.')
    tmp_array = np.array( tmp_list )

    if decision.lower().startswith('or'): decision_array = np.any( tmp_array, axis=0 )
    else: decision_array = np.all( tmp_array, axis=0 )

    return decision_array

def apply_btag_sf( jets,
                  correction_file='data/JEC/BTagSF2016/btagging_legacy16_deepJet_itFit.json.gz',
                  correction_type="deepJet_shape",
                  btag_uncertainties = None,
                  dataset = '',
                  btagSF_norm_file='ZZ4b/nTupleAnalysis/weights/btagSF_norm.pkl',
                  ):
    '''
    Can be replace with coffea.btag_tools when btag_tools accept jsonpog files
    '''

    btagSF = correctionlib.CorrectionSet.from_file(correction_file)[correction_type]

    weights = {}
    j, nj = ak.flatten(jets), ak.num(jets)
    hf, eta, pt, tag = ak.to_numpy(j.hadronFlavour), ak.to_numpy(abs(j.eta)), ak.to_numpy(j.pt), ak.to_numpy(j.btagScore)

    cj_bl = jets[jets.hadronFlavour!=4]
    nj_bl = ak.num(cj_bl)
    cj_bl = ak.flatten(cj_bl)
    hf_bl, eta_bl, pt_bl, tag_bl = ak.to_numpy(cj_bl.hadronFlavour), ak.to_numpy(abs(cj_bl.eta)), ak.to_numpy(cj_bl.pt), ak.to_numpy(cj_bl.btagScore)
    SF_bl= btagSF.evaluate('central', hf_bl, eta_bl, pt_bl, tag_bl)
    SF_bl = ak.unflatten(SF_bl, nj_bl)
    SF_bl = np.prod(SF_bl, axis=1)

    cj_c = jets[jets.hadronFlavour==4]
    nj_c = ak.num(cj_c)
    cj_c = ak.flatten(cj_c)
    hf_c, eta_c, pt_c, tag_c = ak.to_numpy(cj_c.hadronFlavour), ak.to_numpy(abs(cj_c.eta)), ak.to_numpy(cj_c.pt), ak.to_numpy(cj_c.btagScore)
    SF_c= btagSF.evaluate('central', hf_c, eta_c, pt_c, tag_c)
    SF_c = ak.unflatten(SF_c, nj_c)
    SF_c = np.prod(SF_c, axis=1)

    ### btag norm
    try:
        with open(btagSF_norm_file, 'rb') as f:
            btagSF_norm = pickle.load(f)[dataset]
            logging.info(f'btagSF_norm {btagSF_norm}')
    except FileNotFoundError:
        btagSF_norm = 1.0

    btag_var = [ 'central' ]
    if btag_uncertainties:
        btag_var += [ f'{updown}_{btagvar}' for updown in ['up', 'down',] for btagvar in btag_uncertainties ]
    for sf in btag_var:
        if sf == 'central':
            SF = btagSF.evaluate('central', hf, eta, pt, tag)
            SF = ak.unflatten(SF, nj)
            # hf = ak.unflatten(hf, nj)
            # pt = ak.unflatten(pt, nj)
            # eta = ak.unflatten(eta, nj)
            # tag = ak.unflatten(tag, nj)
            # for i in range(len(selev)):
            #     for j in range(nj[i]):
            #         print(f'jetPt/jetEta/jetTagScore/jetHadronFlavour/SF = {pt[i][j]}/{eta[i][j]}/{tag[i][j]}/{hf[i][j]}/{SF[i][j]}')
            #     print(np.prod(SF[i]))
            SF = np.prod(SF, axis=1)
        if '_cf' in sf:
            SF = btagSF.evaluate(sf, hf_c, eta_c, pt_c, tag_c)
            SF = ak.unflatten(SF, nj_c)
            SF = SF_bl * np.prod(SF, axis=1) # use central value for b,l jets
        if '_hf' in sf or '_lf' in sf or '_jes' in sf:
            SF = btagSF.evaluate(sf, hf_bl, eta_bl, pt_bl, tag_bl)
            SF = ak.unflatten(SF, nj_bl)
            SF = SF_c * np.prod(SF, axis=1) # use central value for charm jets

        weights[f'btagSF_{sf}'] = SF * btagSF_norm

    logging.debug(weights)
    return weights


def drClean(coll1,coll2,cone=0.4):

    from coffea.nanoevents.methods import vector
    j_eta = coll1.eta
    j_phi = coll1.phi
    l_eta = coll2.eta
    l_phi = coll2.phi

    j_eta, l_eta = ak.unzip(ak.cartesian([j_eta, l_eta], nested=True))
    j_phi, l_phi = ak.unzip(ak.cartesian([j_phi, l_phi], nested=True))
    delta_eta = j_eta - l_eta
    delta_phi = vector._deltaphi_kernel(j_phi,l_phi)
    dr = np.hypot(delta_eta, delta_phi)
    nolepton_mask = ~ak.any(dr < cone, axis=2)
    jets_noleptons = coll1[nolepton_mask]
    return [jets_noleptons, nolepton_mask]

def update_events(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out
