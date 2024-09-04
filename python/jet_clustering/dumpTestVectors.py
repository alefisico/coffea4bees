import awkward as ak

def dumpTestVectors_bbj(chunk, selev, jets_for_clustering):
    split_name = "splitting_1b1j/1b0j"
    print(selev.fields,"\n")
    print(f'{chunk} num splitting {ak.num(selev[split_name])}')
    print(f'{chunk} mask {ak.num(selev[split_name]) > 0}')
    bbj_mask = ak.num(selev[split_name]) > 0
    jets_for_clustering_bbj = jets_for_clustering[bbj_mask]
    n_jets_clustering = len(jets_for_clustering_bbj)
    post_fix = "_bbj"
    print(f'{chunk}\n\n')
    print(f'{chunk} self.input_jet_pt{post_fix}      = {[jets_for_clustering_bbj[iE].pt.tolist()               for iE in range(n_jets_clustering)]}')
    print(f'{chunk} self.input_jet_eta{post_fix}     = {[jets_for_clustering_bbj[iE].eta.tolist()              for iE in range(n_jets_clustering)]}')
    print(f'{chunk} self.input_jet_phi{post_fix}     = {[jets_for_clustering_bbj[iE].phi.tolist()              for iE in range(n_jets_clustering)]}')
    print(f'{chunk} self.input_jet_mass{post_fix}    = {[jets_for_clustering_bbj[iE].mass.tolist()             for iE in range(n_jets_clustering)]}')
    print(f'{chunk} self.input_jet_flavor{post_fix}  = {[jets_for_clustering_bbj[iE].jet_flavor.tolist()       for iE in range(n_jets_clustering)]}')
    print(f'{chunk} self.input_btagDeepFlavB{post_fix}  = {[jets_for_clustering_bbj[iE].btagDeepFlavB.tolist() for iE in range(n_jets_clustering)]}')
    print(f'{chunk}\n\n')
