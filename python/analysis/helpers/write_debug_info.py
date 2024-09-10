

def add_debug_info_to_output(event, processOutput):
    # passSR = (selev["quadJet_selected"].SR)
    # passSR = (selev["SR"])


    out_data = {}
    # out_data["SvB"    ] = selev["SvB_MA"].ps[passSR]
    # out_data["event"  ] = selev["event"][passSR]
    # out_data["run"    ] = selev["run"][passSR]

    #debug_mask = ((event["event"] == 66688  ) |
    #              (event["event"] == 249987 ) |
    #              (event["event"] == 121603 ) |
    #              (event["event"] == 7816   ) |
    #              (event["event"] == 25353  ) |
    #              (event["event"] == 165389 ) |
    #              (event["event"] == 293138 ) |
    #              (event["event"] == 150164 ) |
    #              (event["event"] == 262806 ) |
    #              (event["event"] == 281111 ) )


    out_data["passJetMult_event"  ]    = event["event"][event.passJetMult]
    out_data["passJetMult_run"    ]    = event["run"][event.passJetMult]
    out_data["passJetMult_jet_pt"    ] = event.Jet.pt[event.passJetMult].to_list()
    out_data["passJetMult_jet_eta"   ] = event.Jet.eta[event.passJetMult].to_list()
    out_data["passJetMult_jet_phi"   ] = event.Jet.phi[event.passJetMult].to_list()
    out_data["passJetMult_jet_pu"    ] = event.Jet.pileup[event.passJetMult].to_list()
    out_data["passJetMult_jet_jetId" ] = event.Jet.jetId[event.passJetMult].to_list()
    out_data["passJetMult_jet_lep"   ] = event.Jet.lepton_cleaned[event.passJetMult].to_list()

    out_data["passDiJetMass_event"  ]    = event["event"][event.passDiJetMass]
    out_data["passDiJetMass_run"    ]    = event["run"][event.passDiJetMass]
    out_data["passDiJetMass_jet_pt"    ] = event.Jet.pt[event.passDiJetMass].to_list()
    out_data["passDiJetMass_jet_eta"   ] = event.Jet.eta[event.passDiJetMass].to_list()
    out_data["passDiJetMass_jet_phi"   ] = event.Jet.phi[event.passDiJetMass].to_list()
    out_data["passDiJetMass_jet_pu"    ] = event.Jet.pileup[event.passDiJetMass].to_list()
    out_data["passDiJetMass_jet_jetId" ] = event.Jet.jetId[event.passDiJetMass].to_list()
    out_data["passDiJetMass_jet_lep"   ] = event.Jet.lepton_cleaned[event.passDiJetMass].to_list()


    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)



def add_debug_info_to_output_clustering_inputs(event, jets_for_clustering, processOutput):

    out_data = {}

    out_data["clusteringInputs_event"  ]    = event["event"][0:10]
    out_data["clusteringInputs_run"    ]    = event["run"][0:10]
    out_data["clusteringInputs_jet_pt"    ] = jets_for_clustering.pt  [0:10].to_list()
    out_data["clusteringInputs_jet_eta"   ] = jets_for_clustering.eta [0:10].to_list()
    out_data["clusteringInputs_jet_phi"   ] = jets_for_clustering.phi [0:10].to_list()
    out_data["clusteringInputs_jet_mass"  ] = jets_for_clustering.mass[0:10].to_list()

    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


def add_debug_info_to_output_clustering_outputs(event, clustered_jets, processOutput):

    out_data = {}

    out_data["clusteringOutputs_event"  ]    = event["event"][0:10]
    out_data["clusteringOutputs_run"    ]    = event["run"][0:10]
    out_data["clusteringOutputs_jet_pt"    ] = clustered_jets.pt  [0:10].to_list()
    out_data["clusteringOutputs_jet_eta"   ] = clustered_jets.eta [0:10].to_list()
    out_data["clusteringOutputs_jet_phi"   ] = clustered_jets.phi [0:10].to_list()
    out_data["clusteringOutputs_jet_mass"  ] = clustered_jets.mass[0:10].to_list()
    out_data["clusteringOutputs_jet_flavor"] = clustered_jets.jet_flavor[0:10].to_list()

    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


def add_debug_info_to_output_declustering_outputs(event, declustered_jets, processOutput):

    out_data = {}

    out_data["declusteredJets_event"  ]      = event["event"][0:10]
    out_data["declusteredJets_run"    ]      = event["run"][0:10]
    out_data["declusteredJets_jet_pt"    ]   = declustered_jets.pt  [0:10].to_list()
    out_data["declusteredJets_jet_eta"   ]   = declustered_jets.eta [0:10].to_list()
    out_data["declusteredJets_jet_phi"   ]   = declustered_jets.phi [0:10].to_list()
    out_data["declusteredJets_jet_mass"  ]   = declustered_jets.mass[0:10].to_list()
    out_data["declusteredJets_jet_flavor"] = declustered_jets.jet_flavor[0:10].to_list()
    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)
