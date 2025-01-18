

def add_debug_info_to_output(event, processOutput, weights, list_weight_names, analysis_selections):
    # passSR = (selev["quadJet_selected"].SR)
    # passSR = (selev["SR"])


    out_data = {}
    # out_data["SvB"    ] = selev["SvB_MA"].ps[passSR]
    out_data["event"  ] = event["event"]#[passSR]
    out_data["run"    ] = event["run"]  #[passSR]
    out_data["fourTag"    ] = event["fourTag"]  #[passSR]

    out_data["passPreSel"    ] = event["passJetMult"]
    out_data["lumimask"    ] = event["lumimask"]
    out_data["passNoiseFilter"    ] = event["passNoiseFilter"]
    out_data["passHLT"    ] = event["passHLT"]
    out_data["passJetMult"    ] = event["passJetMult"]

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


    #print(f"\n {event.Jet.pt[event.passJetMult].to_list()[0:5]} \n")

    #out_data["passJetMult_event"  ]    = event["event"][event.passJetMult]
    #out_data["passJetMult_run"    ]    = event["run"][event.passJetMult]
    #out_data["passJetMult_passDiJetMass"    ]    = event["passDiJetMass"][event.passJetMult]
    #out_data["passJetMult_weight" ]    = event["weight"]
    #
    #for _w in list_weight_names:
    #    print(f"adding {_w}\n")
    #    out_data[f"passJetMult_weight_{_w}"] = weights.partial_weight(include=[_w])[analysis_selections]


    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)




def add_debug_Run3_synthetic_data(event, processOutput):

    out_data = {}

    #debug_mask = event.region > 2
    out_data["SR_event"  ]    = event["event"]
    out_data["SR_run"    ]    = event["run"]
    out_data["SR_region"    ] = event["region"]
    out_data["SR_jet_pt"    ] = event.canJet.pt  .to_list()
    out_data["SR_jet_eta"   ] = event.canJet.eta .to_list()
    out_data["SR_jet_phi"   ] = event.canJet.phi .to_list()
    out_data["SR_jet_mass"  ] = event.canJet.mass.to_list()

    out_data["SR_selJet_pt"    ] = event.selJet.pt  .to_list()
    out_data["SR_selJet_eta"   ] = event.selJet.eta .to_list()
    out_data["SR_selJet_phi"   ] = event.selJet.phi .to_list()
    out_data["SR_selJet_mass"  ] = event.selJet.mass.to_list()

    out_data["SR_selJet_no_bRegCorr_pt"    ] = event.selJet_no_bRegCorr.pt  .to_list()
    out_data["SR_selJet_no_bRegCorr_eta"   ] = event.selJet_no_bRegCorr.eta .to_list()
    out_data["SR_selJet_no_bRegCorr_phi"   ] = event.selJet_no_bRegCorr.phi .to_list()
    out_data["SR_selJet_no_bRegCorr_mass"  ] = event.selJet_no_bRegCorr.mass.to_list()


    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)



def add_debug_Run3_declustering(event, jets_for_clustering, declustered_jets, processOutput):

    out_data = {}

    #debug_mask = event.region > 2
    out_data["event"  ]    = event["event"]
    out_data["run"    ]    = event["run"]

    out_data["selJet_pt"    ] = event.selJet.pt  .to_list()
    out_data["selJet_eta"   ] = event.selJet.eta .to_list()
    out_data["selJet_phi"   ] = event.selJet.phi .to_list()
    out_data["selJet_mass"  ] = event.selJet.mass.to_list()

    out_data["Jet_bCalib_pt"    ] = event.Jet_bCalib.pt  .to_list()
    out_data["Jet_bCalib_pt_raw" ] = event.Jet_bCalib.pt_raw  .to_list()
    out_data["Jet_bCalib_mass"  ] = event.Jet_bCalib.mass.to_list()

    out_data["Jet_nonbCalib_pt"    ] = event.Jet_nonbCalib.pt  .to_list()
    out_data["Jet_nonbCalib_pt_raw"    ] = event.Jet_nonbCalib.pt_raw  .to_list()
    out_data["Jet_nonbCalib_mass"  ] = event.Jet_nonbCalib.mass.to_list()


    out_data["selJet_no_bRegCorr_pt"    ] = event.selJet_no_bRegCorr.pt  .to_list()
    out_data["selJet_no_bRegCorr_pt_preCalib"    ] = event.selJet_no_bRegCorr.pt_preCalib  .to_list()
    out_data["selJet_no_bRegCorr_eta"   ] = event.selJet_no_bRegCorr.eta .to_list()
    out_data["selJet_no_bRegCorr_phi"   ] = event.selJet_no_bRegCorr.phi .to_list()
    out_data["selJet_no_bRegCorr_mass"  ] = event.selJet_no_bRegCorr.mass.to_list()
    out_data["selJet_no_bRegCorr_mass_preCalib"  ] = event.selJet_no_bRegCorr.mass_preCalib.to_list()


    out_data["input_jet_pt"    ] = jets_for_clustering.pt  .to_list()
    out_data["input_jet_eta"   ] = jets_for_clustering.eta .to_list()
    out_data["input_jet_phi"   ] = jets_for_clustering.phi .to_list()
    out_data["input_jet_mass"  ] = jets_for_clustering.mass.to_list()

    out_data["output_jet_pt"    ] = declustered_jets.pt  .to_list()
    out_data["output_jet_eta"   ] = declustered_jets.eta .to_list()
    out_data["output_jet_phi"   ] = declustered_jets.phi .to_list()
    out_data["output_jet_mass"  ] = declustered_jets.mass.to_list()


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


def debug_three_tag_events(event, processOutput):

    for iEvent in range(5):

        print(f'event num: {event["event"][event.threeTag][iEvent]}')
        print(f'event run: {event["run"][event.threeTag][iEvent]}')
        print(f'event fourTag: {event.fourTag[event.threeTag][iEvent]}')
        print(f'event threeTag: {event.fourTag[event.threeTag][iEvent]}')
        print(f'event jetPt:  {[i for i in event.Jet.pt    [event.threeTag][iEvent]]}')
        print(f'event eta:    {[i for i in event.Jet.eta   [event.threeTag][iEvent]]}')
        print(f'event phi:    {[i for i in event.Jet.phi   [event.threeTag][iEvent]]}')
        print(f'event selected: {[i for i in event.Jet.selected[event.threeTag][iEvent]]}')
        print(f'\t event pileup: {[i for i in event.Jet.pileup[event.threeTag][iEvent]]}')
        print(f'\t\t event puId: {[i for i in event.Jet.puId[event.threeTag][iEvent]]}')
        print(f'\t event jetId: {[i for i in event.Jet.jetId[event.threeTag][iEvent]]}')
        print(f'\t event lepton_cleaned: {[i for i in event.Jet.lepton_cleaned[event.threeTag][iEvent]]}')
        print(f'event tagged: {[i for i in event.Jet.tagged[event.threeTag][iEvent]]}')
        print(f'event btagScore: {[i for i in event.Jet.btagScore[event.threeTag][iEvent]]}')
        print("\n")

    #out_data["passJetMult_event"  ]    = event["event"][event.threeTag]
    #out_data["passJetMult_run"    ]    = event["run"][event.passJetMult]
    #out_data["passJetMult_jet_pt"    ] = event.Jet.pt[event.passJetMult].to_list()
    #out_data["passJetMult_jet_eta"   ] = event.Jet.eta[event.passJetMult].to_list()
    #out_data["passJetMult_jet_phi"   ] = event.Jet.phi[event.passJetMult].to_list()
    #out_data["passJetMult_jet_pu"    ] = event.Jet.pileup[event.passJetMult].to_list()
    #out_data["passJetMult_jet_jetId" ] = event.Jet.jetId[event.passJetMult].to_list()
    #out_data["passJetMult_jet_lep"   ] = event.Jet.lepton_cleaned[event.passJetMult].to_list()
