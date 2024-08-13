import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
from jet_clustering.sample_jet_templates import sample_PDFs_vs_pT

def extract_all_parentheses_substrings(s):
    substrings = []
    start_indices = []
    counter = 0

    for i, char in enumerate(s):
        if char == '(':
            if counter == 0:
                start_indices.append(i)
            counter += 1
        elif char == ')':
            counter -= 1
            if counter == 0:
                start_index = start_indices.pop(0)
                substrings.append(s[start_index:i+1])

    return substrings


def children_jet_flavors(comb_flavor):

    if len(comb_flavor) < 2:
        print(f"ERROR len of combined flavor is too low {len(comb_flavor)}  {comb_flavor}")

    sub_combs = extract_all_parentheses_substrings(comb_flavor)

    if len(sub_combs) == 0:
        child_A = comb_flavor[0]
        child_B = comb_flavor[1]
    elif len(sub_combs) == 1:
        child_A = sub_combs[0][1:-1] # the 1:-1 remove the leading and trailing parems
        child_B = str(comb_flavor).replace(sub_combs[0],"")
    elif len(sub_combs) == 2:
        child_A = sub_combs[0][1:-1]
        child_B = sub_combs[1][1:-1]
    else:
        print(f"ERROR comb_flavor is {comb_flavor} sub_combs is {sub_combs} len {len(sub_combs)}")

    return child_A, child_B


def get_splitting_summary(comb_flavor):

    childA, childB = children_jet_flavors(comb_flavor)

    n_b_A = childA.count("b")
    n_j_A = childA.count("j")

    n_b_B = childB.count("b")
    n_j_B = childB.count("j")


    return (n_b_A, n_j_A), (n_b_B, n_j_B)


def get_splitting_name(comb_flavor):

    A_stats, B_stats = get_splitting_summary(comb_flavor)
    #if n_Xs[0] > 4 or (n_Xs[0] + n_Xs[1]) > 5:
    #    return f"{n_Xs[0]}/{n_Xs[1]}"


    return f"{A_stats[0]}b{A_stats[1]}j/{B_stats[0]}b{B_stats[1]}j"



def get_list_of_combined_jet_types(jets):
    """
      returns a list of all the splitting types that are the results of a combination
        (ie: no b or j )
    """
    all_jet_types =  get_list_of_splitting_types(jets)
    splitting_types = []
    for _s in all_jet_types:

        if len(_s) == 1:
            continue

        splitting_types.append(_s)

    return splitting_types


def get_list_of_all_sub_splittings(splitting):
    """
      returns a list of all the sub splitting types (including the original)
    """
    if len(splitting) > 1:
        childA, childB = children_jet_flavors(splitting)
        return [splitting] + get_list_of_all_sub_splittings(childA) + get_list_of_all_sub_splittings(childB)

    return []



def get_list_of_ISR_splittings(splitting_types):

    ISR_splittings = []
    for _s in splitting_types:

        if len(_s) == 1:
            continue

        child_A, child_B = children_jet_flavors(_s)

        child_A_nBs = child_A.count("b")
        child_B_nBs = child_B.count("b")

        #
        #  All splittings are ISR unless there is a b in both children
        #
        if(child_A_nBs > 0 and child_B_nBs > 0):
            continue

        ISR_splittings.append(_s)


    return ISR_splittings


def get_list_of_splitting_types(splittings):
    unique_splittings = set(ak.flatten(splittings.jet_flavor).to_list())
    return list(unique_splittings)

def get_list_of_splitting_names(splittings):
    unique_splittings = set(ak.flatten(splittings.splitting_name).to_list())
    return list(unique_splittings)



def compute_decluster_variables(clustered_splittings):

    #
    # z-axis
    #
    z_axis      = ak.zip({"x": 0, "y": 0, "z": 1,}, with_name="ThreeVector", behavior=vector.behavior,)
    boost_vec_z = ak.zip({"x": 0, "y": 0, "z": clustered_splittings.boostvec.z,}, with_name="ThreeVector", behavior=vector.behavior,)

    #
    #  Boost to pz0
    #
    clustered_splittings_pz0        = clustered_splittings.boost(-boost_vec_z)
    clustered_splittings_part_A_pz0 = clustered_splittings.part_A.boost(-boost_vec_z)
    clustered_splittings_part_B_pz0 = clustered_splittings.part_B.boost(-boost_vec_z)

    comb_z_plane_hat = z_axis.cross(clustered_splittings_pz0).unit
    decay_plane_hat = clustered_splittings_part_A_pz0.cross(clustered_splittings_part_B_pz0).unit

    #
    #  Clustering (calc variables to histogram)
    #
    clustered_splittings["zA"]        = clustered_splittings_pz0.dot(clustered_splittings_part_A_pz0) / (clustered_splittings_pz0.pt**2)
    clustered_splittings["mA"]        = clustered_splittings.part_A.mass
    clustered_splittings["rhoA"]      = clustered_splittings.part_A.mass / (clustered_splittings.pt * clustered_splittings.zA)
    clustered_splittings["abs_eta"]   = np.abs(clustered_splittings.eta)


    clustered_splittings["mB"]        = clustered_splittings.part_B.mass
    clustered_splittings["rhoB"]      = clustered_splittings.part_B.mass / (clustered_splittings.pt * (1 - clustered_splittings.zA))

    clustered_splittings["thetaA"]    = np.arccos(clustered_splittings_pz0.unit.dot(clustered_splittings_part_A_pz0.unit))
    clustered_splittings["tan_thetaA"]    = np.tan(np.arccos(clustered_splittings_pz0.unit.dot(clustered_splittings_part_A_pz0.unit)))
    clustered_splittings["decay_phi"] = np.arccos(decay_plane_hat.dot(comb_z_plane_hat))
    clustered_splittings["dr_AB"]     = clustered_splittings.part_A.delta_r(clustered_splittings.part_B)
    clustered_splittings["dpt_AB"]    = clustered_splittings.part_A.pt - (clustered_splittings.pt * clustered_splittings.zA)
    return


def rotateZ(particles, angle):
    sinT = np.sin(angle)
    cosT = np.cos(angle)
    x_rotated = cosT * particles.x - sinT * particles.y
    y_rotated = sinT * particles.x + cosT * particles.y

    return ak.zip(
        {
            "x": x_rotated,
            "y": y_rotated,
            "z": particles.z,
            "t": particles.t,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )


def rotateX(particles, angle):
    sinT = np.sin(angle)
    cosT = np.cos(angle)
    y_rotated = cosT * particles.y - sinT * particles.z
    z_rotated = sinT * particles.y + cosT * particles.z

    return ak.zip(
        {
            "x": particles.x,
            "y": y_rotated,
            "z": z_rotated,
            "t": particles.t,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )



def decluster_combined_jets(input_jet, debug=False):

    n_jets = np.sum(ak.num(input_jet))

    jet_flav_flat = ak.flatten(input_jet.jet_flavor)
    simple_comb_mask = (np.char.str_len(jet_flav_flat) == 2)

    # For some reason this dummy string has to be as long as the longest possible replacement
    # jet_flav_child_A = np.full(n_jets, "XXXXXXXXXXXXXXX")
    # jet_flav_child_B = np.full(n_jets, "XXXXXXXXXXXXXXX")
    dummy_str = "XXXXXXXXXXXXXXXXXXXXXXXXX"
    len_dummy_str = 25
    #dummy_str = "XXX"
    #len_dummy_str = 3
    jet_flav_child_A = np.full(n_jets, dummy_str)
    jet_flav_child_B = np.full(n_jets, dummy_str)

    #
    #  The simple combinations
    #  Fix this for the b and j ordERING!!!!
    _simple_flav_child_A = [str(s)[0] for s in jet_flav_flat[simple_comb_mask]]
    _simple_flav_child_B = [str(s)[1] for s in jet_flav_flat[simple_comb_mask]]
    jet_flav_child_A[simple_comb_mask] = _simple_flav_child_A
    jet_flav_child_B[simple_comb_mask] = _simple_flav_child_B

    #
    #  The nested combinations
    #   # A is always the more complex
    _children = [children_jet_flavors(s) for s in jet_flav_flat[~simple_comb_mask]]
    _nested_flav_child_A = [child[0] for child in _children]
    _nested_flav_child_B = [child[1] for child in _children]

    over_flow_child_A = any(len(s) > len_dummy_str for s in _nested_flav_child_A)
    if over_flow_child_A:
        print(f"\n ERROR: child A flavor overflow {_nested_flav_child_A} \n")

    over_flow_child_B = any(len(s) > len_dummy_str for s in _nested_flav_child_B)
    if over_flow_child_B:
        print(f"\n ERROR: child B flavor overflow {_nested_flav_child_B} \n")


    #print(f'child A {_nested_flav_child_A}')
    #print(f'child B {_nested_flav_child_B}')

    jet_flav_child_A[~simple_comb_mask] = _nested_flav_child_A
    jet_flav_child_B[~simple_comb_mask] = _nested_flav_child_B

    jet_flavor_A = ak.unflatten(jet_flav_child_A, ak.num(input_jet))
    jet_flavor_B = ak.unflatten(jet_flav_child_B, ak.num(input_jet))



    combined_pt = input_jet.pt
    tanThetaA = np.tan(input_jet.thetaA)
    tanThetaB = input_jet.zA / (1 - input_jet.zA) * tanThetaA

    #
    #  pA (in frame with pz=0 phi=0 decay_phi = 0)
    #
    pA_pz0_px = input_jet.zA * combined_pt
    pA_pz0_py = 0
    pA_pz0_pz = - input_jet.zA * combined_pt * tanThetaA
    #pA_mass   = input_jet.rhoA * input_jet.pt * input_jet.zA
    pA_mass   = input_jet.mA
    pA_pz0_E  = np.sqrt(pA_pz0_px**2 + pA_pz0_pz**2 + pA_mass**2)

    pA_pz0_phi0_decayPhi0 = ak.zip(
        {
            "x": pA_pz0_px,
            "y": pA_pz0_py,
            "z": pA_pz0_pz,
            "t": pA_pz0_E,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    pB_pz0_px = (1 - input_jet.zA) * combined_pt
    pB_pz0_py = 0
    pB_pz0_pz = (1 - input_jet.zA) * combined_pt * tanThetaB
    #pB_mass   = input_jet.rhoB * input_jet.pt * (1 - input_jet.zA)
    pB_mass   = input_jet.mB
    pB_pz0_E  = np.sqrt(pB_pz0_px**2 + pB_pz0_pz**2 + pB_mass**2)



    pB_pz0_phi0_decayPhi0 = ak.zip(
        {
            "x": pB_pz0_px,
            "y": pB_pz0_py,
            "z": pB_pz0_pz,
            "t": pB_pz0_E,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    #
    # Do Rotation of the decay plane
    #

    # Pseudo-random number to decide if we rotate by phi or phi + pi
    decay_phi = input_jet.decay_phi + np.pi * ((input_jet.pt % 1) > 0.5)

    pA_pz0_phi0 = rotateX(pA_pz0_phi0_decayPhi0, decay_phi)
    pB_pz0_phi0 = rotateX(pB_pz0_phi0_decayPhi0, decay_phi)

    #
    #  Boost back to jet pZ
    #
    boost_vec_z = ak.zip(
        {
            "x": 0,
            "y": 0,
            "z": input_jet.boostvec.z,
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )

    pA_phi0 = pA_pz0_phi0.boost(boost_vec_z)
    pB_phi0 = pB_pz0_phi0.boost(boost_vec_z)

    #
    #  Rotate to jet phi
    #
    pA = rotateZ(pA_phi0, input_jet.phi)
    pB = rotateZ(pB_phi0, input_jet.phi)

    pA = ak.zip(
        {
            "pt":  pA.pt,
            "eta": pA.eta,
            "phi": pA.phi,
            "mass":pA.mass,
            "jet_flavor":jet_flavor_A,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    pB = ak.zip(
        {
            "pt":  pB.pt,
            "eta": pB.eta,
            "phi": pB.phi,
            "mass":pB.mass,
            "jet_flavor":jet_flavor_B,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )


    return pA, pB
    #return ak.concatenate([pA, pB], axis=1)



def decluster_splitting_types(input_jets, splitting_types, input_pdfs, debug=False):

    #
    #  Create a mask for all the jets that need declustered
    #
    input_jets['split_mask'] = False
    for _s in splitting_types:

        _split_mask  = input_jets.jet_flavor == _s
        input_jets["split_mask"] = _split_mask | input_jets.split_mask

    #
    #  Save the jets that dont need to be declustered
    #
    unclustered_jets = input_jets[~input_jets.split_mask]



    #
    #  Mask the jets to be declustered
    #
    input_jets_to_decluster = input_jets[input_jets.split_mask]

    #
    #  Need to iterate b/c
    #   - Some unclusterings fail the jet pt and eta
    #   - Some lead to dR too close (Not checked yet!)
    #   - Some of the splittings are recursive (no implemented yet!)
    num_trys = 0


    while(ak.any(input_jets_to_decluster)):

        if debug: print(f" (decluster_splitting_types) num_trys {num_trys} ")

        splittings_info = []
        if debug: print(f"splittings_types is {splitting_types} num_trys {num_trys}")
        for _s in splitting_types:

            # Pre compute these to save time
            _s_mask = input_jets_to_decluster.jet_flavor == _s
            _num_samples   = np.sum(ak.num(input_jets_to_decluster[_s_mask]))
            _indicies = np.where(ak.flatten(_s_mask))
            _indicies_tuple = (_indicies[0].to_list())

            splittings_info.append((get_splitting_name(_s), _num_samples, _indicies_tuple))


        #
        #  Sample the PDFs,  add sampled varibales to the jets to be declustered
        #
        sample_PDFs_vs_pT(input_jets_to_decluster, input_pdfs, splittings_info)

        #
        #  do the declustering
        #
        declustered_jets_A, declustered_jets_B  = decluster_combined_jets(input_jets_to_decluster, debug=debug)

        #
        #  Check for declustered jets vailing kinematic requirements
        #
        # Update to only be bjets
        fail_pt_mask    = (declustered_jets_A.pt < 20) | (declustered_jets_B.pt < 20)
        fail_pt_b_mask  = ((declustered_jets_A.pt < 40) &  (declustered_jets_A.jet_flavor == "b")) | ((declustered_jets_B.pt < 40) &  (declustered_jets_B.jet_flavor == "b"))
        fail_eta_b_mask = ((declustered_jets_A.jet_flavor == "b") & (np.abs(declustered_jets_A.eta) > 2.5)) | ((declustered_jets_A.jet_flavor == "b") & (np.abs(declustered_jets_B.eta) > 2.5))
        fail_dr_mask  = declustered_jets_A.delta_r(declustered_jets_B) < 0.4
        clustering_fail = fail_pt_mask | fail_pt_b_mask | fail_eta_b_mask | fail_dr_mask

        #print(ak.any(fail_dr_mask))
        if num_trys > 4:
            print(f"Bailing with {np.sum(ak.num(input_jets_to_decluster))}\n")
            clustering_fail = ~(fail_pt_mask | ~fail_pt_mask)  #All False

        #
        #  Save unclustered jets that are OK
        #
        unclustered_jets = ak.concatenate([unclustered_jets, declustered_jets_A[~clustering_fail], declustered_jets_B[~clustering_fail]], axis=1)

        #
        #  Try again with the other jets
        #
        #print(f"Was {np.sum(ak.num(input_jets_decluster))}\n")
        input_jets_to_decluster = input_jets_to_decluster[clustering_fail]
        #print(f"Now {np.sum(ak.num(input_jets_decluster))}\n")
        num_trys += 1

    return unclustered_jets




def make_synthetic_event_core(input_jets, input_pdfs, debug=False):


    #
    # This needs to be recurseive !!!
    #

    #
    #  Get all the different types of splitted needed
    #
    splitting_types = get_list_of_combined_jet_types(input_jets)

    if debug: print(f" (make_synthetic_event_core) splitting_types {splitting_types}")

    while len(splitting_types):

        if debug: print(f"(make_synthetic_event_core) splitting_types was {splitting_types}")
        input_jets = decluster_splitting_types(input_jets, splitting_types, input_pdfs, debug=debug)

        splitting_types = get_list_of_combined_jet_types(input_jets)

        if debug: print(f"(make_synthetic_event_core) splitting_types is now {splitting_types}")

    if debug: print(f" (make_synthetic_event_core) splitting_types now {splitting_types}")
    #min_dr = get_min_dr(input_jets)

    return input_jets

def make_synthetic_event(input_jets, input_pdfs, debug=False):
    return make_synthetic_event_core(input_jets, input_pdfs, debug=debug)


# def make_synthetic_event(input_jets, input_pdfs):
#
#
#     #input_events_to_decluster = copy(input_jets) # Copy needed?
#
#     # Start with all True
#     events_to_decluster_mask = np.ones(len(input_jets), dtype=bool)
#
#     #output_events = [[] for _ in range(len(input_jets))]
#
#     #output_events = np.empty(len(input_jets), dtype=object)
#     #
#     #for i in range(len(input_jets)):
#     #    output_events[i] = []
#
#     n_events = len(input_jets)
#
#     empty_pt = ak.Array(np.zeros(0))
#     empty_eta = ak.Array(np.zeros(0))
#     empty_phi = ak.Array(np.zeros(0))
#     empty_mass = ak.Array(np.zeros(0))
#
#     # Create an empty PtEtaPhiMLorentzVectorArray
#     empty_vector = ak.zip({
#         "pt": empty_pt,
#         "eta": empty_eta,
#         "phi": empty_phi,
#         "mass": empty_mass
#     }, with_name="PtEtaPhiMLorentzVector")
#
#     # Create a NumPy array to hold n empty PtEtaPhiMLorentzVectorArrays
#     output_events = np.array([empty_vector] * n_events, dtype=object)
#
#
#     #
#     # Loop until all False
#     #
#     while(np.any(events_to_decluster_mask)):
#
#         to_decluster_indicies = np.where(events_to_decluster_mask)[0]
#
#         declustered_events = make_synthetic_event_core(input_jets[to_decluster_indicies], input_pdfs)
#
#
#         #
#         #  Check the min dr
#         #
#         print(declustered_events)
#         delta_r2_matrix = declustered_events.delta_r2(declustered_events[:, None])
#         delta_r2_matrix_flat = ak.flatten(delta_r2_matrix)
#         delta_r2_matrix_flat_flat = ak.flatten(delta_r2_matrix_flat).to_numpy()
#         delta_r2_matrix_flat_flat[delta_r2_matrix_flat_flat == 0] = np.inf
#         delta_r2_matrix_flat_masked = ak.unflatten(delta_r2_matrix_flat_flat, ak.num(delta_r2_matrix_flat))
#         delta_r2_matrix_masked = ak.unflatten(delta_r2_matrix_flat_masked, ak.num(delta_r2_matrix))
#
#         min_dr = ak.min(ak.min(delta_r2_matrix_masked,axis=1),axis=1)
#
#         pass_dr_mask = min_dr > 0.16 # 0.4**2
#
#         sucessful_deccluster_indicies = np.where(pass_dr_mask)
#         breakpoint()
#         update_indicies = to_decluster_indicies[sucessful_deccluster_indicies]
#
#
#         input_jets[update_indicies] = declustered_events[sucessful_deccluster_indicies]
#
#         to_decluster_indicies = np.where(~pass_dr_mask)[0]
#
#     return newly_declustered_events


def clean_ISR(clustered_jets, splittings, debug=False):

    all_jet_types =  get_list_of_splitting_types(clustered_jets)

    if debug:
        print(f" (clean_ISR) all_jet_types {all_jet_types}")


    ISR_splittings_types = get_list_of_ISR_splittings(all_jet_types)


    if debug:
        print(f" (clean_ISR) ISR_splittings_types {ISR_splittings_types}")


    #
    #  Will need recusion here
    #
    clustered_jets_clean = clustered_jets

    while(len(ISR_splittings_types)):

        for _isr_splitting in ISR_splittings_types:

            ISR_mask = clustered_jets_clean.jet_flavor == _isr_splitting
            ISR_jets = clustered_jets_clean[ISR_mask]

            ISR_splittings_mask = splittings.jet_flavor == _isr_splitting
            ISR_splittings = splittings[ISR_splittings_mask]

            pairs = ak.cartesian([ISR_jets, ISR_splittings], axis=1, nested=True)
            delta_r_values = pairs[:,"0"].delta_r(pairs[:,"1"])
            closest_indices = ak.argmin(delta_r_values, axis=2)
            match_splitting = ISR_splittings[closest_indices]

            if debug:
                print(f" ISR_jets: {ISR_jets.pt}  {ISR_jets.eta} {ISR_jets.phi} ")
                print(f" match_splitting: {match_splitting.pt}  {match_splitting.eta} {match_splitting.phi} ")
                print(f" ISR_splittings: {ISR_splittings.pt}  {ISR_splittings.eta} {ISR_splittings.phi} ")

            declustered_A = match_splitting.part_A
            declustered_B = match_splitting.part_B

            # To ADd
            #  detclustered_A_jets = decluster(detclustered_A) # recurseive deculstering
            #  detclustered_A_jets = decluster(detclustered_A) # recurseive deculstering

            declustered_ISR_jets = ak.concatenate([declustered_A, declustered_B], axis=1)

            clustered_jets_clean = clustered_jets_clean[~ISR_mask]
            clustered_jets_clean = ak.concatenate([clustered_jets_clean, declustered_ISR_jets], axis=1)

        #
        # Recompute ISR splitting_types
        #
        all_jet_types =  get_list_of_splitting_types(clustered_jets_clean)

        if debug:
            print(f" (clean_ISR) all_jet_types now {all_jet_types}")

        ISR_splittings_types = get_list_of_ISR_splittings(all_jet_types)

        if debug:
            print(f" (clean_ISR) ISR_splittings_types now {ISR_splittings_types}")



    return clustered_jets_clean
