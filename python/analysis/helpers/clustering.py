import numpy as np
import awkward as ak
from copy import copy
from coffea.nanoevents.methods import vector

# anti kt
# dij = min(1 / (part_A['pt']**2), 1 / (part_B['pt']**2)) * delta_r(part_A['eta'], part_A['phi'], part_B['eta'], part_B['phi'])**2 / R**2
# diB = 1 / (part_A['pt'] ** 2)

# C/A
# dij = delta_r(part_A['eta'], part_A['phi'], part_B['eta'], part_B['phi'])**2 / R**2
# diB = R**2

# Speed this up by using avoiding loop...?

# If R = 0 exlusive clustereing
def get_distances(particles, R):
    distances = []
    for i, part_A in enumerate(particles):
        for j, part_B in enumerate(particles):
            if i < j:
                dij = min(part_A.pt**2, part_B.pt**2) * part_A.delta_r(part_B)**2
                if R:
                    dij = dij / R**2
                distances.append((dij, i, j))
        if R:
            diB = part_A.pt ** 2
            distances.append((diB, i, None))
    return distances


# If R = 0 exlusive clustereing
def get_min_indicies(particles, R):
    distances = []
    for i, part_A in enumerate(particles):
        for j, part_B in enumerate(particles):
            if i < j:
                dij = min(part_A.pt**2, part_B.pt**2) * part_A.delta_r(part_B)**2
                #dij = part_A.delta_r(part_B)**2            # KT
                if R:
                    dij = dij / R**2
                distances.append((dij, i, j))
        if R:
            diB = part_A.pt ** 2
            distances.append((diB, i, None))
    # Find the minimum distance
    _, idx_A, idx_B = min(distances)

    return idx_A, idx_B



def distance_matrix_kt(vectors):
    pt1 = ak.values_astype(vectors.pt, np.float64)
    eta1 = ak.values_astype(vectors.eta, np.float64)
    phi1 = ak.values_astype(vectors.phi, np.float64)

    pt2 = pt1[:, np.newaxis]
    eta2 = eta1[:, np.newaxis]
    phi2 = phi1[:, np.newaxis]

    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    deta = eta1 - eta2

    dr2 = deta**2 + dphi**2

    dij = np.minimum(pt1**2, pt2**2) * dr2

    dij = np.array(dij)
    # Mask to ignore the diagonal elements (where ΔR = 0)
    mask = np.eye(dij.shape[0], dtype=bool)

    # Set the diagonal elements to a large value to ignore them
    dij[mask] = np.inf
    return dij

def get_min_indicies_fast(particles, R):
    distances = []

    dij_matrix = distance_matrix_kt(particles)

    dij_min_per_jet    = np.min(dij_matrix,axis=1)
    dij_argmin_per_jet = np.argmin(dij_matrix,axis=1)

    idx_A = np.argmin(dij_min_per_jet)
    idx_B = dij_argmin_per_jet[idx_A]
    return idx_A, idx_B


def remove_indices(particles, indices_to_remove):
    mask = np.ones(len(particles), dtype=bool)
    mask[indices_to_remove] = False
    return particles[mask]

# Order by size of cluster then by flavour
def comb_jet_flavor(flavor_A, flavor_B):

    # Add Parens if the input is already clustered
    if len(flavor_A) > 1:
        flavor_A = f"({str(flavor_A)})"
    if len(flavor_B) > 1:
        flavor_B = f"({str(flavor_B)})"

    if len(flavor_A) < len(flavor_B):
        return flavor_A + flavor_B

    if len(flavor_B) < len(flavor_A):
        return flavor_A + flavor_B

    _name_list = [flavor_A, flavor_B]
    _name_list.sort()
    return "".join(_name_list)


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
        child_A = comb_flavor[0]
        child_B = sub_combs[0]
    elif len(sub_combs) == 2:
        child_A = sub_combs[0]
        child_B = sub_combs[1]
    else:
        print(f"ERROR comb_flavor is {comb_flavor} sub_combs is {sub_combs} len {len(sub_combs)}")

    return child_A, child_B

#def delta_bs(comb_flavor):
#    if len(comb_flavor) < 2:
#        print(f"ERROR len of combined flavor is too low {len(comb_flavor)}  {comb_flavor}")
#
#    sub_combs = extract_all_parentheses_substrings(comb_flavor)
#
#    if len(sub_combs) == 0:
#        child_A = comb_flavor[0]
#        child_B = comb_flavor[1]
#    elif len(sub_combs) == 1:
#        child_A = comb_flavor[0]
#        child_B = sub_combs[0]
#    elif len(sub_combs) == 2:
#        child_A = sub_combs[0]
#        child_B = sub_combs[1]
#    else:
#        print(f"ERROR comb_flavor is {comb_flavor} sub_combs is {sub_combs} len {len(sub_combs)}")
#
#    #print(f"children: {child_A} {child_A}")
#    #print(f"counts: {child_A.count('b')} {child_B.count('b')}")
#
#    if child_A.count("b") and child_B.count("b"):
#        return -1
#
#    return 0




def combine_particles(part_A, part_B, *, debug=False):
    part_comb = part_A + part_B

    new_part_A = part_A
    new_part_B = part_B
    if part_A.pt < part_B.pt:
        new_part_A = part_B
        new_part_B = part_A

    part_comb_jet_flavor = comb_jet_flavor(part_A.jet_flavor, part_B.jet_flavor)

    part_comb_array = ak.zip(
        {
            "pt": [part_comb.pt],
            "eta": [part_comb.eta],
            "phi": [part_comb.phi],
            "mass": [part_comb.mass],
            "jet_flavor": [part_comb_jet_flavor],
            "part_A": [new_part_A],
            "part_B": [new_part_B],
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    return part_comb_array



# Define the kt clustering algorithm
def cluster_bs_core(event_jets, distance_function, *, debug = False):
    clustered_jets = []
    splittings = []

    nevents = len(event_jets)

    for iEvent in range(nevents):
        particles = copy(event_jets[iEvent])
        if debug: print(particles)

        if debug: print(f"iEvent {iEvent}")
        if debug: print(f"==============================")
        if debug: print(f"nParticles {len(particles)}")
        # Maybe later allow more than 4 bs
        # number_of_unclustered_bs = 4

        splittings.append([])

        while True: # Break when try to combine more than 2 bs # number_of_unclustered_bs > 2:

            #
            # Calculate the distance measures
            #  R=0 turns off clustering to the beam
            #distances = get_distances(particles, R=0)
            #distances = distance_function(particles, R=0)
            idx_A, idx_B = distance_function(particles, R=0)

            if debug: print(f"clustering {idx_A} and {idx_B}")
            if debug: print(f"size partilces {len(particles)}")

            if debug: print(f"size partilces {len(particles)}")

            part_comb_array = combine_particles(particles[idx_A], particles[idx_B])

            #
            #  Stop if going to combine 3 bs
            #
            if part_comb_array.jet_flavor[0].count("b") > 2:
                if debug: print(f"breaking on {part_comb_array.jet_flavor[0]}")
                break


            if debug: print(part_comb_array.jet_flavor)
            if debug: print(f"{part_comb_array.jet_flavor}")
            #if debug: print(f"was {number_of_unclustered_bs}")

            # number_of_unclustered_bs += delta_bs(part_comb_array.jet_flavor[0])

            # if debug: print(f"now {number_of_unclustered_bs}")
            splittings[-1].append(part_comb_array[0])

            particles = remove_indices(particles, [idx_A, idx_B])

            particles = ak.concatenate([particles, part_comb_array])
            if debug: print(f"size partilces {len(particles)}")

        clustered_jets.append(particles)

    # Create the PtEtaPhiMLorentzVectorArray with ndim=2
    clustered_events = ak.zip(
        {
            "pt":         ak.Array([[v.pt for v in sublist] for sublist in clustered_jets]),
            "eta":        ak.Array([[v.eta for v in sublist] for sublist in clustered_jets]),
            "phi":        ak.Array([[v.phi for v in sublist] for sublist in clustered_jets]),
            "mass":       ak.Array([[v.mass for v in sublist] for sublist in clustered_jets]),
            "jet_flavor": ak.Array([[v.jet_flavor for v in sublist] for sublist in clustered_jets]),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )


    # Create the PtEtaPhiMLorentzVectorArray with ndim=2
    splittings_events = ak.zip(
        {
            "pt":  ak.Array([[v.pt  for v in sublist] for sublist in splittings]),
            "eta": ak.Array([[v.eta for v in sublist] for sublist in splittings]),
            "phi": ak.Array([[v.phi for v in sublist] for sublist in splittings]),
            "mass": ak.Array([[v.mass for v in sublist] for sublist in splittings]),
            "jet_flavor": ak.Array([[v.jet_flavor for v in sublist] for sublist in splittings]),
            "part_A": ak.zip(
                {
                    "pt":         ak.Array([[v.part_A.pt  for v in sublist] for sublist in splittings]),
                    "eta":        ak.Array([[v.part_A.eta for v in sublist] for sublist in splittings]),
                    "phi":        ak.Array([[v.part_A.phi for v in sublist] for sublist in splittings]),
                    "mass":       ak.Array([[v.part_A.mass for v in sublist] for sublist in splittings]),
                    "jet_flavor": ak.Array([[v.part_A.jet_flavor for v in sublist] for sublist in splittings]),
                    },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            ),
            "part_B": ak.zip(
                {
                    "pt":         ak.Array([[v.part_B.pt  for v in sublist] for sublist in splittings]),
                    "eta":        ak.Array([[v.part_B.eta for v in sublist] for sublist in splittings]),
                    "phi":        ak.Array([[v.part_B.phi for v in sublist] for sublist in splittings]),
                    "mass":       ak.Array([[v.part_B.mass for v in sublist] for sublist in splittings]),
                    "jet_flavor": ak.Array([[v.part_B.jet_flavor for v in sublist] for sublist in splittings]),
                    },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            ),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )

    return clustered_events, splittings_events



def cluster_bs(event_jets, *, debug = False):
    return cluster_bs_core(event_jets, get_min_indicies)


def cluster_bs_fast(event_jets, *, debug = False):
    return cluster_bs_core(event_jets, get_min_indicies_fast)


# Define the kt clustering algorithm
def kt_clustering(event_jets, R, *, debug = False):
    clustered_jets = []

    nevents = len(event_jets)

    for iEvent in range(nevents):
        particles = copy(event_jets[iEvent])
        if debug: print(particles)
        clustered_jets.append([])
        if debug: print(f"iEvent {iEvent}")

        while ak.any(particles):

            #
            # Calculate the distance measures
            #
            distances = get_distances(particles, R)

            # Find the minimum distance
            min_dist, idx_A, idx_B = min(distances)

            if idx_B is None:
                # If the minimum distance is diB, declare part_A as a jet
                if debug: print("adding clustered jet")
                clustered_jets[-1].append(copy(particles[idx_A]))

                particles = remove_indices(particles, [idx_A])

                if debug: print(f"size partilces {len(particles)}")

            else:
                if debug: print(f"clustering {idx_A} and {idx_B}")
                if debug: print(f"size partilces {len(particles)}")
                # If the minimum distance is dij, combine particles i and j
                part_A = copy(particles[idx_A])
                part_B = copy(particles[idx_B])

                particles = remove_indices(particles, [idx_A, idx_B])

                if debug: print(f"size partilces {len(particles)}")

                part_comb_array = combine_particles(part_A, part_B)

                particles = ak.concatenate([particles, part_comb_array])
                if debug: print(f"size partilces {len(particles)}")

    # Create the PtEtaPhiMLorentzVectorArray with ndim=2
    clustered_events = ak.zip(
        {
            "pt":         ak.Array([[v.pt for v in sublist] for sublist in clustered_jets]),
            "eta":        ak.Array([[v.eta for v in sublist] for sublist in clustered_jets]),
            "phi":        ak.Array([[v.phi for v in sublist] for sublist in clustered_jets]),
            "mass":       ak.Array([[v.mass for v in sublist] for sublist in clustered_jets]),
            "jet_flavor": ak.Array([[v.jet_flavor for v in sublist] for sublist in clustered_jets]),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )

    return clustered_events


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
    clustered_splittings["dr_AB"]      = clustered_splittings.part_A.delta_r(clustered_splittings.part_B)
    return


def decluster_combined_jets(input_jet):

    #
    #  Need a nested way to propogate the jet flavors
    #
    n_jets = np.sum(ak.num(input_jet))

    jet_flav_flat = ak.flatten(input_jet.jet_flavor)
    simple_comb_mask = (np.char.str_len(jet_flav_flat) == 2)

    jet_flav_child_A = np.full(n_jets, "XXX")
    jet_flav_child_B = np.full(n_jets, "XXX")

    #
    #  The simple combinations
    #
    _simple_flav_child_A = [s[0] for s in jet_flav_flat[simple_comb_mask]]
    _simple_flav_child_B = [s[1] for s in jet_flav_flat[simple_comb_mask]]
    jet_flav_child_A[simple_comb_mask] = _simple_flav_child_A
    jet_flav_child_B[simple_comb_mask] = _simple_flav_child_B


    #
    #  The nested combinations
    #
    _nested_flav_child_A = [s.lstrip("(").split("(")[0] for s in jet_flav_flat[~simple_comb_mask]]
    _nested_flav_child_B = [s.split("(")[1].rstrip(")") for s in jet_flav_flat[~simple_comb_mask]]

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



def sample_PDFs(input_jets_decluster, input_pdfs, splittings):

    n_jets   = np.sum(ak.num(input_jets_decluster))

    #
    #  Sample the PDFs for the jets we will uncluster
    #
    for _var_name in input_pdfs["varNames"]:

        if _var_name.find("_vs_") == -1:
            is_1d_pdf = True
            _sampled_data = np.ones(n_jets)
        else:
            is_1d_pdf = False
            _sampled_data_x = np.ones(n_jets)
            _sampled_data_y = np.ones(n_jets)

        # Sample the pdfs from the different splitting options
        for _splitting_name, _num_samples, _indicies_tuple in splittings:

            if is_1d_pdf:
                probs   = np.array(input_pdfs[_splitting_name][_var_name]["probs"], dtype=float)
                centers = np.array(input_pdfs[_splitting_name][_var_name]["bin_centers"], dtype=float)
                _sampled_data[_indicies_tuple] = np.random.choice(centers, size=_num_samples, p=probs)
            else:
                probabilities_flat   = np.array(input_pdfs[_splitting_name][_var_name]["probabilities_flat"], dtype=float)
                xcenters        = np.array(input_pdfs[_splitting_name][_var_name]["xcenters"],      dtype=float)
                ycenters        = np.array(input_pdfs[_splitting_name][_var_name]["ycenters"],      dtype=float)

                xcenters_flat = np.repeat(xcenters, len(ycenters))
                ycenters_flat = np.tile(ycenters, len(xcenters))

                sampled_indices = np.random.choice(len(probabilities_flat), size=_num_samples, p=probabilities_flat)

                _sampled_data_x[_indicies_tuple] = xcenters_flat[sampled_indices]
                _sampled_data_y[_indicies_tuple] = ycenters_flat[sampled_indices]

        #
        # Save the sampled data to the jets to be uclustered for use in decluster_combined_jets
        #
        if is_1d_pdf:
            input_jets_decluster[_var_name]         = ak.unflatten(_sampled_data,    ak.num(input_jets_decluster))
        else:
            input_jets_decluster["zA"]         = ak.unflatten(_sampled_data_x,    ak.num(input_jets_decluster))
            input_jets_decluster["thetaA"]     = ak.unflatten(_sampled_data_y,    ak.num(input_jets_decluster))



def sample_PDFs_vs_pT(input_jets_decluster, input_pdfs, splittings):

    n_jets   = np.sum(ak.num(input_jets_decluster))

    n_pt_bins = len(input_pdfs["pt_bins"]) - 1
    pt_masks = []
    for iPt in range(n_pt_bins):
        _min_pt = float(input_pdfs["pt_bins"][iPt])
        _max_pt = float(input_pdfs["pt_bins"][iPt+1])
        if _max_pt == "inf":
            _max_pt = np.inf

        _this_mask = (input_jets_decluster.pt > _min_pt) & (input_jets_decluster.pt < _max_pt)
        pt_masks.append( _this_mask )


    #
    #  Sample the PDFs for the jets we will uncluster
    #
    for _var_name in input_pdfs["varNames"]:

        if _var_name.find("_vs_") == -1:
            is_1d_pdf = True

            _sampled_data = np.ones(n_jets)
            _sampled_data_vs_pT = []
            for _iPt in range(n_pt_bins):
                _sampled_data_vs_pT.append(np.ones(n_jets))
        else:
            is_1d_pdf = False

            _sampled_data_x = np.ones(n_jets)
            _sampled_data_y = np.ones(n_jets)
            _sampled_data_x_vs_pT = []
            _sampled_data_y_vs_pT = []
            for _iPt in range(n_pt_bins):
                _sampled_data_x_vs_pT.append(np.ones(n_jets))
                _sampled_data_y_vs_pT.append(np.ones(n_jets))

        # Sample the pdfs from the different splitting options
        for _splitting_name, _num_samples, _indicies_tuple in splittings:

            for _iPt in range(n_pt_bins):

                if is_1d_pdf:
                    probs   = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["probs"], dtype=float)
                    centers = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["bin_centers"], dtype=float)
                    _sampled_data_vs_pT[_iPt][_indicies_tuple] = np.random.choice(centers, size=_num_samples, p=probs)
                else:
                    probabilities_flat = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["probabilities_flat"], dtype=float)
                    xcenters           = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["xcenters"],      dtype=float)
                    ycenters           = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["ycenters"],      dtype=float)

                    xcenters_flat = np.repeat(xcenters, len(ycenters))
                    ycenters_flat = np.tile(ycenters, len(xcenters))

                    sampled_indices = np.random.choice(len(probabilities_flat), size=_num_samples, p=probabilities_flat)

                    _sampled_data_x_vs_pT[_iPt][_indicies_tuple] = xcenters_flat[sampled_indices]
                    _sampled_data_y_vs_pT[_iPt][_indicies_tuple] = ycenters_flat[sampled_indices]


            #
            #  Now work out which pT bins to use
            #
            if is_1d_pdf:

                for iPt in range(n_pt_bins):
                    _pt_indicies = np.where(ak.flatten(pt_masks[iPt]))[0]
                    _sampled_data[_pt_indicies] = _sampled_data_vs_pT[iPt][_pt_indicies]

            else:

                for iPt in range(n_pt_bins):
                    _pt_indicies = np.where(ak.flatten(pt_masks[iPt]))[0]
                    _sampled_data_x[_pt_indicies] = _sampled_data_x_vs_pT[iPt][_pt_indicies]
                    _sampled_data_y[_pt_indicies] = _sampled_data_y_vs_pT[iPt][_pt_indicies]


        #
        # Save the sampled data to the jets to be uclustered for use in decluster_combined_jets
        #
        if is_1d_pdf:
            input_jets_decluster[_var_name]         = ak.unflatten(_sampled_data,    ak.num(input_jets_decluster))
        else:
            input_jets_decluster["zA"]         = ak.unflatten(_sampled_data_x,    ak.num(input_jets_decluster))
            input_jets_decluster["thetaA"]     = ak.unflatten(_sampled_data_y,    ak.num(input_jets_decluster))


def get_list_of_combined_jet_types(jets):
    all_jet_types =  get_list_of_splitting_types(jets)
    splitting_types = []
    for _s in all_jet_types:

        if len(_s) == 1:
            continue

        splitting_types.append(_s)

    return splitting_types


def decluster_splitting_types(input_jets, splitting_types, input_pdfs):

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

        splittings_info = []

        for _s in splitting_types:

            # Pre compute these to save time
            _s_mask = input_jets_to_decluster.jet_flavor == _s
            _num_samples   = np.sum(ak.num(input_jets_to_decluster[_s_mask]))
            _indicies = np.where(ak.flatten(_s_mask))
            _indicies_tuple = (_indicies[0].to_list())

            splittings_info.append((_s, _num_samples, _indicies_tuple))


        #
        #  Sample the PDFs,  add sampled varibales to the jets to be declustered
        #
        sample_PDFs_vs_pT(input_jets_to_decluster, input_pdfs, splittings_info)

        #
        #  do the declustering
        #
        declustered_jets_A, declustered_jets_B  = decluster_combined_jets(input_jets_to_decluster)

        #
        #  Check for declustered jets vailing kinematic requirements
        #
        fail_pt_mask  = (declustered_jets_A.pt < 40) | (declustered_jets_B.pt < 40)
        fail_eta_mask = (np.abs(declustered_jets_A.eta) > 2.5) | (np.abs(declustered_jets_B.eta) > 2.5)
        clustering_fail = fail_pt_mask | fail_eta_mask

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




def make_synthetic_event(input_jets, input_pdfs):


    #
    # This needs to be recurseive !!!
    #

    #
    #  Get all the different types of splitted needed
    #
    splitting_types = get_list_of_combined_jet_types(input_jets)

    while len(splitting_types):

        input_jets = decluster_splitting_types(input_jets, splitting_types, input_pdfs)

        splitting_types = get_list_of_combined_jet_types(input_jets)

    return input_jets



def get_list_of_splitting_types(splittings):
    unique_splittings = set(ak.flatten(splittings.jet_flavor).to_list())
    return list(unique_splittings)


def clean_ISR(clustered_jets, splittings):

    all_jet_types =  get_list_of_splitting_types(clustered_jets)

    ISR_splittings = []
    for _s in all_jet_types:

        if len(_s) == 1:
            continue

        child_A, child_B = children_jet_flavors(_s)

        if child_A.count("b") == 0 and child_B.count("b") > 1:
            ISR_splittings.append(_s)

    #
    #  Will need recusion here
    #
    clustered_jets_noISR = clustered_jets

    for _isr_splitting in ISR_splittings:

        ISR_mask = clustered_jets.jet_flavor == _isr_splitting
        ISR_jets = clustered_jets[ISR_mask]

        ISR_splittings_mask = splittings.jet_flavor == 'j(bb)'
        ISR_splittings = splittings[ISR_splittings_mask]

        match_splitting = (ISR_splittings.delta_r(ISR_jets) == 0)
        declustered_A = ISR_splittings[match_splitting].part_A

        # To ADd
        #  detclustered_A_jets = decluster(detclustered_A) # recurseive deculstering

        declustered_B = ISR_splittings[match_splitting].part_B
        declustered_ISR_jets = ak.concatenate([declustered_A, declustered_B], axis=1)

        clustered_jets_noISR = clustered_jets[~ISR_mask]
        clustered_jets_noISR = ak.concatenate([clustered_jets_noISR, declustered_ISR_jets], axis=1)

    return clustered_jets_noISR
