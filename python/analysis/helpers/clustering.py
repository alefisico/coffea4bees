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

def combine_particles(part_A, part_B, debug=False):
    part_comb = part_A + part_B

    jet_flavor_pair = (part_A.jet_flavor, part_B.jet_flavor)

    match jet_flavor_pair:
        case ("b","b"):
            part_comb_jet_flavor = "g_bb"
        case ("b","g_bb") | ("g_bb", "b") :
            part_comb_jet_flavor = "bstar"
        case _:
            if debug: print(f"ERROR: combining {jet_flavor_pair}")
            part_comb_jet_flavor = f"ERROR {part_A.jet_flavor} and {part_B.jet_flavor}"

    part_comb_array = ak.zip(
        {
            "pt": [part_comb.pt],
            "eta": [part_comb.eta],
            "phi": [part_comb.phi],
            "mass": [part_comb.mass],
            "jet_flavor": [part_comb_jet_flavor],
            "part_A": [part_A],
            "part_B": [part_B],
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    return part_comb_array



# Define the kt clustering algorithm
def cluster_bs_core(event_jets, distance_function, debug = False):
    clustered_jets = []
    splittings = []

    nevents = len(event_jets)

    for iEvent in range(nevents):
        particles = copy(event_jets[iEvent])
        if debug: print(particles)

        if debug: print(f"iEvent {iEvent}")

        # Maybe later allow more than 4 bs
        number_of_unclustered_bs = 4

        splittings.append([])

        while number_of_unclustered_bs > 2:

            #
            # Calculate the distance measures
            #  R=0 turns off clustering to the beam
            #distances = get_distances(particles, R=0)
            #distances = distance_function(particles, R=0)
            idx_A, idx_B = distance_function(particles, R=0)

            if debug: print(f"clustering {idx_A} and {idx_B}")
            if debug: print(f"size partilces {len(particles)}")
            # If the minimum distance is dij, combine particles i and j

            part_A = copy(particles[idx_A])
            part_B = copy(particles[idx_B])
            particles = remove_indices(particles, [idx_A, idx_B])

            if debug: print(f"size partilces {len(particles)}")

            part_comb_array = combine_particles(part_A, part_B)

            if debug: print(part_comb_array.jet_flavor)
            match part_comb_array.jet_flavor:
                case "g_bb" | "bstar":
                    number_of_unclustered_bs -= 1
                case _:
                    print(f"ERROR: counting {part_comb_array.jet_flavor}")

            splittings[-1].append(part_comb_array[0])

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



def cluster_bs(event_jets, debug = False):
    return cluster_bs_core(event_jets, get_min_indicies)


def cluster_bs_fast(event_jets, debug = False):
    return cluster_bs_core(event_jets, get_min_indicies_fast)


# Define the kt clustering algorithm
def kt_clustering(event_jets, R, debug = False):
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
    clustered_splittings["mA"]        = clustered_splittings.part_A.mass
    clustered_splittings["mB"]        = clustered_splittings.part_B.mass
    clustered_splittings["zA"]        = clustered_splittings_pz0.dot(clustered_splittings_part_A_pz0)/(clustered_splittings_pz0.pt**2)
    clustered_splittings["thetaA"]    = np.arccos(clustered_splittings_pz0.unit.dot(clustered_splittings_part_A_pz0.unit))
    clustered_splittings["decay_phi"] = np.arccos(decay_plane_hat.dot(comb_z_plane_hat))
    clustered_splittings["dr_AB"]      = clustered_splittings.part_A.delta_r(clustered_splittings.part_B)
    return

def decluster_combined_jets(input_jet):

    combined_pt = input_jet.pt
    tanThetaA = np.tan(input_jet.thetaA)
    tanThetaB = input_jet.zA / (1 - input_jet.zA) * tanThetaA

    #
    #  pA (in frame with pz=0 phi=0)
    #
    pA_pz0_px = input_jet.zA * combined_pt
    pA_pz0_py = 0
    pA_pz0_pz = - input_jet.zA * combined_pt * tanThetaA
    pA_pz0_E  = np.sqrt(pA_pz0_px**2 + pA_pz0_pz**2 + input_jet.mA**2)

    pA_pz0_phi0 = ak.zip(
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
    pB_pz0_E  = np.sqrt(pB_pz0_px**2 + pB_pz0_pz**2 + input_jet.mB**2)

    pB_pz0_phi0 = ak.zip(
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
    # Do Rotation
    #
    pA_pz0 = rotateX(pA_pz0_phi0, input_jet.decay_phi)
    pB_pz0 = rotateX(pB_pz0_phi0, input_jet.decay_phi)

    #
    #  De-Clustering
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

    #
    #  Boost back
    #
    pA = pA_pz0.boost(boost_vec_z)
    pB = pB_pz0.boost(boost_vec_z)

    pA_x_flat = np.array(ak.flatten(pA.x))
    pA_y_flat = np.array(ak.flatten(pA.y))
    pA_z_flat = np.array(ak.flatten(pA.z))
    pA_t_flat = np.array(ak.flatten(pA.t))

    input_jet_x = ak.flatten(input_jet.x)
    input_jet_y = ak.flatten(input_jet.y)
    input_jet_z = ak.flatten(input_jet.z)
    input_jet_t = ak.flatten(input_jet.t)

    original_jet_indicies = np.where(ak.flatten(~input_jet.decluster_mask))
    original_jet_indicies_tuple = (original_jet_indicies[0].to_list())

    pA_x_flat[original_jet_indicies_tuple] = input_jet_x[original_jet_indicies_tuple]
    pA_y_flat[original_jet_indicies_tuple] = input_jet_y[original_jet_indicies_tuple]
    pA_z_flat[original_jet_indicies_tuple] = input_jet_z[original_jet_indicies_tuple]
    pA_t_flat[original_jet_indicies_tuple] = input_jet_t[original_jet_indicies_tuple]

    pA_masked = ak.zip(
        {
            "x": ak.unflatten(pA_x_flat, ak.num(input_jet)),
            "y": ak.unflatten(pA_y_flat, ak.num(input_jet)),
            "z": ak.unflatten(pA_z_flat, ak.num(input_jet)),
            "t": ak.unflatten(pA_t_flat, ak.num(input_jet)),
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    pB_masked = pB[input_jet.decluster_mask]

    #return ak.concatenate([pA_masked, pB_masked], axis=1)
    return pA_masked, pB_masked
