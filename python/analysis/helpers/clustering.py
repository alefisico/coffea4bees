import numpy as np
import awkward as ak
from copy import copy
from coffea.nanoevents.methods import vector


# If R = 0 exlusive clustereing
def get_distances(particles, R):
    distances = []
    for i, part_i in enumerate(particles):
        for j, part_j in enumerate(particles):
            if i < j:
                dij = min(part_i.pt**2, part_j.pt**2) * part_i.delta_r(part_j)**2
                if R:
                    dij = dij / R**2
                distances.append((dij, i, j))
        if R:
            diB = part_i.pt ** 2
            distances.append((diB, i, None))
    return distances


def remove_indices(particles, indices_to_remove):
    mask = np.ones(len(particles), dtype=bool)
    mask[indices_to_remove] = False
    return particles[mask]

def combine_particles(part_i, part_j, debug=False):
    part_comb = part_i + part_j

    jet_flavor_pair = (part_i.jet_flavor, part_j.jet_flavor)

    match jet_flavor_pair:
        case ("b","b"):
            part_comb_jet_flavor = "g_bb"
        case ("b","g_bb") | ("g_bb", "b") :
            part_comb_jet_flavor = "bstar"
        case _:
            if debug: print(f"ERROR: combining {jet_flavor_pair}")
            part_comb_jet_flavor = f"ERROR {part_i.jet_flavor} and {part_j.jet_flavor}"

    part_comb_array = ak.zip(
        {
            "pt": [part_comb.pt],
            "eta": [part_comb.eta],
            "phi": [part_comb.phi],
            "mass": [part_comb.mass],
            "jet_flavor": [part_comb_jet_flavor],
            "part_i": [part_i],
            "part_j": [part_j],
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    return part_comb_array



# Define the kt clustering algorithm
def cluster_bs(event_jets, debug = False):
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
            distances = get_distances(particles, R=0)

            # Find the minimum distance
            min_dist, idx_i, idx_j = min(distances)

            if debug: print(f"clustering {idx_i} and {idx_j}")
            if debug: print(f"size partilces {len(particles)}")
            # If the minimum distance is dij, combine particles i and j

            part_i = copy(particles[idx_i])
            part_j = copy(particles[idx_j])
            particles = remove_indices(particles, [idx_i, idx_j])

            if debug: print(f"size partilces {len(particles)}")

            part_comb_array = combine_particles(part_i, part_j)

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
            "part_i": ak.zip(
                {
                    "pt":         ak.Array([[v.part_i.pt  for v in sublist] for sublist in splittings]),
                    "eta":        ak.Array([[v.part_i.eta for v in sublist] for sublist in splittings]),
                    "phi":        ak.Array([[v.part_i.phi for v in sublist] for sublist in splittings]),
                    "mass":       ak.Array([[v.part_i.mass for v in sublist] for sublist in splittings]),
                    "jet_flavor": ak.Array([[v.part_i.jet_flavor for v in sublist] for sublist in splittings]),
                    },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            ),
            "part_j": ak.zip(
                {
                    "pt":         ak.Array([[v.part_j.pt  for v in sublist] for sublist in splittings]),
                    "eta":        ak.Array([[v.part_j.eta for v in sublist] for sublist in splittings]),
                    "phi":        ak.Array([[v.part_j.phi for v in sublist] for sublist in splittings]),
                    "mass":       ak.Array([[v.part_j.mass for v in sublist] for sublist in splittings]),
                    "jet_flavor": ak.Array([[v.part_j.jet_flavor for v in sublist] for sublist in splittings]),
                    },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            ),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )

    return clustered_events, splittings_events



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
            min_dist, idx_i, idx_j = min(distances)

            if idx_j is None:
                # If the minimum distance is diB, declare part_i as a jet
                if debug: print("adding clustered jet")
                clustered_jets[-1].append(copy(particles[idx_i]))

                particles = remove_indices(particles, [idx_i])

                if debug: print(f"size partilces {len(particles)}")

            else:
                if debug: print(f"clustering {idx_i} and {idx_j}")
                if debug: print(f"size partilces {len(particles)}")
                # If the minimum distance is dij, combine particles i and j
                part_i = copy(particles[idx_i])
                part_j = copy(particles[idx_j])

                particles = remove_indices(particles, [idx_i, idx_j])

                if debug: print(f"size partilces {len(particles)}")

                part_comb_array = combine_particles(part_i, part_j)

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
    clustered_splittings_part_i_pz0 = clustered_splittings.part_i.boost(-boost_vec_z)
    clustered_splittings_part_j_pz0 = clustered_splittings.part_j.boost(-boost_vec_z)

    comb_z_plane_hat = z_axis.cross(clustered_splittings_pz0).unit
    decay_plane_hat = clustered_splittings_part_i_pz0.cross(clustered_splittings_part_j_pz0).unit

    #
    #  Clustering (calc variables to histogram)
    #
    clustered_splittings["mA"]        = clustered_splittings.part_i.mass
    clustered_splittings["mB"]        = clustered_splittings.part_j.mass
    clustered_splittings["zA"]        = clustered_splittings_pz0.dot(clustered_splittings_part_i_pz0)/(clustered_splittings_pz0.pt**2)
    clustered_splittings["thetaA"]    = np.arccos(clustered_splittings_pz0.unit.dot(clustered_splittings_part_i_pz0.unit))
    clustered_splittings["decay_phi"] = np.arccos(decay_plane_hat.dot(comb_z_plane_hat))

    return

def decluster_combined_jets(input_jet, zA, thetaA, mA, mB, decay_phi):

    combined_pt = input_jet.pt
    tanThetaA = np.tan(thetaA)
    tanThetaB = zA / (1 - zA) * tanThetaA

    #
    #  pA (in frame with pz=0 phi=0)
    #
    pA_pz0_px = zA * combined_pt
    pA_pz0_py = 0
    pA_pz0_pz = - zA * combined_pt * tanThetaA
    pA_pz0_E  = np.sqrt(pA_pz0_px**2 + pA_pz0_pz**2 + mA**2)

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

    pB_pz0_px = (1 - zA) * combined_pt
    pB_pz0_py = 0
    pB_pz0_pz = (1 - zA) * combined_pt * tanThetaB
    pB_pz0_E  = np.sqrt(pB_pz0_px**2 + pB_pz0_pz**2 + mB**2)

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
    pA_pz0 = rotateX(pA_pz0_phi0, decay_phi)
    pB_pz0 = rotateX(pB_pz0_phi0, decay_phi)

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

    return pA, pB
