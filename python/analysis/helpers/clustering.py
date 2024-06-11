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

    nevents = len(event_jets)

    for iEvent in range(nevents):
        particles = copy(event_jets[iEvent])
        if debug: print(particles)

        if debug: print(f"iEvent {iEvent}")        

        # Maybe later allow more than 4 bs
        number_of_unclustered_bs = 4
        
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

            print(part_comb_array.jet_flavor)
            match part_comb_array.jet_flavor:
                case "g_bb" | "bstar":
                    number_of_unclustered_bs -= 1
                case _:
                    print(f"ERROR: counting {part_comb_array.jet_flavor}")

            
            particles = ak.concatenate([particles, part_comb_array])
            if debug: print(f"size partilces {len(particles)}")

        clustered_jets.append(particles)
    return clustered_jets



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
        
    return clustered_jets

