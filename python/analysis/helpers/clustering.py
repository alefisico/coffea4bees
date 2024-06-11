import numpy as np
import awkward as ak
from copy import copy
from coffea.nanoevents.methods import vector





# Define the kt clustering algorithm
def cluster_bjets(event_jets, R):
    clustered_jets = []

    nevents = len(event_jets)

    for iEvent in range(nevents):
        particles = copy(event_jets[iEvent])
        clustered_jets.append([])
        print(f"iEvent {iEvent}")        

        # Maybe allow 5 bs later
        number_of_unclustered_bs = 4

        while number_of_unclustered_bs > 2:
            
            #
            # Calculate the distance measures
            #
            distances = []
            for i, part_i in enumerate(particles):
                diB = part_i['pt'] ** 2
                for j, part_j in enumerate(particles):
                    if i < j:
                        dij = min(part_i.pt**2, part_j.pt**2) * part_i.delta_r(part_j)**2 / R**2
                        distances.append((dij, i, j))
                distances.append((diB, i, None))
            
            # Find the minimum distance
            min_dist, idx_i, idx_j = min(distances)

            if idx_j is None:
                # If the minimum distance is diB, declare part_i as a jet
                print("adding clustered jet")
                clustered_jets[-1].append(copy(particles[idx_i]))
                index_to_remove = idx_i
                mask = np.arange(len(particles)) != index_to_remove
                particles = particles[mask]
                print(f"size partilces {len(particles)}")
                
            else:
                print(f"clustering {idx_i} and {idx_j}")
                print(f"size partilces {len(particles)}")
                # If the minimum distance is dij, combine particles i and j
                part_i = copy(particles[idx_i])
                part_j = copy(particles[idx_j])

                indices_to_remove = [idx_i, idx_j]  # Remove the entries at index 1 and 3
                mask = np.ones(len(particles), dtype=bool)
                mask[indices_to_remove] = False

                particles = particles[mask]
                print(f"size partilces {len(particles)}")
                
                part_comb = part_i + part_j

                jet_flavor_pair = (part_i.jet_flavor, part_j.jet_flavor)
                
                match jet_flavor_pair:
                    case ("b","b"):
                        part_comb_jet_flavor = "g_bb"
                    case ("b","g_bb") | ("g_bb", "b") :
                        part_comb_jet_flavor = "bstar"                        
                    case _:
                        print(f"ERROR: combining {jet_flavor_pair}")
                        part_comb_jet_flavor = "ERROR"

                
                part_comb_array = ak.zip(
                    {
                        "pt": [part_comb.pt],
                        "eta": [part_comb.eta],
                        "phi": [part_comb.phi],
                        "mass": [part_comb.mass],
                        "jet_flavor": [part_comb_jet_flavor],                        
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.behavior,
                )

                particles = ak.concatenate([particles, part_comb_array])
                print(f"size partilces {len(particles)}")
        
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
            distances = []
            for i, part_i in enumerate(particles):
                diB = part_i['pt'] ** 2
                for j, part_j in enumerate(particles):
                    if i < j:
                        dij = min(part_i.pt**2, part_j.pt**2) * part_i.delta_r(part_j)**2 / R**2
                        distances.append((dij, i, j))
                distances.append((diB, i, None))
            
            # Find the minimum distance
            min_dist, idx_i, idx_j = min(distances)

            if idx_j is None:
                # If the minimum distance is diB, declare part_i as a jet
                if debug: print("adding clustered jet")
                clustered_jets[-1].append(copy(particles[idx_i]))
                index_to_remove = idx_i
                mask = np.arange(len(particles)) != index_to_remove
                particles = particles[mask]
                if debug: print(f"size partilces {len(particles)}")
                
            else:
                if debug: print(f"clustering {idx_i} and {idx_j}")
                if debug: print(f"size partilces {len(particles)}")
                # If the minimum distance is dij, combine particles i and j
                part_i = copy(particles[idx_i])
                part_j = copy(particles[idx_j])

                indices_to_remove = [idx_i, idx_j]  # Remove the entries at index 1 and 3
                mask = np.ones(len(particles), dtype=bool)
                mask[indices_to_remove] = False

                particles = particles[mask]
                if debug: print(f"size partilces {len(particles)}")
                
                part_comb = part_i + part_j

                jet_flavor_pair = (part_i.jet_flavor, part_j.jet_flavor)
                
                match jet_flavor_pair:
                    case ("b","b"):
                        part_comb_jet_flavor = "g_bb"
                    case ("b","g_bb") | ("g_bb", "b") :
                        part_comb_jet_flavor = "bstar"                        
                    case _:
                        if debug: print(f"ERROR: combining {jet_flavor_pair}")
                        part_comb_jet_flavor = "ERROR"

                
                part_comb_array = ak.zip(
                    {
                        "pt": [part_comb.pt],
                        "eta": [part_comb.eta],
                        "phi": [part_comb.phi],
                        "mass": [part_comb.mass],
                        "jet_flavor": [part_comb_jet_flavor],                        
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.behavior,
                )

                particles = ak.concatenate([particles, part_comb_array])
                if debug: print(f"size partilces {len(particles)}")
        
    return clustered_jets

