import numpy as np
import awkward as ak
from copy import copy
from coffea.nanoevents.methods import vector

# Define the kt clustering algorithm
def kt_clustering(event_jets, R):
    clustered_jets = []

    nevents = len(event_jets)

    for iEvent in range(nevents):
        particles = copy(event_jets[iEvent])
        print(particles)
        clustered_jets.append([])
        print(f"iEvent {iEvent}")        
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

                part_comb_array = ak.zip(
                    {
                        "pt": [part_comb.pt],
                        "eta": [part_comb.eta],
                        "phi": [part_comb.phi],
                        "mass": [part_comb.mass],
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.behavior,
                )

                particles = ak.concatenate([particles, part_comb_array])
                print(f"size partilces {len(particles)}")
                
                #particles = [part_comb] 
                #
                ## Combine particles using four-vector addition
                #px_i = part_i['pt'] * np.cos(part_i['phi'])
                #py_i = part_i['pt'] * np.sin(part_i['phi'])
                #pz_i = part_i['pt'] * np.sinh(part_i['eta'])
                #E_i = np.sqrt(px_i**2 + py_i**2 + pz_i**2 + part_i.mass**2)
                #
                #px_j = part_j['pt'] * np.cos(part_j['phi'])
                #py_j = part_j['pt'] * np.sin(part_j['phi'])
                #pz_j = part_j['pt'] * np.sinh(part_j['eta'])
                #E_j = np.sqrt(px_j**2 + py_j**2 + pz_j**2 + part_j.mass**2)
                #
                #px = px_i + px_j
                #py = py_i + py_j
                #pz = pz_i + pz_j
                #E = E_i + E_j
                #
                #pt = np.sqrt(px**2 + py**2)
                #eta = 0.5 * np.log((E + pz) / (E - pz)) if E != pz else 0
                #phi = np.arctan2(py, px)
                #
                #breakpoint()
                #
                #
                #particles[idx_i] = {
                #    'pt': pt,
                #    'eta': eta,
                #    'phi': phi,
                #    'mass': np.sqrt(E**2 - px**2 - py**2 - pz**2)
                #}
        
    return clustered_jets

