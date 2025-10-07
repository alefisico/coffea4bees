# Rochester corrections to muons
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/RochcorMuon
# https://twiki.cern.ch/twiki/pub/CMS/RochcorMuon/roccor.Run2.v5.tgz
# Based on https://programtalk.com/vs4/python/CoffeaTeam/coffea/tests/test_lookup_tools.py
### taken from https://gitlab.cern.ch/HPlus/NanoAnalysis/-/blob/master/NanoAODAnalysis/Framework/python/RochesterCorrections.py

import os
import numpy as np
import awkward as ak

from coffea import lookup_tools

class RochesterCorrections():
    def __init__(self,run,isData, filename):
        self.isData = isData
        self.notCorrecting = False
        if filename == "":
            self.notCorrecting = True
            print("Not using Rochester corrections")
        else:
            if os.path.exists(filename):
                rochester_data = lookup_tools.txt_converters.convert_rochester_file(filename, loaduncs=True)
                self.rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)
            else:
                print("Rochester correction file not found.")
                print("Not using Rochester corrections")
                self.notCorrecting = True

    def apply(self,events):
        if self.notCorrecting:
            return events.Muon

        if self.isData:
            SF = self.rochester.kScaleDT(events.Muon.charge, events.Muon.pt, events.Muon.eta, events.Muon.phi)
        else:
            hasgen = ~np.isnan(ak.fill_none(events.Muon.matched_gen.pt, np.nan))
            mc_kspread = self.rochester.kSpreadMC(
                events.Muon.charge[hasgen],
                events.Muon.pt[hasgen],
                events.Muon.eta[hasgen],
                events.Muon.phi[hasgen],
                events.Muon.matched_gen.pt[hasgen],
            )

            nMuons = len(ak.flatten(events.Muon.pt,axis=1))
            mc_rand = np.random.rand(len(ak.flatten(events.Muon.pt,axis=1)))
            mc_rand = ak.unflatten(mc_rand,ak.num(events.Muon.pt))
            mc_ksmear = self.rochester.kSmearMC(
                events.Muon.charge[~hasgen],
                events.Muon.pt[~hasgen],
                events.Muon.eta[~hasgen],
                events.Muon.phi[~hasgen],
                events.Muon.nTrackerLayers[~hasgen],
                mc_rand[~hasgen],
            )

            SF = np.array(ak.flatten(ak.ones_like(events.Muon.pt)))
            hasgen_flat = np.array(ak.flatten(hasgen))
            SF[hasgen_flat] = np.array(ak.flatten(mc_kspread))
            SF[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
            SF = ak.unflatten(SF,ak.num(events.Muon.pt))

        correctedMuons = events.Muon
        correctedMuons['pt'] = events.Muon.pt * SF
        return correctedMuons

