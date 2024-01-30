import numpy as np

class cutFlow:

    def __init__(self, cuts):
        self._cutFlowThreeTag = {}
        self._cutFlowFourTag  = {}

        for c in cuts:
            self._cutFlowThreeTag[c] = (0, 0)    # weighted, raw
            self._cutFlowFourTag [c] = (0, 0)    # weighted, raw

    def fill(self, cut, event, allTag=False, wOverride=None):

        if allTag:

            if wOverride:
                sumw = wOverride
            else:
                sumw = np.sum(event.weight)

            sumn_3, sumn_4 = len(event), len(event)
            sumw_3, sumw_4 = sumw, sumw
        else:
            e3, e4 = event[event.threeTag], event[event.fourTag]

            sumw_3 = np.sum(e3.weight)
            sumn_3 = len(e3.weight)

            sumw_4 = np.sum(e4.weight)
            sumn_4 = len(e4.weight)

        self._cutFlowThreeTag[cut] = (sumw_3, sumn_3)     # weighted, raw
        self._cutFlowFourTag [cut] = (sumw_4, sumn_4)     # weighted, raw


    def addOutput(self, o, dataset):

        o["cutFlowFourTag"] = {}
        o["cutFlowFourTagUnitWeight"] = {}
        o["cutFlowFourTag"][dataset] = {}
        o["cutFlowFourTagUnitWeight"][dataset] = {}
        for k, v in  self._cutFlowFourTag.items():
            o["cutFlowFourTag"][dataset][k] = v[0]
            o["cutFlowFourTagUnitWeight"][dataset][k] = v[1]

        o["cutFlowThreeTag"] = {}
        o["cutFlowThreeTagUnitWeight"] = {}
        o["cutFlowThreeTag"][dataset] = {}
        o["cutFlowThreeTagUnitWeight"][dataset] = {}
        for k, v in  self._cutFlowThreeTag.items():
            o["cutFlowThreeTag"][dataset][k] = v[0]
            o["cutFlowThreeTagUnitWeight"][dataset][k] = v[1]

        return

