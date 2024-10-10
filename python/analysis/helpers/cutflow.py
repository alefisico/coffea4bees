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
                sumw = float(np.sum(event.weight))

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


    def addOutputSkim(self, o, dataset, lumi=1.0, xs=1.0, kFactor=1.0):

        if "lumi" not in o[dataset]:
            o[dataset]["lumi"] = [lumi]
        if "xs" not in o[dataset]:
            o[dataset]["xs"] = [xs]
        if "kFactor" not in o[dataset]:
            o[dataset]["kFactor"] = [kFactor]

        o[dataset]["cutFlowFourTag"]           = {}
        o[dataset]["cutFlowFourTagUnitWeight"] = {}
        for k, v in  self._cutFlowFourTag.items():
            o[dataset]["cutFlowFourTag"][k]           = v[0]
            o[dataset]["cutFlowFourTagUnitWeight"][k] = v[1]

        o[dataset]["cutFlowThreeTag"] = {}
        o[dataset]["cutFlowThreeTagUnitWeight"] = {}
        for k, v in  self._cutFlowThreeTag.items():
            o[dataset]["cutFlowThreeTag"][k] = v[0]
            o[dataset]["cutFlowThreeTagUnitWeight"][k] = v[1]

        return

    def addOutputLumisProcessed(self, o, dataset, runs, luminosityBlocks):

        o[dataset]["lumis_processed"]           = {}
        run_list = set(runs)
        for r in run_list:
            run_mask = (runs == r)
            lbs_per_run = list(set(luminosityBlocks[run_mask]))
            o[dataset]["lumis_processed"][r] = list(lbs_per_run)
