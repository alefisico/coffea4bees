from hist import Hist
import numpy as np

class cutFlow:

    def __init__(self, cuts, do_truth_hists=False):
        self._cutFlowThreeTag = {}
        self._cutFlowFourTag  = {}

        for c in cuts:
            self._cutFlowThreeTag[c] = (0, 0)    # weighted, raw
            self._cutFlowFourTag [c] = (0, 0)    # weighted, raw

        if do_truth_hists:
            self._hists  = {}
        else:
            self._hists  = None

    def fill(self, cut, event, allTag=False, wOverride=None):

        if allTag:
            if self._hists is not None:
                m4b = event.truth_v4b.mass

            if wOverride:
                sumw = wOverride
                m4b_weights = wOverride
            else:
                sumw = float(np.sum(event.weight))
                m4b_weights = event.weight

            sumn_3, sumn_4 = len(event), len(event)
            sumw_3, sumw_4 = sumw, sumw


        else:
            e3, e4 = event[event.threeTag], event[event.fourTag]

            if self._hists is not None:
                m4b = e4.truth_v4b.mass

            m4b_weights = e4.weight

            sumw_3 = np.sum(e3.weight)
            sumn_3 = len(e3.weight)

            sumw_4 = np.sum(e4.weight)
            sumn_4 = len(e4.weight)


        self._cutFlowThreeTag[cut] = (sumw_3, sumn_3)     # weighted, raw
        self._cutFlowFourTag [cut] = (sumw_4, sumn_4)     # weighted, raw

        if self._hists is not None:
            self._hists[cut] = Hist.new.Reg(120, 0, 1200, name="mass", label="Values").Double()
            self._hists[cut].fill(m4b, weight=m4b_weights)


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

        if self._hists is not None:
            o["cutflow_hists"] = {}
            o["cutflow_hists"][dataset] = {}
            for k, v in  self._hists.items():
                o["cutflow_hists"][dataset][k] = v

        return


    def addOutputSkim(self, o, dataset):

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
