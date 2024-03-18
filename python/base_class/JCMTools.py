from scipy.special import comb
import numpy as np
from coffea.util import load
from base_class.plots.plots import get_cut_dict
import base_class.plots.iPlot_config as cfg
import hist
from copy import copy
from scipy.optimize import curve_fit
import scipy.stats


class modelParameter:
    def __init__(self, name="", index=0, lowerLimit=0, upperLimit=1, default=0.5, fix=None):
        self.name = name
        self.value = None
        self.error = None
        self.percentError = None
        self.index = index
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.default = default
        self.fix = fix

    def dump(self):
        self.percentError = self.error / self.value * 100 if self.value else 0
        print((self.name + " %1.4f +/- %0.5f (%1.1f%%)") % (self.value, self.error, self.percentError))


class jetCombinatoricModel:

    def __init__(self, *, tt4b_nTagJets, tt4b_nTagJets_errors, qcd3b, qcd3b_errors, tt4b):

        self.pseudoTagProb         = modelParameter("pseudoTagProb",         index=0, lowerLimit=0,   upperLimit=1,       default=0.05)
        self.pairEnhancement       = modelParameter("pairEnhancement",       index=1, lowerLimit=0,   upperLimit=3,       default=1.0,)
        self.pairEnhancementDecay  = modelParameter("pairEnhancementDecay",  index=2, lowerLimit=0.1, upperLimit=100,     default=0.7)
        self.threeTightTagFraction = modelParameter("threeTightTagFraction", index=3, lowerLimit=0,   upperLimit=1000000, default=1000)

        self.parameters         = [self.pseudoTagProb, self.pairEnhancement, self.pairEnhancementDecay, self.threeTightTagFraction]


        #
        #  Data
        #
        self.tt4b_nTagJets        = tt4b_nTagJets
        self.tt4b_nTagJets_errors = tt4b_nTagJets_errors
        self.qcd3b                = qcd3b
        self.qcd3b_errors         = qcd3b_errors
        self.tt4b                 = tt4b

        self.default_parameters = []
        self.fit_parameters = []
        self.parameters_lower_bounds = []
        self.parameters_upper_bounds = []
        for p in self.parameters:
            self.fit_parameters.append(p)
            self.parameters_lower_bounds.append(p.lowerLimit)
            self.parameters_upper_bounds.append(p.upperLimit)
            self.default_parameters.append(p.default)

        self.nParameters = len(self.parameters)

    def dump(self):
        for parameter in self.parameters:
            parameter.dump()

    def fixParameter(self, name, value):
        for ip, p in enumerate(self.parameters):
            if name is p.name:
                print(f"Fixing {name} to {value}")
                p.fix = value
                self.fit_parameters.pop(ip)
                self.default_parameters.pop(ip)
                self.parameters_lower_bounds.pop(ip)
                self.parameters_upper_bounds.pop(ip)


        #
        #  Fix the normalizaiton to the threeTightTagFraction
        #
        self.bkgd_func_njet_constrained = lambda x, f, e, d, debug=False: self.bkgd_func_njet(x, f, e, d, value, debug)

    def _nTagPred_values(self, par, n):
        output = np.zeros(len(n))
        output = copy(self.tt4b_nTagJets)

        for ibin, this_nTag in enumerate(n):
            for nj in range(this_nTag, 14):
                nPseudoTagProb = getPseudoTagProbs(nj, par[0], par[1], par[2], par[3])
                output[ibin + 4] += nPseudoTagProb[this_nTag - 3] * self.qcd3b[nj]

        return np.array(output, float)

    def nTagPred_values(self, n):
        return self._nTagPred_values(self.fit_parameters + [self.threeTightTagFraction.fix], n)

    def nJetPred_values(self, n):
        return self.bkgd_func_njet_constrained(n, *self.fit_parameters)

    def getCombinatoricWeightList(self):
        output_weights = []
        for nj in range(4, 16):
            output_weights.append(getCombinatoricWeight(nj, *(self.fit_parameters + [self.threeTightTagFraction.fix])))
        return output_weights
    
    def _nTagPred_errors(self, par, n):
        output = np.zeros(len(n))
        output = self.tt4b_nTagJets_errors**2

        for ibin, this_nTag in enumerate(n):
            for nj in range(this_nTag, 14):
                nPseudoTagProb = getPseudoTagProbs(nj, par[0], par[1], par[2], par[3])
                output[ibin + 4] += (nPseudoTagProb[this_nTag - 3] * self.qcd3b_errors[nj])**2

        output = output**0.5
        return np.array(output, float)

    def nTagPred_errors(self, n):
        return self._nTagPred_errors(self.fit_parameters + [self.threeTightTagFraction.fix], n)

    def bkgd_func_njet(self, x, f, e, d, norm, debug=False):
        nj = x.astype(int)
        output = np.zeros(len(x))

        nTags = nj + 4
        nTags_pred_result = self._nTagPred_values([f, e, d, norm], nTags)
        output[0:4] = nTags_pred_result[4:8]
        if debug:
            print(f"output is {output}")

        for ibin, this_nj in enumerate(nj):
            if this_nj < 4:
                continue

            w = getCombinatoricWeight(this_nj, f, e, d, norm)
            output[this_nj] += w * self.qcd3b[this_nj]
            output[this_nj] += self.tt4b[this_nj]

        return output

    def fit(self, bin_centers, bin_values, bin_errors):

        #
        # Do the fit
        #
        popt, errs = curve_fit(self.bkgd_func_njet_constrained, bin_centers, bin_values, self.default_parameters, sigma=bin_errors,
                               bounds=(self.parameters_lower_bounds, self.parameters_upper_bounds)
                               )

        self.fit_errs = errs
        sigma_p1 = [np.absolute(errs[i][i])**0.5 for i in range(len(popt))]

        for parameter in self.parameters:
            if parameter.fix:
                parameter.value = parameter.fix
                parameter.error = 0
                continue

            parameter.value = popt[parameter.index]
            parameter.error = sigma_p1[parameter.index]
            self.fit_parameters[parameter.index] = popt[parameter.index]

        self.fit_chi2 = np.sum((self.bkgd_func_njet_constrained(bin_centers, *popt) - bin_values)**2 / bin_errors**2)
        self.fit_ndf = len(bin_values) - len(popt)    # tf1_bkgd_njet.GetNDF()
        self.fit_prob = scipy.stats.chi2.sf(self.fit_chi2, self.fit_ndf)

        residuals  = bin_values - self.bkgd_func_njet_constrained(bin_centers, *popt)
        pulls      = residuals / bin_errors
        #print("residuals",residuals)
        #print("pulls",pulls)
        return residuals, pulls


def getPseudoTagProbs(nj, f, e=0.0, d=1.0, norm=1.0):
    nbt = 3    # number of required bTags
    nlt = nj - nbt    # number of selected untagged jets ("light" jets)
    nPseudoTagProb = np.zeros(nlt + 1)

    for npt in range(0, nlt + 1):   # npt is the number of pseudoTags in this combination
        nt = nbt + npt
        nnt = nlt - npt    # number of not tagged

        # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^nlt * (1-pseudoTagProb)^{nlt-npt}
        w_npt = norm * comb(nlt, npt, exact=True) * f**npt * (1 - f)**nnt

        if (nt % 2) == 0:
            w_npt *= 1 + e / nlt**d

        nPseudoTagProb[npt] += w_npt
    return nPseudoTagProb


def getCombinatoricWeight(nj, f, e=0.0, d=1.0, norm=1.0):
    nPseudoTagProb = getPseudoTagProbs(nj, f, e, d, norm)
    return np.sum(nPseudoTagProb[1:])


def loadROOTHists(inputFile):

    #
    #  > From python analysis/tests/dumpROOTToHist.py -o analysis/tests/HistsFromROOTFile.coffea -c passPreSel -r SB
    #
    h = load(inputFile)["Hists"]

    data4b           = h["data4b"]
    data3b           = h["data3b"]
    tt4b             = h["tt4b"]
    tt3b             = h["tt3b"]
    qcd4b            = h["qcd4b"]
    qcd3b            = h["qcd3b"]
    data4b_nTagJets  = h["data4b_nTagJets"]
    tt4b_nTagJets    = h["tt4b_nTagJets"]
    qcd3b_nTightTags = h["qcd3b_nTightTags"]

    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags


def loadCoffeaHists(cfg, *, cut="passPreSel", year="RunII", weightRegion="SB"):

    cutDict = get_cut_dict(cut, cfg.cutList)

    codes = cfg.plotConfig["codes"]
    year = sum if year == "RunII" else year
    region_selection = sum if weightRegion in ["sum", sum] else hist.loc(codes["region"][weightRegion])

    region_year_dict = {
        "year": year,
        "region": region_selection,
    }

    fourTag_dict  = {"tag": hist.loc(codes["tag"]["fourTag"])}
    threeTag_dict = {"tag": hist.loc(codes["tag"]["threeTag"])}

    fourTag_data_dict  = {"process": 'data'} | fourTag_dict | region_year_dict | cutDict
    threeTag_data_dict = {"process": 'data'} | threeTag_dict | region_year_dict | cutDict

    ttbar_list = ['TTTo2L2Nu', 'TTToSemiLeptonic', 'TTToHadronic']
    fourTag_ttbar_dict   = {"process": ttbar_list} | fourTag_dict  | region_year_dict | cutDict
    threeTag_ttbar_dict  = {"process": ttbar_list} | threeTag_dict | region_year_dict | cutDict

    hists = cfg.hists[0]['hists']

    data4b                = hists['selJets_noJCM.n']      [fourTag_data_dict]
    data4b_nTagJets       = hists['tagJets_noJCM.n']      [fourTag_data_dict]

    data3b                = hists['selJets_noJCM.n']      [threeTag_data_dict]
    data3b_nTagJets       = hists['tagJets_loose_noJCM.n'][threeTag_data_dict]
    data3b_nTagJets_tight = hists['tagJets_noJCM.n']      [threeTag_data_dict]

    tt4b                  = hists['selJets_noJCM.n']      [fourTag_ttbar_dict][sum, :]
    tt4b_nTagJets         = hists['tagJets_noJCM.n']      [fourTag_ttbar_dict][sum, :]

    tt3b                  = hists['selJets_noJCM.n']      [threeTag_ttbar_dict][sum, :]
    tt3b_nTagJets         = hists['tagJets_loose_noJCM.n'][threeTag_ttbar_dict][sum, :]
    tt3b_nTagJets_tight   = hists['tagJets_noJCM.n']      [threeTag_ttbar_dict][sum, :]

    qcd4b = copy(data4b)
    qcd4b.view().value = data4b.values() - tt4b.values()
    qcd4b.view().variance = data4b.variances() + tt4b.variances()

    qcd3b = copy(data3b)
    qcd3b.view().value = data3b.values() - tt3b.values()
    qcd3b.view().variance = data3b.variances() + tt3b.variances()

    qcd3b_nTightTags = copy(data3b_nTagJets_tight)
    qcd3b_nTightTags.view().value = data3b_nTagJets_tight.values() - tt3b_nTagJets_tight.values()
    qcd3b_nTightTags.view().variance = data3b_nTagJets_tight.variances() + tt3b_nTagJets_tight.variances()

    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags


def data_from_Hist(inputHist, maxBin=15):
    x_centers = inputHist.axes[0].centers
    values = inputHist.values()
    errors = np.sqrt(inputHist.variances())

    if x_centers[0] == 0.5:
        x_centers = x_centers - 0.5

    return x_centers[0:maxBin], values[0:maxBin], errors[0:maxBin]


#
#  Puth teh number of additional tag jets in teh first 4 bins
#
def prepHists(data4b, qcd3b, tt4b, data4b_nTagJets, tt4b_nTagJets):

    data4b_new_values         = data4b.values()
    data4b_new_variances      = data4b.variances()
    data4b_new_values   [0:4] = data4b_nTagJets.values()   [4:8]
    data4b_new_variances[0:4] = data4b_nTagJets.variances()[4:8]
    data4b.view().value       = data4b_new_values
    data4b.view().variance    = data4b_new_variances

    tt4b_new_values         = tt4b.values()
    tt4b_new_variances      = tt4b.variances()
    tt4b_new_values   [0:4] = tt4b_nTagJets.values()   [4:8]
    tt4b_new_variances[0:4] = tt4b_nTagJets.variances()[4:8]
    tt4b.view().value       = tt4b_new_values
    tt4b.view().variance    = tt4b_new_variances
