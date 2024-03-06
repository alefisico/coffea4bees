import sys
from copy import copy
import argparse
import numpy as np
import os
import scipy.stats

from scipy.optimize import curve_fit
from hist import Hist

sys.path.insert(0, os.getcwd())


from base_class.JCMTools import getCombinatoricWeight, getPseudoTagProbs, loadROOTHists, loadCoffeaHists, data_from_Hist


#
#  To do:
#    - add limit consgtraints
#    - make plots

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
        self.percentError = self.error/self.value*100 if self.value else 0
        print((self.name+" %1.4f +/- %0.5f (%1.1f%%)")%(self.value,self.error,self.percentError))

class jetCombinatoricModel:
    def __init__(self):
        self.pseudoTagProb       = modelParameter("pseudoTagProb",        index=0, lowerLimit=0,   upperLimit= 1, default=0.05)
        self.pairEnhancement     = modelParameter("pairEnhancement",      index=1, lowerLimit=0,   upperLimit= 3, default=1.0,
                                                  #fix=0,
                                                  )
        self.pairEnhancementDecay= modelParameter("pairEnhancementDecay", index=2, lowerLimit=0.1, upperLimit=100, default=0.7,
                                                  #fix=1,
                                                  )
        self.threeTightTagFraction = modelParameter("threeTightTagFraction",   index=3, lowerLimit=0, upperLimit=1000000, default=1000)

        self.parameters         = [self.pseudoTagProb, self.pairEnhancement, self.pairEnhancementDecay, self.threeTightTagFraction]

        self.default_parameters = [] 
        self.fit_parameters = []
        for p in self.parameters:
            self.fit_parameters.append(p)
            self.default_parameters.append(p.default)

        self.nParameters = len(self.parameters)

    def dump(self):
        for parameter in self.parameters:
            parameter.dump()

    def fixParameter(self, name, value):
        for p in self.parameters:
            if name is p.name:
                print(f"Fixing {name} to {value}")
                p.fix = value
                self.fit_parameters.remove(p)
                self.default_parameters.remove(p.default)


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


def nTagPred_values(par, n, tt4b_nTagJets_values, qcd3b_values):
    output = np.zeros(len(n))
    output = copy(tt4b_nTagJets_values)

    for ibin, this_nTag in enumerate(n):
        for nj in range(this_nTag,14):
            nPseudoTagProb = getPseudoTagProbs(nj, par[0],par[1],par[2],threeTightTagFraction)
            output[ibin+4] += nPseudoTagProb[this_nTag-3] * qcd3b_values[nj]

    return np.array(output,float)


def nTagPred_errors(par, n, tt4b_nTagJets_errors, qcd3b_errors):
    output = np.zeros(len(n))
    output = tt4b_nTagJets_errors**2

    #print(f"n {n}")
    
    for ibin, this_nTag in enumerate(n):
        for nj in range(this_nTag,14):
            nPseudoTagProb = getPseudoTagProbs(nj, par[0],par[1],par[2],threeTightTagFraction)
            output[ibin+4] += (nPseudoTagProb[this_nTag-3] * qcd3b_errors[nj])**2
    output = output**0.5            
    #print(f"nTagPred {output}")
    return np.array(output,float)


def write_to_JCM_file(text, value):
    jetCombinatoricModelFile.write(text+"               "+str(value)+"\n")

    jetCombinatoricModelFile_yml.write(text+":\n")
    jetCombinatoricModelFile_yml.write("        "+str(value)+"\n")


# To add

##
##    #jetCombinatoricModelRoot.Close()
##
##    samples=collections.OrderedDict()
##    samples[JCMROOTFileName] = collections.OrderedDict()
##    samples[JCMROOTFileName][data4b.GetName()] = {
##        "label" : ("Data %.1f/fb, "+o.year)%(lumi),
##        "legend": 1,
##        "isData" : True,
##        "ratio" : "numer A",
##        "color" : "ROOT.kBlack"}
##    samples[JCMROOTFileName][qcd3b.GetName()] = {
##        "label" : "Multijet Model",
##        "weight": mu_qcd,
##        "legend": 2,
##        "stack" : 3,
##        "ratio" : "denom A",
##        "color" : "ROOT.kYellow"}
##    if tt4b:
##        samples[JCMROOTFileName][tt4b.GetName()] = {
##            "label" : "t#bar{t}",
##            "legend": 3,
##            "stack" : 2,
##            "ratio" : "denom A",
##            "color" : "ROOT.kAzure-9"}
##    #samples[JCMROOTFileName][tf1_bkgd_njet.GetName()] = {
##    samples[JCMROOTFileName]["background_TH1"] = {
##        "label" : "JCM Fit",
##        "legend": 4,
##        "ratio": "denom A", 
##        "color" : "ROOT.kRed"}
##
##    #xTitle = "Number of b-tags - 4"+" "*31+"Number of Selected Jets"
##    xTitle = "Extra b-tags"+" "*36+"Number of Selected Jets"
##    parameters = {"titleLeft"   : "#bf{CMS} Internal",
##                  "titleCenter" : regionNames[o.weightRegion],
##                  "titleRight"  : cutTitle,
##                  "maxDigits"   : 4,
##                  "ratio"     : True,
##                  "rMin"      : 0,
##                  "rMax"      : 2,
##                  "xMin"      : 0.5,
##                  "xMax"      : 14.5,
##                  "rTitle"    : "Data / Bkgd.",
##                  "xTitle"    : xTitle,
##                  "yTitle"    : "Events",
##                  "legendSubText" : ["",
##                                     "#bf{Fit Result:}",
##                                     "#font[12]{f} = %0.3f #pm %0.1f%%"%(jetCombinatoricModels[cut].pseudoTagProb.value, jetCombinatoricModels[cut].pseudoTagProb.percentError),
##                                     "#font[12]{e} = %0.2f #pm %0.1f%%"%(jetCombinatoricModels[cut].pairEnhancement.value, jetCombinatoricModels[cut].pairEnhancement.percentError),
##                                     "#font[12]{d} = %0.2f #pm %0.1f%%"%(jetCombinatoricModels[cut].pairEnhancementDecay.value, jetCombinatoricModels[cut].pairEnhancementDecay.percentError),
##                                     "#chi^{2}/DoF = %0.2f"%(chi2/ndf),
##                                     "p-value = %2.0f%%"%(prob*100),
##                                     ],
##                  "outputDir" : o.outputDir,
##                  "outputName": "nSelJets"+st+"_"+cut+"_postfit_tf1"}
##
##    PlotTools.plot(samples, parameters)


if __name__ == "__main__":
    
    #parser = optparse.OptionParser()
    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('--noFitWeight',dest='noFitWeight',default="")
    parser.add_argument('-w', '--weightSet',dest="weightSet",default="")
    parser.add_argument('-r',dest="weightRegion",default="")
    parser.add_argument('-c',dest="cut",default="passXWt")
    parser.add_argument('-i', '--inputFile', dest="inputFile", default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o', '--outputDir',dest='outputDir',default="")
    parser.add_argument(      '--ROOTInputs',action="store_true")
    parser.add_argument('-y', '--year',                 dest="year",          default="RunII", help="Year specifies trigger (and lumiMask for data)")
    parser.add_argument('--debug',                 action="store_true")
    parser.add_argument('-l', '--lumi',                 dest="lumi",          default="1",    help="Luminosity for MC normalization: units [pb]")
    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="analysis/metadata/plotsAll.yml",
                        help='Metadata file.')
    
    args = parser.parse_args()
    #o, a = parser.parse_args()
    
    lumi = float(args.lumi)/1000
    
    if not os.path.isdir(args.outputDir):
        os.mkdir(args.outputDir)

    #
    #  Output files
    #
    jetCombinatoricModelName = args.outputDir+"/"+"jetCombinatoricModel_"+args.weightRegion+"_"+args.weightSet+".txt"
    print(jetCombinatoricModelName)
    jetCombinatoricModelFile     = open(jetCombinatoricModelName, "w")
    jetCombinatoricModelFile_yml = open(f'{jetCombinatoricModelName.replace(".txt",".yml")}', 'w')
    
    cut=args.cut

    #
    # Get Hists
    #
    if args.ROOTInputs:
        data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags = loadROOTHists(args.inputFile)
    else:
        data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags = loadCoffeaHists(args.inputFile, args.metadata, 
                                                                                                                     cut=cut, year=args.year, weightRegion=args.weightRegion)

    #
    # Prep Hists
    #
    prepHists(data4b, qcd3b, tt4b, data4b_nTagJets, tt4b_nTagJets)
    
    print("nSelJetsUnweighted", "data4b.Integral()", np.sum(data4b.values()), "\ndata3b.Integral()", np.sum(data3b.values()))
    print("nSelJetsUnweighted", "  tt4b.Integral()", np.sum(tt4b.values()),   "\nqcd3b.Integral()",   np.sum(qcd3b.values()))
    
    mu_qcd = np.sum(qcd4b.values())/np.sum(qcd3b.values())
    n4b = np.sum(data4b.values())
    n5b_true = data4b_nTagJets.values()[5]#GetBinContent(data4b_nTagJets.GetXaxis().FindBin(5))
    threeTightTagFraction = qcd3b_nTightTags.values()[3] / np.sum(qcd3b_nTightTags.values())

    print("threeTightTagFraction",threeTightTagFraction)

    #
    # Define the model
    #
    JCM_model = jetCombinatoricModel()
    JCM_model.fixParameter("threeTightTagFraction", threeTightTagFraction)
    #JCM_model.threeTightTagFraction.fix = threeTightTagFraction

    # 
    #  Updating the errors to have the errors of the templates as well
    # 
    #combined_variances = data4b.variances()
    mu_qcd_bin_by_bin     = np.zeros(len(qcd4b.values()))
    qcd3b_non_zero_filter = qcd3b.values() > 0
    mu_qcd_bin_by_bin[qcd3b_non_zero_filter] = qcd4b.values()[qcd3b_non_zero_filter]/qcd3b.values()[qcd3b_non_zero_filter] # if qcd3b.GetBinContent(ibin) else 0
    data3b_error = np.sqrt(data3b.variances()) * mu_qcd_bin_by_bin
    data3b_variances = data3b_error**2

    combined_variances = data4b.variances() + data3b_variances + tt4b.variances() + tt3b.variances()
    combined_error = np.sqrt(combined_variances)
    previous_error = np.sqrt(data4b.variances())


    #
    #  Print increases in errors
    #
    tt4b_error = np.sqrt(tt4b.variances())
    tt3b_error = np.sqrt(tt3b.variances())

    for ibin in range(len(data4b.values())-1):
        x = data4b.axes[0].centers[ibin]-0.5
        increase = 100*combined_error[ibin]/previous_error[ibin] if previous_error[ibin] else 100
        print(f'{ibin:2}, {x:2.0f}| {previous_error[ibin]:5.1f}, {data3b_error[ibin]:5.1f}, {tt4b_error[ibin]:5.1f}, {tt3b_error[ibin]:5.1f}, {increase:5.0f}%')

    data4b.view().variance = combined_variances


    #
    #  Get data to fit
    #
    bin_centers,             bin_values,           bin_errors = data_from_Hist(data4b)
    _,             tt4b_nTagJets_values, tt4b_nTagJets_errors = data_from_Hist(tt4b_nTagJets)
    _,                      tt4b_values,          _           = data_from_Hist(tt4b)
    _,                     qcd3b_values,         qcd3b_errors = data_from_Hist(qcd3b)

    if args.debug:
        print("bin_centers", bin_centers, len(bin_centers))
        print(bin_values)
        print(bin_errors)

    def objective(x, f, e, d, norm, debug=False):
        nj = x.astype(int) 
        output = np.zeros(len(x))

        nTags = nj + 4
        nTags_pred_result = nTagPred_values([f,e,d,norm], nTags, tt4b_nTagJets_values, qcd3b_values)
        output[0:4] = nTags_pred_result[4:8]
        if debug: print(f"output is {output}")
        
        for ibin, this_nj in enumerate(nj):
            if this_nj < 4: continue

            w = getCombinatoricWeight(this_nj, f,e,d,norm)
            output[this_nj] += w * qcd3b_values[this_nj]
            output[this_nj] += tt4b_values[this_nj]

        return output

    #
    #  Fix the normalizaiton to the threeTightTagFraction
    #
    objective_constrained = lambda x, f, e, d, debug=False: objective(x, f, e, d, JCM_model.threeTightTagFraction.fix, debug)

    #
    #  Give empty bins ~poisson uncertianties
    #
    bin_errors[bin_errors==0] = 3

    if args.debug:
        for ibin, center in enumerate(bin_centers):
            print(f"{ibin} {bin_values[ibin]} {bin_errors[ibin]} {center} {objective_constrained(bin_centers, *JCM_model.default_parameters)[ibin]}")

    #
    # Do the fit
    #
    popt, errs = curve_fit(objective_constrained, bin_centers, bin_values, JCM_model.default_parameters, sigma=bin_errors)

    chi2 = np.sum((objective_constrained(bin_centers, *popt) - bin_values)**2 / bin_errors**2)
    ndf =  len(bin_values) - len(popt) #tf1_bkgd_njet.GetNDF()
    prob = scipy.stats.chi2.sf(chi2, ndf)
    print(f"chi^2 ={chi2}  ndf ={ndf} chi^2/ndf ={chi2/ndf} | p-value ={prob}")

    #
    #  Print the pulls
    #
    print("Pulls:")
    residuals  = bin_values - objective_constrained(bin_centers, *popt)
    pulls      = residuals / bin_errors
    for iBin, res in enumerate(residuals):
        print(f"{iBin:2}| {res:5.1f}  / {bin_errors[iBin]:5.1f} = {pulls[iBin]:4.1f}")

    #
    #  Print the fit parameters
    #
    sigma_p1 = [np.absolute(errs[i][i])**0.5 for i in range(len(popt))]
    for parameter in JCM_model.parameters:
        if parameter.fix:
            parameter.value = parameter.fix
            parameter.error = 0
            continue
            
        parameter.value = popt[parameter.index]
        parameter.error = sigma_p1[parameter.index]
    
    JCM_model.dump()

    for parameter in JCM_model.parameters:
        write_to_JCM_file(parameter.name+"_"+cut,             parameter.value)
        write_to_JCM_file(parameter.name+"_"+cut+"_err",      parameter.error)
        write_to_JCM_file(parameter.name+"_"+cut+"_pererr",   parameter.percentError)
    
    write_to_JCM_file("chi^2"     ,chi2)
    write_to_JCM_file("ndf"       ,ndf)
    write_to_JCM_file("chi^2/ndf" ,chi2/ndf)
    write_to_JCM_file("p-value"   ,prob)

    n5b_pred = nTagPred_values(popt, bin_centers.astype(int)+4, tt4b_nTagJets_values, qcd3b_values)[5]
    n5b_pred_error = nTagPred_errors(popt,bin_centers.astype(int)+4,tt4b_nTagJets_errors, qcd3b_errors)[5]
    print(f"Fitted number of 5b events: {n5b_pred:5.1f} +/- {n5b_pred_error:5f}")
    print(f"Actual number of 5b events: {n5b_true:5.1f}, ({(n5b_true-n5b_pred)/n5b_pred**0.5:3.1f} sigma pull)")
    write_to_JCM_file("n5b_pred"   ,n5b_pred)
    write_to_JCM_file("n5b_true"   ,n5b_true)

##    background_TH1 = data4b.Clone("background_TH1")
##    background_TH1.Reset()
##    
##    # Reset bin error for plotting
##    for bin in range(1,data4b.GetSize()-2):
##        if data4b.GetBinContent(bin) > 0:
##            data4b_error = data4b.GetBinContent(bin)**0.5
##            data4b.SetBinError(bin, data4b_error)
##    
##        binCenter = int(background_TH1.GetBinCenter(bin))
##        bc = tf1_bkgd_njet.Eval(binCenter)
##        background_TH1.SetBinContent(bin, bc)
##        if binCenter < 4:
##            bc, be = nTagPred(tf1_bkgd_njet.GetParameters(), binCenter+4)
##        else:
##            te = tt4b.GetBinError(bin) if tt4b else 0
##            qc = qcd3b.GetBinContent(bin)
##            qe = qcd3b.GetBinError(bin)
##            be = (te**2 + (qe*bc/qc if qc else 0)**2)**0.5
##        background_TH1.SetBinError(bin, be)
##    background_TH1.Write()
##    
##    c=ROOT.TCanvas(cut+"_postfit_tf1","Post-fit")
##    #data4b.SetLineColor(ROOT.kBlack)
##    data4b.GetYaxis().SetTitleOffset(1.5)
##    data4b.GetYaxis().SetTitle("Events")
##    xTitle = "Number of b-tags - 4"+" "*63+"Number of Selected Jets"
##    data4b.GetXaxis().SetTitle(xTitle)
##    data4b.Draw("P EX0")
##    data4b.Write()
##    qcdDraw = ROOT.TH1F(qcd3b)
##    qcdDraw.SetName(qcd3b.GetName()+"draw")
##    qcd3b.Write()
##    
##    stack = ROOT.THStack("stack","stack")
##    #mu_qcd = qcd4b.Integral()/qcdDraw.Integral()
##    print("mu_qcd = %f +/- %f%%"%(mu_qcd, 100*n4b**-0.5))
##    write_to_JCM_file("mu_qcd_"+cut, str(mu_qcd))
##    #jetCombinatoricModelFile.write("mu_qcd_"+cut+"       "+str(mu_qcd)+"\n")
##    qcdDraw.Scale(mu_qcd)
##    qcdDraw.SetLineColor(ROOT.kMagenta)
##    #stack.Add(qcdDraw,"hist")
##    #stack.Draw("HIST SAME")
##    if tt4b:
##        stack.Add(tt4b)
##        tt4b.Write()
##    stack.Add(qcdDraw)
##    #qcdDraw.Write()
##    stack.Draw("HIST SAME")
##    #qcd3b.Draw("HIST SAME")
##    data4b.SetStats(0)
##    data4b.SetMarkerStyle(20)
##    data4b.SetMarkerSize(0.7)
##    data4b.Draw("P EX0 SAME axis")
##    data4b.Draw("P EX0 SAME")
##    background_TH1.SetLineColor(ROOT.kRed)
##    background_TH1.Draw("HIST SAME")
##    #tf1_bkgd_njet.SetLineColor(ROOT.kRed)
##    #tf1_bkgd_njet.Draw("SAME")
##    tf1_bkgd_njet.Write()
##    
##    xleg, yleg = [0.67, 0.9-0.035], [0.9-0.06*4, 0.9-0.035]
##    leg = ROOT.TLegend(xleg[0], yleg[0], xleg[1], yleg[1])
##    leg.AddEntry(data4b, "Data "+str(lumi)+"/fb, "+args.year)
##    leg.AddEntry(qcdDraw, "Multijet Model")
##    if tt4b:
##        leg.AddEntry(tt4b, "t#bar{t}")
##    leg.AddEntry(background_TH1, "JCM Fit")
##    #leg.AddEntry(tf1_bkgd_njet, "JCM Fit")
##    leg.Draw()
##    
##    c.Update()
##    print(c.GetFrame().GetY1(),c.GetFrame().GetY2())
##    line=ROOT.TLine(3.5,-5000,3.5,c.GetFrame().GetY2())
##    line.SetLineColor(ROOT.kBlack)
##    line.Draw()
##    histName = args.outputDir+"/"+"nJets_"+cut+"_postfit_tf1.pdf"
##    print(histName)
##    c.SaveAs(histName)
    
    
    jetCombinatoricModelFile.close()
    jetCombinatoricModelFile_yml.close()
