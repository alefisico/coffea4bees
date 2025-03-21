import sys
from copy import copy
import argparse
import numpy as np
import os
from hist import Hist

sys.path.insert(0, os.getcwd())
import base_class.plots.iPlot_config as cfg
from base_class.JCMTools import getCombinatoricWeight, getPseudoTagProbs, loadROOTHists, loadCoffeaHists, data_from_Hist, prepHists, jetCombinatoricModel
from base_class.plots.plots import load_config, load_hists, read_axes_and_cuts, makePlot
from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel as JCMModel
import matplotlib.pyplot as plt

#
#  To do:
#    - add limit consgtraints
#    - Ration of data to JCM in plots


def write_to_JCM_file(text, value):
    jetCombinatoricModelFile.write(text + "               " + str(value) + "\n")

    jetCombinatoricModelFile_yml.write(text + ":\n")
    jetCombinatoricModelFile_yml.write("        " + str(value) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='make JCM weights', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--noFitWeight', dest='noFitWeight', default="")
    parser.add_argument('-w', '--weightSet', dest="weightSet", default="")
    parser.add_argument('-r', dest="weightRegion", default="SB")
    parser.add_argument('--data4bName', default="data")
    parser.add_argument('-c', dest="cut", default="passPreSel")
    parser.add_argument('-fix_e', action="store_true")
    parser.add_argument('-fix_d', action="store_true")
    parser.add_argument('-i', '--inputFile',  nargs="+", dest="inputFile", default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o', '--outputDir', dest='outputDir', default="")
    parser.add_argument('--ROOTInputs', action="store_true")
    parser.add_argument('-y', '--year',                 dest="year",          default="RunII",  help="Year specifies trigger (and lumiMask for data)")
    parser.add_argument('--debug',                 action="store_true")
    parser.add_argument('-l', '--lumi',                 dest="lumi",          default="1",    help="Luminosity for MC normalization: units [pb]")
    parser.add_argument('--combine_input_files', action="store_true", help='')
    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="plots/metadata/plotsJCM.yml",
                        help='Metadata file.')

    args = parser.parse_args()
    # o, a = parser.parse_args()

    lumi = float(args.lumi) / 1000

    if not os.path.isdir(args.outputDir):
        os.mkdir(args.outputDir)

    #
    #  Output files
    #
    jetCombinatoricModelName = args.outputDir + "/" + "jetCombinatoricModel_" + args.weightRegion + "_" + args.weightSet + ".txt"
    print(jetCombinatoricModelName)
    jetCombinatoricModelFile     = open(jetCombinatoricModelName, "w")
    jetCombinatoricModelFile_yml = open(f'{jetCombinatoricModelName.replace(".txt",".yml")}', 'w')

    cut = args.cut

    #
    # Get Hists
    #
    if args.ROOTInputs:
        inputFile = args.inputFile[0]
        data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags = loadROOTHists(inputFile)
    else:
        cfg.plotConfig = load_config(args.metadata)
        cfg.hists = load_hists(args.inputFile)
        cfg.combine_input_files = args.combine_input_files
        cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

        data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags = loadCoffeaHists(cfg,
                                                                                                                     cut=cut, year=args.year, weightRegion=args.weightRegion,
                                                                                                                     data4bName=args.data4bName)

    #
    # Prep Hists
    #
    prepHists(data4b, qcd3b, tt4b, data4b_nTagJets, tt4b_nTagJets)

    print("nSelJetsUnweighted", "data4b.Integral()", np.sum(data4b.values()), "\ndata3b.Integral()", np.sum(data3b.values()))
    print("nSelJetsUnweighted", "  tt4b.Integral()", np.sum(tt4b.values()),   "\nqcd3b.Integral()",   np.sum(qcd3b.values()))

    mu_qcd = np.sum(qcd4b.values()) / np.sum(qcd3b.values())
    # n4b = np.sum(data4b.values())
    threeTightTagFraction = qcd3b_nTightTags.values()[3] / np.sum(qcd3b_nTightTags.values())

    print("threeTightTagFraction", threeTightTagFraction)

    #
    #  Updating the errors to have the errors of the templates as well
    #
    mu_qcd_bin_by_bin     = np.zeros(len(qcd4b.values()))
    qcd3b_non_zero_filter = qcd3b.values() > 0
    mu_qcd_bin_by_bin[qcd3b_non_zero_filter] = np.abs(qcd4b.values()[qcd3b_non_zero_filter] / qcd3b.values()[qcd3b_non_zero_filter])
    mu_qcd_bin_by_bin[mu_qcd_bin_by_bin < 0] = 0
    data3b_error = np.sqrt(data3b.variances()) * mu_qcd_bin_by_bin
    data3b_variances = data3b_error**2

    #
    #  Set poission errors
    #
    data4b_variance = data4b.variances()
    data4b_variance[data4b_variance == 0] = 1.17

    combined_variances = data4b.variances() + data3b_variances + tt4b.variances() + tt3b.variances()
    combined_error = np.sqrt(combined_variances)
    previous_error = np.sqrt(data4b.variances())
    data4b.view().variance = combined_variances

    #
    #  Print increases in errors
    #
    tt4b_error = np.sqrt(tt4b.variances())
    tt3b_error = np.sqrt(tt3b.variances())

    for ibin in range(len(data4b.values()) - 1):
        x = data4b.axes[0].centers[ibin] - 0.5
        # increase = 100 * combined_error[ibin] / previous_error[ibin] if previous_error[ibin] else 100
        increase = 100 * np.sqrt(data4b.variances()[ibin]) / previous_error[ibin] if previous_error[ibin] else 100
        print(f'{ibin:2}, {x:2.0f}| {data4b.values()[ibin]:9.1f} | {previous_error[ibin]:5.1f}, {data3b_error[ibin]:5.1f}, {tt4b_error[ibin]:5.1f}, {tt3b_error[ibin]:5.1f}, {increase:5.0f}%')

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

    #
    # Define the model
    #
    JCM_model = jetCombinatoricModel(tt4b_nTagJets=tt4b_nTagJets_values, tt4b_nTagJets_errors=tt4b_nTagJets_errors, qcd3b=qcd3b_values, qcd3b_errors=qcd3b_errors, tt4b=tt4b_values)
    #JCM_model.fixParameter("threeTightTagFraction", threeTightTagFraction)

    if args.fix_e:
        JCM_model.fixParameter_e_d_norm(threeTightTagFraction)
    elif args.fix_d:
        JCM_model.fixParameter_d_norm(threeTightTagFraction)
    else:
        JCM_model.fixParameter_norm(threeTightTagFraction)


    #
    #  Give empty bins ~poisson uncertianties
    #
    bin_errors[bin_errors == 0] = 1.17

    if args.debug:
        for ibin, center in enumerate(bin_centers):
            print(f"{ibin} {bin_values[ibin]} {bin_errors[ibin]} {center} {objective_constrained(bin_centers, *JCM_model.default_parameters)[ibin]}")

    #
    # Do the fit
    #
    residuals, pulls = JCM_model.fit(bin_centers, bin_values, bin_errors)
    print(f"chi^2 ={JCM_model.fit_chi2}  ndf ={JCM_model.fit_ndf} chi^2/ndf ={JCM_model.fit_chi2/JCM_model.fit_ndf} | p-value ={JCM_model.fit_prob}")

    #
    #  Print the pulls
    #
    print("Pulls:")
    for iBin, res in enumerate(residuals):
        print(f"{iBin:2}| {res:5.1f}  / {bin_errors[iBin]:5.1f} = {pulls[iBin]:4.1f}")

    #
    #  Print the fit parameters
    #
    JCM_model.dump()

    for parameter in JCM_model.parameters:
        write_to_JCM_file(parameter.name + "_" + cut,             parameter.value)
        write_to_JCM_file(parameter.name + "_" + cut + "_err",      parameter.error)
        write_to_JCM_file(parameter.name + "_" + cut + "_pererr",   parameter.percentError)

    write_to_JCM_file("mu_qcd",    mu_qcd)
    write_to_JCM_file("chi^2",     JCM_model.fit_chi2)
    write_to_JCM_file("ndf",       JCM_model.fit_ndf)
    write_to_JCM_file("chi^2/ndf", JCM_model.fit_chi2 / JCM_model.fit_ndf)
    write_to_JCM_file("p-value",   JCM_model.fit_prob)

    n5b_true = data4b_nTagJets.values()[5]
    nTag_pred = JCM_model.nTagPred_values(bin_centers.astype(int) + 4)
    n5b_pred = nTag_pred[5]
    n5b_pred_error = JCM_model.nTagPred_errors(bin_centers.astype(int) + 4)[5]
    print(f"Fitted number of 5b events: {n5b_pred:5.1f} +/- {n5b_pred_error:5f}")
    print(f"Actual number of 5b events: {n5b_true:5.1f}, ({(n5b_true-n5b_pred)/n5b_pred**0.5:3.1f} sigma pull)")
    write_to_JCM_file("n5b_pred", n5b_pred)
    write_to_JCM_file("n5b_true", n5b_true)

    #
    #   Write the event weights
    #
    write_to_JCM_file("JCM_weights",JCM_model.getCombinatoricWeightList())

    jetCombinatoricModelFile.close()
    jetCombinatoricModelFile_yml.close()


    #
    #  Read back the weights and check that they are consistent
    #
    JCM = JCMModel(jetCombinatoricModelName.replace(".txt",".yml"))
    for i in range(1,13):
        jets = np.array(range(i))
        diff = JCM([jets])[0] - JCM.JCM_weights[i-1]
        if diff > 0.001:
            print("ERROR nPSeudoTagJets",i,JCM([jets]), "vs", JCM.JCM_weights[i-1])


    #
    #  Plots
    #
    if not args.ROOTInputs:

        #
        #  Sclae QCD by mu_qcd
        #
        for p in ["data_3tag", "TTTo2L2Nu_3tag", "TTToSemiLeptonic_3tag", "TTToHadronic_3tag"]:
            cfg.plotConfig["stack"]["MultiJet"]["sum"][p]["scalefactor"]            *= mu_qcd

        #
        #  Plot the jet multiplicity
        #
        nJet_pred = JCM_model.nJetPred_values(bin_centers.astype(int))
        nJet_pred[0:4] = 0

        #
        # Add dummy values to add the JCM process
        #
        dummy_data = {
            'process': ['JCM'],
            'year': ['UL18'],  'tag': "fourTag",  'region': "SB",
            'passPreSel': [True], 'passSvB': [False],  'failSvB': [False],   'n': [0],
        }
        try:
            cfg.hists[0]["selJets_noJCM.n"].fill(**dummy_data)
            noSvB = False
        except:
            del dummy_data['passSvB'], dummy_data['failSvB']
            cfg.hists[0]["selJets_noJCM.n"].fill(**dummy_data)
            noSvB = True

        #
        # OVerwrite with predicted values
        #
        for iBin in range(14):
            if noSvB:
                cfg.hists[0]["selJets_noJCM.n"]["JCM", "UL18", 1, 1, True, iBin] = (nJet_pred[iBin], 0)
            else:
                cfg.hists[0]["selJets_noJCM.n"]["JCM", "UL18", 1, 1, True, False, False, iBin] = (nJet_pred[iBin], 0)

        plot_options = {"doRatio": True,
                        "xlim": [4, 15],
                        "rlim": [0, 2],
                        "debug" : False
                        }

        fig, ax = makePlot(cfg, var="selJets_noJCM.n",
                           cut="passPreSel", region="SB",
                           #outputFolder=args.outputDir,
                           **plot_options)

        fit_text = ""
        plot_param_name = {"pseudoTagProb": "f",
                           "pairEnhancement": "e",
                           "pairEnhancementDecay": "d"}
        for parameter in JCM_model.parameters:
            if parameter.name == "threeTightTagFraction":
                continue
            fit_text += f"  {plot_param_name[parameter.name]} = {round(parameter.value,2)} +/- {round(parameter.error,3)}  ({round(parameter.percentError,1)}%)\n"
        fit_text  += f"  $\chi^2$ / DoF = {round(JCM_model.fit_chi2,1)} / {JCM_model.fit_ndf} = {round(JCM_model.fit_chi2/JCM_model.fit_ndf,1)}\n"
        fit_text += f"  p-value: {round(100*JCM_model.fit_prob)}%\n"


        plt.text(10, 6, "Fit Result:", fontsize=20, color='black', fontweight='bold',
                 horizontalalignment='left', verticalalignment='center')

        plt.text(10, 5.15, fit_text, fontsize=15, color='black',
                 horizontalalignment='left', verticalalignment='center')

        fig.savefig(args.outputDir+"/selJets_noJCM_n.pdf")

        #
        #  Plot NTagged Jets
        #
        cfg.hists[0]["tagJets_noJCM.n"].fill(**dummy_data)

        #
        # OVerwrite with predicted values
        #
        for iBin in range(15):
            if noSvB: cfg.hists[0]["tagJets_noJCM.n"]["JCM", "UL18", 1, 1, True, iBin] = (nTag_pred[iBin], 0)
            else: cfg.hists[0]["tagJets_noJCM.n"]["JCM", "UL18", 1, 1, True, False, False, iBin] = (nTag_pred[iBin], 0)

        plot_options = {"doRatio": True,
                        "xlim": [4, 8],
                        "yscale": "log",
                        "rlim": [0.8, 1.2],
                        "ylim": [0.1, None]
                        }
        fig, ax = makePlot(cfg, var="tagJets_noJCM.n",
                           cut="passPreSel", region="SB",
                           #outputFolder=args.outputDir,
                           **plot_options)


        #plt.text(10, 6, "Fit Result:", fontsize=20, color='black', fontweight='bold',
        #         horizontalalignment='left', verticalalignment='center')
        #
        #plt.text(10, 5, fit_text, fontsize=15, color='black',
        #         horizontalalignment='left', verticalalignment='center')

        fig.savefig(args.outputDir+"/tagJets_noJCM_n.pdf")
