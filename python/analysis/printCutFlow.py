import os, sys
import yaml
import hist
import argparse
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coffea.util import load
from hist.intervals import ratio_uncertainty

def _round(val):
    return round(float(val),1)

def printLine(words):
    print(f'\t{words[0]:<20}\t{words[1]:<10}   {words[2]:<10} \t\t {words[3]:<10}\t{words[4]:<10}')

def printCF(procKey, cf4, cf4_unit, cf3, cf3_unit):

    bar = "-"*10
    print('\n')
    print(procKey,':\n')
    printLine(["Cuts","FourTag","","ThreeTag",""])
    printLine(["",bar,bar,bar,bar])
    printLine(["","weighted","(unit weight)","weighted","(unit weight)"])
    print('\n')
    for cut in cf4.keys():
        printLine([cut,_round(cf4[cut]),_round(cf4_unit[cut]),_round(cf3[cut]),_round(cf3_unit[cut])])
                   
    print("\n")


def add(thisKey):
    print(f"\tadding {thisKey}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-p','--process',   default='data', help='Input process. Default: hists.pkl')
    parser.add_argument('-e','--era',   nargs='+', dest='eras', default=['UL16_preVFPB','UL16_postVFPF','UL17C'], help='Input process. Default: hists.pkl')
    #parser.add_argument('-d', '--datasets', nargs='+', dest='datasets', , help="Name of dataset to run. Example if more than one: -d HH4b ZZ4b")
    args = parser.parse_args()

    with open(f'{args.inputFile}', 'rb') as hfile:
        hists = load(hfile)
        
    cf4      = hists["cutFlowFourTag"]
    cf4_unit = hists["cutFlowFourTagUnitWeight"]
    cf3      = hists["cutFlowThreeTag"]
    cf3_unit = hists["cutFlowThreeTagUnitWeight"]

    eras = args.eras
    eraString = "_".join(eras)
    print(eras)
    print(eraString)
    key = args.process+"_"+eraString
    
    if key not in cf4:
        print(f"summing {key}...")

        cf4      [key] = {}
        cf4_unit [key] = {}
        cf3      [key] = {}
        cf3_unit [key] = {}
        
        for e in eras:

            for cut, v in cf4[args.process+"_"+e].items():
                if cut not in cf4[key]: cf4[key][cut] = 0
                if cut not in cf4_unit[key]: cf4_unit[key][cut] = 0
                if cut not in cf3[key]: cf3[key][cut] = 0
                if cut not in cf3_unit[key]: cf3_unit[key][cut] = 0
                
                cf4[key][cut]      += cf4[args.process+"_"+e][cut]
                cf4_unit[key][cut] += cf4_unit[args.process+"_"+e][cut]
                cf3[key][cut]      += cf3[args.process+"_"+e][cut]
                cf3_unit[key][cut] += cf3_unit[args.process+"_"+e][cut]

    #cutList = hists["cutFlowThreeTagUnitWeight"][key].keys()
    printCF(key, cf4[key], cf4_unit[key], cf3[key], cf3_unit[key])

