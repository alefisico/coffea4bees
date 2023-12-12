import argparse
from coffea.util import load

def printCounts(counts, name):

    print(f"self.{name} = {'{}'}")
    for k in sorted(counts.keys(),reverse=True):

        print(f"self.{name}['{k}'] = {'{'}",end='')
        for cut in ['passJetMult', 'passPreSel', 'passDiJetMass', 'SR', 'SB', 'passSvB', 'failSvB']:
            print(f"'{cut}' : {round(float(counts[k][cut]),2)}, ",end='')
        print(f"{'}'}")
    print("\n\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-p','--process',   default='data', help='Input process. Default: hists.pkl')
    parser.add_argument('-e','--era',   nargs='+', dest='eras', default=['UL17C'], help='Input process. Default: hists.pkl')
    #parser.add_argument('-d', '--datasets', nargs='+', dest='datasets', , help="Name of dataset to run. Example if more than one: -d HH4b ZZ4b")
    #parser.add_argument('-p','--process',   default='data', help='Input process. Default: hists.pkl')
    args = parser.parse_args()

    

    with open(f'{args.inputFile}', 'rb') as hfile:
        hists = load(hfile)
        
    cf4      = hists["cutFlowFourTag"]
    cf4_unit = hists["cutFlowFourTagUnitWeight"]
    cf3      = hists["cutFlowThreeTag"]
    cf3_unit = hists["cutFlowThreeTagUnitWeight"]



    printCounts(cf4, "counts4")
    printCounts(cf3, "counts3")
    printCounts(cf4_unit, "counts4_unit")
    printCounts(cf3_unit, "counts3_unit")

    
