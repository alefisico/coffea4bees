import argparse
from coffea.util import load

def print_counts(counts, name):

    print(f"self.{name} = {'{}'}")
    for k in sorted(counts.keys(),reverse=True):

        print(f"self.{name}['{k}'] = {'{'}",end='')
        for cut in ['passJetMult', 'passPreSel', 'passDiJetMass', 'SR', 'SB', 'passSvB', 'failSvB']:
            print(f"'{cut}' : {round(float(counts[k][cut]),2)}, ",end='')
        print(f"{'}'}")
    print("\n\n")

def print_counts_yaml(counts, name):

    outputFile.write(f"{name}:\n")
    for k in sorted(counts.keys(),reverse=True):

        outputFile.write(f"    {k}:\n")
        for cut in ['passJetMult', 'passPreSel', 'passDiJetMass', 'SR', 'SB', 'passSvB', 'failSvB']:
            outputFile.write(f"        {cut}: {round(float(counts[k][cut]),2)}\n")
        outputFile.write("\n")
    outputFile.write("\n\n")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o','--outputFile', default='knownCounts.yml', help='Input File. Default: hists.pkl')
    args = parser.parse_args()

    outputFile = open(f'{args.outputFile}', 'w')

    with open(f'{args.inputFile}', 'rb') as hfile:
        hists = load(hfile)
        
    cf4      = hists["cutFlowFourTag"]
    cf4_unit = hists["cutFlowFourTagUnitWeight"]
    cf3      = hists["cutFlowThreeTag"]
    cf3_unit = hists["cutFlowThreeTagUnitWeight"]

    print_counts_yaml(cf4, "counts4")
    print_counts_yaml(cf3, "counts3")
    print_counts_yaml(cf4_unit, "counts4_unit")
    print_counts_yaml(cf3_unit, "counts3_unit")



    
