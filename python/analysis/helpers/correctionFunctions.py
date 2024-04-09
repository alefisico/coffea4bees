import pickle

def btagSF_norm(dataset, btagSF_norm_file='ZZ4b/nTupleAnalysis/weights/btagSF_norm.pkl'):
    try:
        with open(btagSF_norm_file, 'rb') as f:
            btagSF_norm = pickle.load(f)
            print(f'btagSF_norm[{dataset}] = {btagSF_norm[dataset]}')
            return btagSF_norm[dataset]
    except FileNotFoundError:
        return 1.0


