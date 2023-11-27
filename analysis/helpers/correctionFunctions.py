import pickle

def btagSF_norm(dataset, btagSF_norm_file='ZZ4b/nTupleAnalysis/weights/btagSF_norm.pkl'):
    try:
        with open(btagSF_norm_file, 'rb') as f:
            btagSF_norm = pickle.load(f)
            print(f'btagSF_norm[{dataset}] = {btagSF_norm[dataset]}')
            return btagSF_norm[dataset]
    except FileNotFoundError:
        return 1.0

def btagVariations(JECSyst='', systematics=False):
    btagVariations = ['central']
    if 'jes' in JECSyst:
        if 'Down' in JECSyst:
            btagVariations = ['down'+JECSyst.replace('Down','')]
        if 'Up' in JECSyst:
            btagVariations = ['up'+JECSyst.replace('Up','')]
    if systematics:
        btagVariations += ['down_hfstats1', 'up_hfstats1']
        btagVariations += ['down_hfstats2', 'up_hfstats2']
        btagVariations += ['down_lfstats1', 'up_lfstats1']
        btagVariations += ['down_lfstats2', 'up_lfstats2']
        btagVariations += ['down_hf', 'up_hf']
        btagVariations += ['down_lf', 'up_lf']
        btagVariations += ['down_cferr1', 'up_cferr1']
        btagVariations += ['down_cferr2', 'up_cferr2']
    return btagVariations


def juncVariations(systematics=False, years = ['YEAR']):
    juncVariations = ['JES_Central']
    if systematics:
        juncSources = ['JES_FlavorQCD',
                       'JES_RelativeBal',
                       'JES_HF',
                       'JES_BBEC1',
                       'JES_EC2',
                       'JES_Absolute',
                       'JES_Total']
        for year in years:
            juncSources += [f'JES_Absolute_{year}',
                            f'JES_HF_{year}',
                            f'JES_EC2_{year}',
                            f'JES_RelativeSample_{year}',
                            f'JES_BBEC1_{year}',
                            f'JER_{year}']
        juncVariations += [f'{juncSource}_{direction}' for juncSource in juncSources for direction in ['up', 'down']]
    return juncVariations

