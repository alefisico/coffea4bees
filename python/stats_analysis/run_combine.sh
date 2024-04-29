combine_inputs="combine_SvB_oldbkg"

text2workspace.py datacards/${combine_inputs}.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/HH:rHH[1,-10,10]' --PO 'map=.*/ZH:rZH[1,-10,10]' --PO 'map=.*/ZZ:rZZ[1,-10,10]'
combine -M AsymptoticLimits datacards/${combine_inputs}.root --redefineSignalPOIs rHH
