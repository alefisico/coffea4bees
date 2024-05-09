combine_inputs="datacards_HH_oldbkg_woZ/combine_SvB"

text2workspace.py ${combine_inputs}.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/HH:rHH[1,-10,10]' --PO 'map=.*/ZH:rZH[1,-10,10]' --PO 'map=.*/ZZ:rZZ[1,-10,10]'
combine -M AsymptoticLimits ${combine_inputs}.root --redefineSignalPOIs rHH
