datacard_folder=$1

text2workspace.py ${datacard_folder}/combine_SvB.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/HH:rHH[1,-10,10]' --PO 'map=.*/ZH:rZH[1,-10,10]' --PO 'map=.*/ZZ:rZZ[1,-10,10]'
combine -M AsymptoticLimits ${datacard_folder}/combine_SvB.root --redefineSignalPOIs rHH

text2workspace.py ${datacard_folder}/combine_SvB_MA.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/HH:rHH[1,-10,10]' --PO 'map=.*/ZH:rZH[1,-10,10]' --PO 'map=.*/ZZ:rZZ[1,-10,10]'
combine -M AsymptoticLimits ${datacard_folder}/combine_SvB_MA.root --redefineSignalPOIs rHH
