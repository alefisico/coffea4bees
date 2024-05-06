#!/bin/sh

inputdir="./" #"root://cmseos.fnal.gov//store/user/algomez/XX4b/20231115/"
classifiers=( "SvB" "SvB_MA" )
years=( "2016" "2017" "2018" )
channel=( "hh:10" ) # "zz:4" "zh:5" )
signals=( "HH:HH4b" "ZZ:ZZ4b" "ZH:bothZH4b" )
mixFile=${inputdir}"files_HIG-20-011/hists_closure_3bDvTMix4bDvT_SR_weights_newSBDef.root"

if [ ! -d "HIG-20-011" ]; then
    mkdir HIG-20-011
fi

for classifier in ${classifiers[@]};
do
    outFileData="HIG-20-011/hist_${classifier}.root"
    outFileMix="HIG-20-011/hist_closure_${classifier}.root"
    rm $outFileData $outFileMix

    if [[ $mixFile == *"MA"* ]]; then
        mixFile=${mixFile//weights_/weights_MA_}
    fi

    for year in ${years[@]};
    do
        for ich in "${channel[@]}";
        do
            ch="${ich%%:*}"
            rebin="${ich##*:}"
            var="${classifier}_ps_${ch}"

            for signal in "${signals[@]}"
            do
                signal_name="${signal%%:*}"
                signal_title="${signal##*:}"
                IFS=","
                set -- $signal
                #Sigmal templates to data file
                echo "python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/${signal_title}${year}/hists.root -o ${outFileData}  -r SR --var ${var} --channel ${ch}${year} -n ${signal_name} --tag four --cut passPreSel --rebin ${rebin} --systematics files_HIG-20-011/systematics.pkl"
                python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/${signal_title}${year}/hists.root -o ${outFileData}  -r SR --var ${var} --channel ${ch}${year} -n ${signal_name} --tag four --cut passPreSel --rebin ${rebin} --systematics files_HIG-20-011/systematics.pkl

                ##Sigmal templates to data file - no rebin
                #python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/${signal_title}${year}/hists.root -o ${outFileData//hist_/hist_no_rebin_}  -r SR --var ${var} --channel ${ch}${year} -n ${signal_name} --tag four --cut passPreSel --rebin 2 --systematics files_HIG-20-011/systematics.pkl
            done

            #Signal templates to mixed data file
            cp $outFileData $outFileMix

            #Multijet template to data file
            echo "python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/data${year}/hists_j_r.root -o ${outFileData}  -r SR --var ${var} --channel ${ch}${year} -n mj --tag three --cut passPreSel --rebin ${rebin} --systematics ${inputdir}files_HIG-20-011/closureResults_${classifier}_${ch}.pkl"
            python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/data${year}/hists_j_r.root -o ${outFileData}  -r SR --var ${var} --channel ${ch}${year} -n mj --tag three --cut passPreSel --rebin ${rebin} --systematics ${inputdir}files_HIG-20-011/closureResults_${classifier}_${ch}.pkl

            #Multijet template to mixed data file
            echo "python old_make_combine_hists.py -i ${mixFile} -o ${outFileMix} --TDirectory 3bDvTMix4bDvT_v0/${ch}${year} --var multijet --channel ${ch}${year} -n mj --rebin ${rebin} --systematics ${inputdir}files_HIG-20-011/closureResults_${classifier}_${ch}.pkl"
            python old_make_combine_hists.py -i ${mixFile} -o ${outFileMix} --TDirectory 3bDvTMix4bDvT_v0/${ch}${year} --var multijet --channel ${ch}${year} -n mj --rebin ${rebin} --systematics ${inputdir}files_HIG-20-011/closureResults_${classifier}_${ch}.pkl

             #ttbar template to data file
            echo "python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/TT${year}/hists_j_r.root -o ${outFileData}  -r SR --var ${var} --channel ${ch}${year} -n tt --tag four --cut passPreSel --rebin ${rebin}"
            python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/TT${year}/hists_j_r.root -o ${outFileData}  -r SR --var ${var} --channel ${ch}${year} -n tt --tag four --cut passPreSel --rebin ${rebin}

            #ttbar template to mixed data file
            echo "python old_make_combine_hists.py -i ${mixFile} -o ${outFileMix} --TDirectory 3bDvTMix4bDvT_v0/${ch}${year} --var ttbar --channel ${ch}${year} -n tt --rebin ${rebin}"
            python old_make_combine_hists.py -i ${mixFile} -o ${outFileMix} --TDirectory 3bDvTMix4bDvT_v0/${ch}${year} --var ttbar --channel ${ch}${year} -n tt --rebin ${rebin}

            #data_obs to data file
            echo "python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/data${year}/hists_j_r.root -o ${outFileData}  -r SR --var ${var} --channel ${ch}${year} -n data_obs --tag four --cut passPreSel --rebin ${rebin}"
            python old_make_combine_hists.py -i ${inputdir}files_HIG-20-011/data${year}/hists_j_r.root -o ${outFileData}  -r SR --var ${var} --channel ${ch}${year} -n data_obs --tag four --cut passPreSel --rebin ${rebin}

            #mix data_obs to mixed data file
            echo "python old_make_combine_hists.py -i ${mixFile} -o ${outFileMix} --TDirectory 3bDvTMix4bDvT_v0/${ch}${year} --var data_obs --channel ${ch}${year} -n data_obs --rebin ${rebin}"
            python old_make_combine_hists.py -i ${mixFile} -o ${outFileMix} --TDirectory 3bDvTMix4bDvT_v0/${ch}${year} --var data_obs --channel ${ch}${year} -n data_obs --rebin ${rebin}

        done
    done

    # Do Workspace
    #make datacards
    echo "python old_make_datacard.py HIG-20-011/combine_${classifier}.txt HIG-20-011/hist_${classifier}.root files_HIG-20-011/systematics.pkl files_HIG-20-011/closureResults_${classifier}_"
    python old_make_datacard.py HIG-20-011/combine_${classifier}.txt HIG-20-011/hist_${classifier}.root ${inputdir}files_HIG-20-011/systematics.pkl ${inputdir}files_HIG-20-011/closureResults_${classifier}_

    echo "python old_make_datacard.py HIG-20-011/combine_stat_only_${classifier}.txt HIG-20-011/hist_${classifier}.root"
    python old_make_datacard.py HIG-20-011/combine_stat_only_${classifier}.txt HIG-20-011/hist_${classifier}.root

    echo "python old_make_datacard.py HIG-20-011/combine_closure_${classifier}.txt HIG-20-011/hist_closure_${classifier}.root ${inputdir}files_HIG-20-011/systematics.pkl ${inputdir}files_HIG-20-011/closureResults_${classifier}_"
    python old_make_datacard.py HIG-20-011/combine_closure_${classifier}.txt HIG-20-011/hist_closure_${classifier}.root ${inputdir}files_HIG-20-011/systematics.pkl ${inputdir}files_HIG-20-011/closureResults_${classifier}_

    ## create workspace
    for icomb in HIG-20-011/combine_${classifier}.txt HIG-20-011/combine_stat_only_${classifier}.txt HIG-20-011/combine_closure_${classifier}.txt;
    do
        echo "text2workspace.py ${icomb} -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/HH:rHH[1,-10,10]'" #" --PO 'map=.*/ZZ:rZZ[1,-10,10]'  --PO 'map=.*/ZH:rZH[1,-10,10]'"
        text2workspace.py ${icomb} -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/HH:rHH[1,-10,10]' # --PO 'map=.*/ZZ:rZZ[1,-10,10]'  --PO 'map=.*/ZH:rZH[1,-10,10]'
    done

done
