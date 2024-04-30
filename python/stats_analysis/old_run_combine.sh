classifiers=( "SvB" "SvB_MA" )

working_dir="HIG-20-011/"

for classifier in ${classifiers[@]};
do
    #### Do postfitplots
#    combine -M MultiDimFit --setParameters rZZ=1,rZH=1,rHH=1 --robustFit 1 -n .fit_s --saveWorkspace --saveFitResult -d ${working_dir}/combine_${classifier}.root
#    mv higgsCombine.fit_s.MultiDimFit.mH120.root ${working_dir}/higgsCombine.fit_s_${classifier}.MultiDimFit.mH120.root
#    mv multidimfit.fit_s.root ${working_dir}/multidimfit.fit_s_${classifier}.root
#    PostFitShapesFromWorkspace -w ${working_dir}/higgsCombine.fit_s_${classifier}.MultiDimFit.mH120.root -f ${working_dir}/multidimfit.fit_s_${classifier}.root:fit_mdf  --total-shapes --postfit --output ${working_dir}/postfit_s_${classifier}.root

    #### do sensitivity
    combine -M Significance ${working_dir}/combine_${classifier}.root --redefineSignalPOIs rHH > ${working_dir}/observed_significance_hh_${classifier}.txt
    combine -M AsymptoticLimits ${working_dir}/combine_${classifier}.root --redefineSignalPOIs rHH > ${working_dir}/observed_limit_hh_${classifier}.txt
    combine -M MultiDimFit --algo cross --cl=0.68 --robustFit 1 -P rHH -d  ${working_dir}/combine_${classifier}.root
    cp higgsCombineTest.MultiDimFit.mH120.root ${working_dir}/higgsCombine.observed_rHH_${classifier}.root

done
