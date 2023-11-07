#!/bin/sh
### Script to set CMSSW enviroment

cmssw_version="CMSSW_11_1_1"

if [ -z "$CMSSW_BASE" ];
then
    echo "Setting base CMSSW environment"
    cmsrel $cmssw_version
    cd $cmssw_version/src/
    cmsenv
else
    echo "$cmssw_version already set."
fi

git cms-addpkg PhysicsTools/ONNXRuntime
git cms-merge-topic patrickbryant:MakePyBind11ParameterSetsIncludingCommandLineArguments
git clone https://github.com/patrickbryant/nTupleAnalysis.git
git clone https://github.com/johnalison/nTupleHelperTools.git
git clone https://github.com/johnalison/TriggerEmulator.git
git clone https://github.com/patrickbryant/ZZ4b.git

git clone https://github.com/cms-nanoAOD/nanoAOD-tools.git PhysicsTools/NanoAODTools

