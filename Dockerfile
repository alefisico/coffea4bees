FROM gitlab-registry.cern.ch/cms-cloud/cmssw-docker/cmssw_11_2_5-slc7_amd64_gcc900:2022-06-20-9afcb2be

RUN bash /opt/cms/entrypoint.sh && \
    cd /home/cmsusr/CMSSW_11_2_5/src/ && \
    eval `/cvmfs/cms.cern.ch/common/scramv1 runtime -sh` && \
    git cms-addpkg PhysicsTools/ONNXRuntime && \
    /cvmfs/cms.cern.ch/common/scramv1 b clean && \
    /cvmfs/cms.cern.ch/common/scramv1 b   && \

