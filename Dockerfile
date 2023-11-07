FROM gitlab-registry.cern.ch/cms-cloud/cmssw-docker/cmssw_11_2_5-slc7_amd64_gcc900:2022-06-20-9afcb2be

#COPY set_env.sh /home/cmsusr/CMSSW_11_2_5/src/set_env.sh

RUN bash /opt/cms/entrypoint.sh && \
    cd /home/cmsusr/CMSSW_11_2_5/src/ && \
    eval `/cvmfs/cms.cern.ch/common/scramv1 runtime -sh` && \
    #git cms-addpkg PhysicsTools/ONNXRuntime && \
    #source set_env.sh && \
    /cvmfs/cms.cern.ch/common/scramv1 b clean && \
    /cvmfs/cms.cern.ch/common/scramv1 b   && \
    eval `/cvmfs/cms.cern.ch/common/scramv1 runtime -sh`

