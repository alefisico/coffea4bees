if ! [ -f shell  ]; then
    echo "No ./shell file in this folder. If this file does not exist check the README for more information"
else
    ./shell ../gitlab-registry.cern.ch/cms-cmu/coffea4bees:latest
fi
