echo "************************************************************************************************"
echo "************************************************************************************************"
echo "This script will be deprecated soon."
echo "Please use the run_coffea function in coffea4bees/python/run_container instead."
echo "To learn about the script do to python and run: ./run_container --help"
echo "To run run_container as this script does, do: ./run_container"
echo "To run a command inside the container, without leaving the container open, you can do:"
echo "./run_container python runner.py --help"
echo "************************************************************************************************"
echo "************************************************************************************************"

if ! [ -f shell  ]; then
    echo "No ./shell file in this folder. If this file does not exist check the README for more information"
else
    ./shell ../gitlab-registry.cern.ch/cms-cmu/coffea4bees:latest
fi
