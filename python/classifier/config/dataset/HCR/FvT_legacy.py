# TODO: this should be removed after new skim
from classifier.task import ArgParser

from . import FvT, _picoAOD, legacy


class Train(FvT.Train, legacy._CommonTrain):
    argparser = ArgParser()
    argparser.remove_argument("--no-JCM")
    defaults = {
        "no_JCM": True,  # JCM is already included in mcPseudoTagWeight
    }


class TrainBaseline(_picoAOD.Background, Train): ...


class Eval(FvT.Eval, legacy.Eval): ...
