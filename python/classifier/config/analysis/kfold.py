import sys
from typing import Iterable

from classifier.task import Analysis, ArgParser, converter

from ..setting import IO, ResultKey


class Merge(Analysis):
    argparser = ArgParser()
    argparser.add_argument(
        "--name",
        help="the name of the merged friend tree",
        default=None,
    )
    argparser.add_argument(
        "--base",
        default="eval",
        help="the base path to store the evaluation results",
        condition="evaluable",
    )
    argparser.add_argument(
        "--naming",
        default=...,
        help="the rule to name friend tree files for evaluation",
        condition="evaluable",
    )
    argparser.add_argument(
        "--step",
        type=converter.int_pos,
        default=sys.maxsize,
        help="the number of entries for each chunk",
    )
    argparser.add_argument(
        "--workers",
        type=converter.int_pos,
        default=1,
        help="the number of workers to run in parallel",
    )
    argparser.add_argument(
        "--clean",
        action="store_true",
        help="remove the original friend trees after merging",
    )

    def analyze(self, results: list[dict]):
        from base_class.root import Friend
        from classifier.root.kfold import merge_kfolds

        kfolds = []
        for result in results:
            predictions: list[dict] = result.get(ResultKey.predictions)
            if predictions is None:
                continue
            for prediction in predictions:
                outputs: list[dict] = prediction.get("outputs")
                if not isinstance(outputs, Iterable):
                    continue
                for output in outputs:
                    if output.get("stage") != "Evaluation":
                        continue
                    friends: dict = output.get("output")
                    if not isinstance(friends, Iterable):
                        continue
                    for friend in friends:
                        if isinstance(friend, Friend):
                            kfolds.append(friend)
                        else:
                            try:
                                kfolds.append(Friend.from_json(friend))
                            except Exception:
                                ...

        if len(kfolds) > 1:
            return [
                merge_kfolds(
                    *kfolds,
                    step=self.opts.step,
                    workers=self.opts.workers,
                    friend_name=self.opts.name,
                    dump_base_path=IO.output / self.opts.base,
                    dump_naming=self.opts.naming,
                    clean=self.opts.clean,
                )
            ]
        return []
