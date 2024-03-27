import pickle
from typing import Mapping

import fsspec
from classifier.task.state import Cascade, _share_global_state
from classifier.typetools import dict_proxy

_SETTING = "setting"


class save(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        with fsspec.open(opts[0], "wb") as f:
            pickle.dump(_share_global_state(), f)

    @classmethod
    def help(cls):
        infos = [
            f"usage: {cls.__mod_name__()} OUTPUT",
            "",
            "Save global states to file.",
            "",
        ]
        return "\n".join(infos)


class load(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        for opt in opts:
            with fsspec.open(opt, "rb") as f:
                pickle.load(f)()

    @classmethod
    def help(cls):
        infos = [
            f"usage: {cls.__mod_name__()} INPUT [INPUT ...]",
            "",
            "Load global states from files.",
            "",
        ]
        return "\n".join(infos)


class setup(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        from classifier.task import EntryPoint, parse

        for opt in opts:
            configs = parse.mapping(f"file:///{opt}")
            for k, v in configs.items():
                _, k = EntryPoint._fetch_module(k, _SETTING)
                if isinstance(v, Mapping):
                    dict_proxy(k).update(v)
                elif isinstance(v, list):
                    k.parse(v)

    @classmethod
    def help(cls):
        infos = [
            f"usage: {cls.__mod_name__()} FILE [FILE ...]",
            "",
            "Setup all settings from input files.",
            "",
        ]
        return "\n".join(infos)
