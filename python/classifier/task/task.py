from __future__ import annotations

import argparse
from collections import ChainMap
from copy import deepcopy
from textwrap import indent
from typing import Iterable, Literal

from .special import TaskBase

_INDENT = "  "
_DASH = "-"


class _Formatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
): ...


class ArgParser(argparse.ArgumentParser):
    def __init__(
        self, workflow: Iterable[tuple[Literal["main", "sub"], str]] = None, **kwargs
    ):
        kwargs["prog"] = kwargs.get("prog", None) or ""
        kwargs["add_help"] = False
        kwargs["conflict_handler"] = "resolve"
        kwargs["formatter_class"] = _Formatter
        if workflow:
            wf = ["workflow:"]
            wf.extend(
                indent(
                    indent(f"->([yellow]{p} process[/yellow])\n{t}", _INDENT)[2:],
                    _INDENT,
                )
                for p, t in workflow
            )
            wf = "\n".join(wf)
            if "description" in kwargs:
                kwargs["description"] += "\n\n" + wf
            else:
                kwargs["description"] = wf
        super().__init__(**kwargs)

    def remove_argument(self, *name_or_flags: str):
        self.add_argument(
            *name_or_flags,
            nargs=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
            default=argparse.SUPPRESS,
        )


class Task(TaskBase):
    argparser: ArgParser = NotImplemented
    defaults: dict[str] = NotImplemented

    @classmethod
    def __mod_name__(cls):
        return ".".join(f"{cls.__module__}.{cls.__name__}".split(".")[3:])

    def __init_subclass__(cls):
        defaults, parents, kwargs = [], [], {}
        for base in cls.__bases__:
            if issubclass(base, Task):
                if base.argparser is not NotImplemented:
                    parents.append(deepcopy(base.argparser))
                if base.defaults is not NotImplemented:
                    defaults.append(base.defaults)
        if "argparser" in vars(cls):
            parents.append(cls.argparser)
            kwargs["prog"] = cls.argparser.prog or cls.__mod_name__()
            for k in ["usage", "description", "epilog"]:
                kwargs[k] = getattr(cls.argparser, k, None)
        else:
            kwargs["prog"] = cls.__mod_name__()
        if "defaults" in vars(cls):
            defaults.append(cls.defaults)
        cls.argparser = ArgParser(**kwargs, parents=parents)
        cls.defaults = (
            NotImplemented if len(defaults) == 0 else dict(ChainMap(*defaults[::-1]))
        )

    def parse(self, args: list[str]):
        self.opts, _ = self.argparser.parse_known_args(args)
        if self.defaults is not NotImplemented:
            for k, v in self.defaults.items():
                if not hasattr(self.opts, k):
                    setattr(self.opts, k, v)
        return self

    def flag(self, opt: str):
        return hasattr(self.opts, opt) and getattr(self.opts, opt)

    @classmethod
    def help(cls):
        if cls.argparser is NotImplemented:
            raise ValueError(f"{cls.__name__}.argparser is not implemented")
        return cls.argparser.format_help()

    @classmethod
    def autocomplete(cls, args: list[str]):
        last = args[-1] if args else _DASH
        if last.startswith(_DASH):
            for action in cls.argparser._actions:
                yield from filter(lambda x: x.startswith(last), action.option_strings)
