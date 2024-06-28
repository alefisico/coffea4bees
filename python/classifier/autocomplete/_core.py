import glob
import inspect
import sys
from collections import deque

from classifier.config.main.help import _walk_packages

from ..task import main as m
from ..task.special import TaskBase


def _subcomplete(cls: TaskBase, args: list[str]):
    last = args[-1] if args else ""
    yield from (i for i in m.EntryPoint._preserved if i.startswith(last))
    if cls is not None and cls.autocomplete is not NotImplemented:
        yield from cls.autocomplete(args)
    else:
        yield from glob.glob(f"{last}*")


def autocomplete():
    args = deque(sys.argv[2:])  # 0 is "_core.py", 1 is "./run_classifier.py"
    main = args.popleft() if args else ""
    if len(args) == 0:
        for part in m.EntryPoint._tasks + [m._FROM, m._TEMPLATE]:
            if part.startswith(main):
                yield part
        return
    subargs = m.EntryPoint._fetch_subargs(args)
    if len(args) == 0:
        yield from _subcomplete(
            m.EntryPoint._fetch_module(f"{main}.Main", m._MAIN)[1], subargs
        )
    while len(args) > 0:
        cat = args.popleft().removeprefix("--")
        mod = args.popleft() if args else ""
        if len(args) == 0:
            if cat in m.EntryPoint._keys:
                target = m.EntryPoint._keys[cat]
                for imp in _walk_packages(f"{m._CLASSIFIER}/{m._CONFIG}/{cat}/"):
                    if imp:
                        imp = f"{imp}."
                    _imp = f"{imp}*"
                    modname, _ = m.EntryPoint._fetch_module_name(_imp, cat)
                    _mod, _ = m.EntryPoint._fetch_module(_imp, cat)
                    if _mod is not None:
                        for name, obj in inspect.getmembers(_mod, inspect.isclass):
                            if (
                                issubclass(obj, target)
                                and not m._is_private(name)
                                and obj.__module__.startswith(modname)
                                and (clsname := f"{imp}{name}").startswith(mod)
                            ):
                                yield clsname
                return
        subargs = m.EntryPoint._fetch_subargs(args)
        if len(args) == 0:
            if cat in m.EntryPoint._keys:
                yield from _subcomplete(
                    m.EntryPoint._fetch_module(mod, cat)[1], subargs
                )
            else:
                # TODO deal with "--from" and "--template"
                yield from _subcomplete(None, subargs)
            return
    yield from ()


if __name__ == "__main__":
    print(" ".join(autocomplete()))
