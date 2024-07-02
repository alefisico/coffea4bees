import inspect
import sys
from collections import deque

from classifier.config.main.help import _walk_packages

from .. import main as m
from ..special import TaskBase

_PATHCODE = 255


def _subcomplete(cls: type[TaskBase], args: list[str]):
    last = args[-1] if args else ""
    yield from (i for i in m.EntryPoint._preserved if i.startswith(last))
    if (
        isinstance(cls, type)
        and issubclass(cls, TaskBase)
        and cls.autocomplete is not NotImplemented
    ):
        yield from cls.autocomplete(args)


def _special(cat: str, args: list[str]):
    if not args:
        return
    last = args[-1]
    if last.startswith("-"):
        yield from _subcomplete(None, [last])
        return
    match cat:
        case m._FROM:
            sys.exit(_PATHCODE)
        case m._TEMPLATE:
            if len(args) > 1:
                sys.exit(_PATHCODE)


def autocomplete():
    args = deque(sys.argv[2:])  # 0 is "_core.py", 1 is "./pyml.py"
    main = args.popleft() if args else ""
    if len(args) == 0:
        for part in m.EntryPoint._tasks + [m._FROM, m._TEMPLATE]:
            if part.startswith(main):
                yield part
        return
    subargs = m.EntryPoint._fetch_subargs(args)
    if len(args) == 0:
        if main in m.EntryPoint._tasks:
            yield from _subcomplete(
                m.EntryPoint._fetch_module(f"{main}.Main", m._MAIN)[1], subargs
            )
        else:
            yield from _special(main, subargs)
        return
    while len(args) > 0:
        cat = args.popleft().removeprefix(m._DASH)
        mod = args.popleft() if args else None
        if len(args) == 0:
            if cat in m.EntryPoint._keys:
                mod = mod or ""
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
                    m.EntryPoint._fetch_module(mod or "", cat)[1], subargs
                )
            else:
                if mod is not None:
                    subargs.insert(0, mod)
                yield from _special(cat, subargs)
            return


if __name__ == "__main__":
    print("\n".join(autocomplete()))
