from . import Schedule


def _noop(*_, **__): ...


class _Dummy:
    def __getattr__(self, _):
        return _noop


class Skim(Schedule):
    epoch = 1
