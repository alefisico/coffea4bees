import pickle

import fsspec
from classifier.task.state import Cascade, share_global_state


class save(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        with fsspec.open(opts[0], 'wb') as f:
            pickle.dump(share_global_state(), f)

    @classmethod
    def help(cls):
        infos = [
            f'usage: {cls.__mod_name__()} OUTPUT',
            '',
            'Save global states to file.',
            '']
        return '\n'.join(infos)


class load(Cascade):
    @classmethod
    def parse(cls, opts: list[str]):
        with fsspec.open(opts[0], 'rb') as f:
            pickle.load(f)()

    @classmethod
    def help(cls):
        infos = [
            f'usage: {cls.__mod_name__()} INPUT',
            '',
            'Load global states from file.',
            '']
        return '\n'.join(infos)
