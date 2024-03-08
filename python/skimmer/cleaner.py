import re

from base_class.system.eos import EOS


class FailedSkimCleaner:
    _pattern = re.compile(
        r'picoAOD_\w{8}-\w{4}-\w{4}-\w{4}-\w{12}_\d+_\d+.root')

    def __init__(self):
        self._dirs = []

    def add_dir(self, dir: str):
        self._dirs.append(dir)

    def clean(self, confirm: bool = True):
        EOS.allow_fail = True
        to_clean: list[EOS] = []
        for d in self._dirs:
            path = EOS(d)
            if not path.is_local:
                files = path.ls()
                for f in files:
                    if self._pattern.fullmatch(f.name):
                        to_clean.append(f)
        if confirm:
            print('The following files will be removed:')
            for f in to_clean:
                print(f)
            if input('Confirm? [y/n] ') != 'y':
                return
        for f in to_clean:
            f.rm()
        self._dirs.clear()
