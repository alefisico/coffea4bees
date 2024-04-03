from classifier.task import Main as _Main


class Main(_Main):
    _standalone = True

    def run(self, _): ...
