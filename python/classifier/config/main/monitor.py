from classifier.monitor.usage import Usage
from classifier.process.monitor import Monitor
from classifier.task import Main as _Main


class Main(_Main):
    def run(self, _):
        Usage().stop()
        try:
            Monitor.current()._listener[1].join()
        except KeyboardInterrupt:
            pass
