from classifier.process.state import GlobalState


class MultiClass(GlobalState):
    labels: list[str] = []

    @classmethod
    def add(cls, *label: str):
        for l in label:
            if l not in cls.labels:
                cls.labels.append(l)

    @classmethod
    def index(cls, label: str):
        return cls.labels.index(label)
