from classifier.task import GlobalState


class MultiClass(GlobalState):
    labels: list[str] = []

    @classmethod
    def add(cls, *labels: str):
        for l in labels:
            if l not in cls.labels:
                cls.labels.append(l)

    @classmethod
    def index(cls, label: str):
        return cls.labels.index(label)

    @classmethod
    def indices(cls, *labels: str):
        return [cls.index(l) for l in labels]
