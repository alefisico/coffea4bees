from classifier.task import GlobalState


class MultiClass(GlobalState):
    labels: list[str] = []

    @classmethod
    def add(cls, *labels: str):
        for label in labels:
            if label not in cls.labels:
                cls.labels.append(label)

    @classmethod
    def index(cls, label: str):
        return cls.labels.index(label)

    @classmethod
    def indices(cls, *labels: str):
        return [cls.index(label) for label in labels]
