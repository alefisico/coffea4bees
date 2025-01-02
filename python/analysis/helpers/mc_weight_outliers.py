import numpy as np
import numpy.typing as npt


class OutlierByMedian:
    """
    - median is less sensitive to chunk size and outlier's value
    - threshold of median can be interpreted as the number of events that an outlier is equivalent to
    """

    def __init__(self, threshold: float = 200, median: float = None):
        self.threshold = threshold
        self.median = median

        self._last_median = None

    def __call__(self, weights: npt.ArrayLike):
        median = self.median
        if self.median is None:
            median = np.median(weights)
        self._last_median = median
        return weights < (median * self.threshold)

    @property
    def last_median(self):
        return self._last_median
