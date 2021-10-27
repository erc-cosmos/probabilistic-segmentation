"""Module for arc length priors."""
import collections
import functools
from typing import Mapping

from scipy.stats import norm  # type:ignore


class ImpossibleCondition(ZeroDivisionError):
    """Exception raised when conditionning by an impossible event."""

    pass


class ContinuousLengthPrior:
    """Prior using a continuous distribution."""

    def __init__(self, cdf, max_length, x):
        """Construct a continuous length prior from a cdf function."""
        if max_length <= 0:
            raise ValueError("Positive max length required")
        self.cdf = cdf
        self.x = x
        self.max_length = max_length

    def __repr__(self):
        """Return a human-readable representation of a continuous prior."""
        return f"ContinuousLengthPrior, max={self.max_length}"

    def eval_cond(self, i, j):
        """
        Return the likelihood of the next boundary being at j given a boundary at i.

        i -- known boundary index
        j -- examined index
        """
        if j <= i or j >= len(self.x):
            return 0
        x0 = self.x[i]
        xmax = min(self.x[-1]-x0, self.max_length)
        xjpred = self.x[j-1]-x0
        xj = self.x[j]-x0

        if xj > self.max_length:  # Cap xj at max
            xj = self.max_length
        if xjpred > self.max_length:  # Cap xjpred at max
            xjpred = self.max_length

        c0, cjpred, cj, cmax = self.cdf([0, xjpred, xj, xmax])

        # Scaling factor for the part of the distribution outside possible values
        # scaling = cmax - c0
        scaling = 1
        if scaling == 0:
            raise ImpossibleCondition("Conditioning on impossible event")
        return (cj-cjpred)/scaling

    @property
    def data_length(self):
        """Return the total length of the data."""
        return self.x[-1]-self.x[0]

    def get_max_index(self, i):
        """Return the maximum arc end with non-null prior for an arc starting at i."""
        if i < 0:
            raise IndexError("Negative indexing not supported")
        xmax = self.x[i] + self.max_length
        imax = i + next((it for it, x in enumerate(self.x[i:]) if x > xmax), len(self.x[i:])-1)
        return imax

    def get_min_index(self, j):
        """Return the minimum arc start with non-null prior for an arc ending at i."""
        if j < 0:
            raise IndexError("Negative indexing not supported")
        xmin = self.x[j] - self.max_length
        imin = next((it-1 for it, x in enumerate(self.x[:j]) if x > xmin), 0)
        if imin == -1:
            imin = 0
        return imin


class NormalLengthPrior(ContinuousLengthPrior):
    """Prior using a truncated normal distribution."""

    def __init__(self, mean, stddev, x, max_length):
        """Construct a NormalLengthPrior."""
        self.mean = mean
        self.stddev = stddev
        cdf = functools.partial(norm.cdf, loc=mean, scale=stddev)
        super().__init__(cdf, max_length, x)

    def __repr__(self):
        """Return a human-readable representation of a normal prior."""
        return f"NormalLengthPrior N({self.mean},{self.stddev}), max={self.max_length} on {len(self.x)} points"


class DiscreteLengthPrior:
    """Prior using a discrete distribution."""

    def __init__(self, data_length: int, distribution: Mapping[int, float], max_length: int = None):
        """Construct a DiscreteLengthPrior."""
        self._dataLength = data_length
        self._distrib = distribution
        self._max_length = max_length if max_length is not None else max(self.distrib)

    def __repr__(self):
        """Return a human-readable representation of an empirical prior."""
        return f"Discrete length prior (max {self._max_length}): {self._distrib}"

    @property
    def distrib(self):
        """Access distrib."""
        return self._distrib

    @property
    def data_length(self):
        """Access dataLength."""
        return self._dataLength

    @property
    def max_length(self) -> int:
        """Access maxLength."""
        return self._max_length

    def eval_cond(self, i, j) -> float:
        """
        Return the likelihood of the next boundary being at j given a boundary at i.

        i -- known boundary index
        j -- examined index
        """
        if j > self.data_length or j < i:
            return 0
        length = j-i+1
        if self.data_length - i > self.max_length:
            scaling = 1
        else:
            scaling = self.scaling_factor(i)
        return self.distrib.get(length, 0)/scaling

    def get_max_index(self, i):
        """Return the maximum arc end with non-null prior for an arc starting at i."""
        if i < 0:
            raise IndexError("Negative indexing not supported")
        elif i >= self.data_length:
            raise IndexError("Arc end out bounds")
        return min(i+self.max_length-1, self.data_length-1)

    def get_min_index(self, j):
        """Return the minimum arc start with non-null prior for an arc ending at i."""
        if j < 0:
            raise IndexError("Negative indexing not supported")
        elif j >= self.data_length:
            raise IndexError("Arc end out bounds")
        return max(j-self.max_length, 0)

    def scaling_factor(self, i):
        """Return the scaling factor for truncating the underlying the distribution."""
        return 1
        result = sum(self.distrib.get(k, 0) for k in range(self.dataLength-i+1))
        if result == 0:
            raise ImpossibleCondition("Conditioning on impossible event")
        return result


class EmpiricalLengthPrior(DiscreteLengthPrior):
    """Prior using an empirical distribution."""

    def __init__(self, data, data_length, max_length=None):
        """Construct an EmpiricalLengthPrior."""
        self._data = data
        distribution = infer_discrete_distribution(data)
        super().__init__(data_length, distribution, max_length)

    def __repr__(self):
        """Return a human-readable representation of an empirical prior."""
        return super().__repr__()+f" based on {len(self._data)} samples"


class GeometricLengthPrior(DiscreteLengthPrior):
    """Prior using a truncated geometric distribution."""

    def __init__(self, data_length, ratio, max_length, min_length=1):
        """Construct a GeometricLengthPrior."""
        distribution = make_geo_dist(min_length, max_length, ratio)
        super().__init__(data_length, distribution, max_length=max_length)


def make_geo_dist(min_len, max_len, ratio):
    """Return a truncated geometric distribution."""
    unscaled = [ratio**i for i in range(max_len-min_len)]
    factor = sum(unscaled)
    return {i: p/factor for i, p in zip(range(min_len, max_len), unscaled)}


def infer_discrete_distribution(data, *, reserve=0, domain=None):
    """Infer a distribution from data."""
    reserve_boost = reserve/len(domain) if reserve != 0 else 0

    counts = collections.Counter(data)
    return {x: (1-reserve) * p / len(data) + reserve_boost for x, p in counts.items()}
