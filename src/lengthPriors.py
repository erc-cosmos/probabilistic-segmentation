"""Module for arc length priors."""
import scipy as sp
import scipy.stats
import functools
import collections


class ImpossibleCondition(ZeroDivisionError):
    """Exception raised when conditionning by an impossible event."""

    pass


class ContinuousLengthPrior:
    """Prior using a continuous distribution."""

    def __init__(self, cdf, maxLength, x):
        """Construct a continuous length prior from a cdf function."""
        if maxLength <= 0:
            raise ValueError("Positive max length required")
        self.cdf = cdf
        self.x = x
        self.maxLength = maxLength

    def __repr__(self):
        """Return a human-readable representation of a continuous prior."""
        return f"ContinuousLengthPrior, max={self.maxLength}"

    def evalCond(self, i, j):
        """
        Return the likelihood of the next boundary being at j given a boundary at i.

        i -- known boundary index
        j -- examined index
        """
        if j <= i or j >= len(self.x):
            return 0
        x0 = self.x[i]
        xmax = min(self.x[-1]-x0, self.maxLength)
        xjpred = self.x[j-1]-x0
        xj = self.x[j]-x0

        if xj > self.maxLength:  # Cap xj at max
            xj = self.maxLength
        if xjpred > self.maxLength:  # Cap xjpred at max
            xjpred = self.maxLength

        c0, cjpred, cj, cmax = self.cdf([0, xjpred, xj, xmax])

        # Scaling factor for the part of the distribution outside possible values
        scaling = cmax - c0
        if scaling == 0:
            raise ImpossibleCondition("Conditioning on impossible event")
        return (cj-cjpred)/scaling

    @property
    def dataLength(self):
        """Return the total length of the data."""
        return self.x[-1]-self.x[0]

    def getMaxIndex(self, i):
        """Return the maximum arc end with non-null prior for an arc starting at i."""
        if i < 0:
            raise IndexError("Negative indexing not supported")
        xmax = self.x[i] + self.maxLength
        imax = i + next((it for it, x in enumerate(self.x[i:]) if x > xmax), len(self.x[i:])-1)
        return imax

    def getMinIndex(self, j):
        """Return the minimum arc start with non-null prior for an arc ending at i."""
        if j < 0:
            raise IndexError("Negative indexing not supported")
        xmin = self.x[j] - self.maxLength
        imin = next((it-1 for it, x in enumerate(self.x[:j]) if x > xmin), 0)
        if imin == -1:
            imin = 0
        return imin


class NormalLengthPrior(ContinuousLengthPrior):
    """Prior using a truncated normal distribution."""

    def __init__(self, mean, stddev, x, maxLength):
        """Construct a NormalLengthPrior."""
        self.mean = mean
        self.stddev = stddev
        cdf = functools.partial(sp.stats.norm.cdf, loc=mean, scale=stddev)
        super().__init__(cdf, maxLength, x)

    def __repr__(self):
        """Return a human-readable representation of a normal prior."""
        return f"NormalLengthPrior N({self.mean},{self.stddev}), max={self.maxLength} on {len(self.x)} points"


class DiscreteLengthPrior:
    """Prior using a discrete distribution."""

    def __init__(self, dataLength, distribution, maxLength=None):
        """Construct a DiscreteLengthPrior."""
        self._dataLength = dataLength
        self._distrib = distribution
        self._maxLength = maxLength if maxLength is not None else max(self.distrib)

    def __repr__(self):
        """Return a human-readable representation of an empirical prior."""
        return f"Discrete length prior (max {self._maxLength}): {self._distrib}"

    @property
    def distrib(self):
        """Access distrib."""
        return self._distrib

    @property
    def dataLength(self):
        """Access dataLength."""
        return self._dataLength

    @property
    def maxLength(self):
        """Access maxLength."""
        return self._maxLength

    def evalCond(self, i, j):
        """
        Return the likelihood of the next boundary being at j given a boundary at i.

        i -- known boundary index
        j -- examined index
        """
        if j > self.dataLength or j < i:
            return 0
        length = j-i+1
        if self.dataLength - i > self.maxLength:
            scaling = 1
        else:
            scaling = self.scalingFactor(i)
        return self.distrib.get(length, 0)/scaling

    def getMaxIndex(self, i):
        """Return the maximum arc end with non-null prior for an arc starting at i."""
        if i < 0:
            raise IndexError("Negative indexing not supported")
        elif i >= self.dataLength:
            raise IndexError("Arc end out bounds")
        return min(i+self.maxLength-1, self.dataLength-1)

    def getMinIndex(self, j):
        """Return the minimum arc start with non-null prior for an arc ending at i."""
        if j < 0:
            raise IndexError("Negative indexing not supported")
        elif j >= self.dataLength:
            raise IndexError("Arc end out bounds")
        return max(j-self.maxLength, 0)

    def scalingFactor(self, i):
        """Return the scaling factor for truncating the underlying the distribution."""
        result = sum(self.distrib.get(k, 0) for k in range(self.dataLength-i+1))
        if result == 0:
            raise ImpossibleCondition("Conditioning on impossible event")
        return result


class EmpiricalLengthPrior(DiscreteLengthPrior):
    """Prior using an empirical distribution."""

    def __init__(self, data, dataLength, maxLength=None):
        """Construct an EmpiricalLengthPrior."""
        self._data = data
        distribution = inferDiscreteDistribution(data)
        super().__init__(dataLength, distribution, maxLength)

    def __repr__(self):
        """Return a human-readable representation of an empirical prior."""
        return super().__repr__()+f" based on {len(self._data)} samples"


def inferDiscreteDistribution(data, *, reserve=0, domain=None):
    """Infer a distribution from data."""
    reserveBoost = reserve/len(domain) if reserve != 0 else 0

    counts = collections.Counter(data)
    return {x: (1-reserve) * p / len(data) + reserveBoost for x, p in counts.items()}
