"""Re-exports random.Random and random.SystemRandom and defines a few more Random subclasses with different random origins."""
from random import Random as Random_, SystemRandom as SystemRandom_
import secrets
import math
import os

# Internal imports

# Standard typing imports for aps
import typing_extensions as _te
import collections.abc as _a
import typing as _ty

if _ty.TYPE_CHECKING:
    import _typeshed as _tsh
import types as _ts

__all__ = ["Random", "SystemRandom", "WeightedFunctions"]  # , "OSRandom", "SecretsRandom"


class _SupportsLenAndGetItem(_ty.Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _ty.Any: ...


class RandomPlusCauchy:
    def cauchyvariate(self, median: float, scale: float) -> float:
        """
        Generate a random number based on the Cauchy distribution with specified median and scale.

        :param median: Median of the Cauchy distribution.
        :param scale: Scale parameter of the Cauchy distribution.
        :return: Random number from the Cauchy distribution.
        """
        u = self.random()
        return median + scale * math.tan(math.pi * (u - 0.5))


class Random(Random_, RandomPlusCauchy):
    ...


class SystemRandom(SystemRandom_, RandomPlusCauchy):
    ...


# class OSRandom(Random):
#     def random(self) -> float:
#         """Return a random float from os.urandom"""
#         return int.from_bytes(os.urandom(7), "big") / (1 << 56)


# class SecretsRandom(Random):
#     def random(self) -> float:
#         """Return a random float from secrets.randbits"""
#         return secrets.randbits(56) / (1 << 56)


class WeightedFunctions:
    TRUNCATE_MAX_TRIES: int = 100_000

    def __init__(self, generator: Random):
        self.g: Random = generator

    @staticmethod
    def _is01(u: float, lower_bound: float | int, upper_bound: float | int) -> float | None:
        """
        Scale the random value to within the given bounds.

        :param u: The initial random value (between 0 and 1).
        :param lower_bound: The lower bound of the output range.
        :param upper_bound: The upper bound of the output range.
        :return: Transformed and scaled value.
        """
        if not 0.0 <= u <= 1.0:
            raise ValueError("Expected u in [0,1]")
        return float(lower_bound + (upper_bound - lower_bound) * u)

    @staticmethod
    @_te.deprecated("We do not use _squash01 any longer")
    def _squash01(u: float) -> float:
        return min(max(u, 0.0), 1.0)

    @staticmethod
    def _truncated01(draw: _a.Callable[[], float], max_tries: int = 100_000) -> float:
        for _ in range(max_tries):
            x = draw()
            if 0.0 <= x <= 1.0:
                return x
        raise RuntimeError("Too many rejections; parameters put almost no mass in [0,1].")

    @staticmethod
    def _logistic01(x: float) -> float:
        # stable logistic
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _check_bounds(lower_bound: float, upper_bound: float) -> None:
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must be > lower_bound")

    @staticmethod
    def fold(u: float, by: float = 0.5) -> float:
        return abs(u - by)

    @staticmethod
    def shift(u: float, by: float = 0.5) -> float:
        u = u - by
        if u < 0:
            return u + 1.0
        return u

    @staticmethod
    def reflect(u: float) -> float:
        return 1.0 - u

    @staticmethod  # TODO: Fix
    def invert(u: float) -> float:
        raise NotImplementedError("I have yet to be able to invert the result of a distribution, and I think it is "
                                  "indeed not possible.")
        return 1.0 / u

    @staticmethod
    def raise_(u: float, by: float = 1.0) -> float:
        return u + by

    @staticmethod
    def floor(x: _a.Callable[[], float], at: float = 0.5) -> _a.Callable[[], float]:
        def _w() -> float:
            while u := x() < at: ...
            return u
        return _w

    @staticmethod
    def ceil(x: _a.Callable[[], float], at: float = 0.5) -> _a.Callable[[], float]:
        def _w() -> float:
            while u := x() > at: ...
            return u
        return _w

    def linear(self, slope: float = 1.0, intercept: float = 0.0, lower_bound: float | int = 0,
               upper_bound: float | int = 1) -> float:
        """
        u = slope*R + intercept, truncated to [0,1]
        Safe always-in-range params: slope in [0,1] and intercept in [0, 1-slope]

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :param slope: Slope of the linear transformation.
        :param intercept: Intercept of the linear transformation.
        :return: Scaled random number from the linear transformation.
        """
        self._check_bounds(lower_bound, upper_bound)
        u = self._truncated01(lambda: slope * self.g.random() + intercept, max_tries=self.TRUNCATE_MAX_TRIES)
        return self._is01(u, lower_bound, upper_bound)

    def power(self, exponent: float = 2.0, lower_bound: float | int = 0, upper_bound: float | int = 1):
        """
        Power distribution on [0,1] with shape a>0.
        Sample: U = R^(1/a)
        - a>1 pushes mass toward 1
        - 0<a<1 pushes mass toward 0

        :param exponent:
        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        """
        if exponent <= 0:
            raise ValueError("a must be > 0")
        self._check_bounds(lower_bound, upper_bound)
        u = self.g.random() ** (1.0 / exponent)
        return self._is01(u, lower_bound, upper_bound)

    def quadratic(self, lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a quadratic distribution and scale it within the specified bounds.

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the quadratic distribution.
        """
        return self.power(3.0, lower_bound, upper_bound)

    def cubic(self, lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a cubic distribution and scale it within the specified bounds.

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the cubic distribution.
        """
        return self.power(4.0, lower_bound, upper_bound)

    def quartic(self, lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a quartic distribution and scale it within the specified bounds.

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the quartic distribution.
        """
        return self.power(5.0, lower_bound, upper_bound)

    def gaussian(self, mu: float | int = 0.5, sigma: float | int = 0.15, lower_bound: float | int = 0,
                 upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on the normal (Gaussian) distribution and scale it within the specified bounds.

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :param mu: Mean of the distribution.
        :param sigma: Standard deviation of the distribution.
        :return: Scaled random number from the normal distribution.
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self._check_bounds(lower_bound, upper_bound)
        u = self._truncated01(lambda: self.g.gauss(mu, sigma))
        return self._is01(u, lower_bound, upper_bound)

    def exponential(self, lambd: float = 2.5, lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        Truncated Exponential(rate=lambd) on [0,1].
        For lambd>0: more mass near 0. For lambd<0: more mass near 1 (increasing density).

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :param lambd: Lambda parameter (1/mean) of the distribution.
        :return: Scaled random number from the exponential distribution.
        """
        if lambd == 0.0:
            return self.g.uniform(lower_bound, upper_bound)
        self._check_bounds(lower_bound, upper_bound)
        r = self.g.random()
        if lambd > 0:
            # F(x) = (1-exp(-l x)) / (1-exp(-l))
            u = -math.log(1.0 - r * (1.0 - math.exp(-lambd))) / lambd
        else:
            # increasing density: use |l| but flip via x -> 1-x
            lp = -lambd
            u0 = -math.log(1.0 - r * (1.0 - math.exp(-lp))) / lp
            u = 1.0 - u0
        return self._is01(u, lower_bound, upper_bound)

    def beta_mean_kappa(self, mean: float = 0.35, kappa: float = 6.0, lower_bound: float | int = 0,
                        upper_bound: float | int = 1) -> float:
        """
        Beta distribution on [0,1] parameterized by mean and concentration (kappa).
        alpha = mean*kappa, beta = (1-mean)*kappa

        :param mean: in (0,1): where mass centers (bias)
        :param kappa: > 0: concentration (bigger = tighter around mean) kappa≈2 is very broad, kappa≈20 is quite tight.
        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the exponential distribution.
        """
        self._check_bounds(float(lower_bound), float(upper_bound))
        if not (0.0 < mean < 1.0):
            raise ValueError("mean must be in (0,1)")
        if kappa <= 0.0:
            raise ValueError("kappa must be > 0")

        a = mean * kappa
        b = (1.0 - mean) * kappa
        u = self.g.betavariate(a, b)  # u in [0,1]
        return self._is01(u, lower_bound, upper_bound)

    def arcsine(self, lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a sinusoidal distribution and scale it within the specified bounds.

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the sinusoidal distribution.
        """
        self._check_bounds(lower_bound, upper_bound)
        u = self.g.betavariate(0.5, 0.5)
        return self._is01(u, lower_bound, upper_bound)

    def triangular(self, mode: float = 0.5, lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a triangular distribution and scale it within the specified bounds.

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :param mode: The value where the peak of the distribution occurs.
        :return: Scaled random number from the triangular distribution.
        """
        if not (0.0 <= mode <= 1.0):
            raise ValueError("mode must be in [0,1]")
        self._check_bounds(lower_bound, upper_bound)
        # random.triangular expects (low, high, mode) in absolute units
        x = self.g.triangular(0.0, 1.0, mode)
        return self._is01(x, lower_bound, upper_bound)

    def beta(self, alpha: float = 2.0, beta: float = 5.0, lower_bound: float | int = 0,
             upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a beta distribution and scale it within the specified bounds.

        :param alpha: Alpha parameter of the beta distribution.
        :param beta: Beta parameter of the beta distribution.
        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the beta distribution.
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha,beta must be > 0")
        self._check_bounds(lower_bound, upper_bound)
        u = self.g.betavariate(alpha, beta)
        return self._is01(u, lower_bound, upper_bound)

    def logit_normal(self, mean: float = 0.0, sigma: float = 1.0, lower_bound: float | int = 0,
                     upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a log-normal distribution and scale it within the specified bounds.

        :param mean: Mean of the underlying normal distribution.
        :param sigma: Standard deviation of the underlying normal distribution.
        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the log-normal distribution.
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        self._check_bounds(lower_bound, upper_bound)
        u = self._logistic01(self.g.gauss(mean, sigma))  # u in (0,1) always
        return self._is01(u, lower_bound, upper_bound)

    def trapezoidal(self, a: float = 0.0, b: float = 0.3, c: float = 0.7, d: float = 1.0, lower_bound: float | int = 0,
                    upper_bound: float | int = 1) -> float:
        """
        Trapezoidal distribution on [0,1] with parameters a<=b<=c<=d. Is linear up on [a,b], flat on [b,c], linear down on [c,d].

        :param a:
        :param b:
        :param c:
        :param d:
        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number
        """
        self._check_bounds(float(lower_bound), float(upper_bound))
        a, b, c, d = float(a), float(b), float(c), float(d)
        if not (0.0 <= a <= b <= c <= d <= 1.0):
            raise ValueError("Require 0 <= a <= b <= c <= d <= 1")

        # Handle degenerate cases:
        if a == d:
            return self._is01(a, float(lower_bound), float(upper_bound))
        if b == a and c == d:
            # uniform on [a,d]
            u = a + (d - a) * self.g.random()
            return self._is01(u, float(lower_bound), float(upper_bound))

        # Let h be plateau height, determined by normalization:
        # Area = 1 = 0.5*h*(b-a) + h*(c-b) + 0.5*h*(d-c)
        denom = (c - b) + 0.5 * (b - a) + 0.5 * (d - c)
        h = 1.0 / max(denom, 1e-15)

        A_rise = 0.5 * h * (b - a)
        A_flat = h * (c - b)
        A_fall = 0.5 * h * (d - c)

        r = self.g.random()

        if r < A_rise:
            # invert CDF on rising edge:
            # r = (h/(2*(b-a))) * (x-a)^2
            return self._is01(a + math.sqrt(2.0 * (b - a) * r / h), float(lower_bound), float(upper_bound))

        r -= A_rise
        if r < A_flat:
            # flat region:
            return self._is01(b + r / h, float(lower_bound), float(upper_bound))

        r -= A_flat
        # falling edge:
        # r = A_fall - (h/(2*(d-c))) * (d-x)^2  (equivalently solve for (d-x))
        return self._is01(d - math.sqrt(2.0 * (d - c) * (A_fall - r) / h), float(lower_bound), float(upper_bound))

    def weibull(self, k: float = 1.5, lam: float = 1.0, lower_bound: float | int = 0,
                upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a Weibull distribution and scale it within the specified bounds.

        :param k: Shape parameter of the Weibull distribution.
        :param lam: Scale parameter of the Weibull distribution.
        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the Weibull distribution.
        """
        if k <= 0 or lam <= 0:
            raise ValueError("k, lam must be > 0")
        self._check_bounds(lower_bound, upper_bound)

        r = self.g.random()
        # base CDF: 1-exp(-(x/lam)^k). Truncate at 1:
        z = 1.0 - math.exp(-((1.0 / lam) ** k))
        u = lam * (-math.log(1.0 - r * z)) ** (1.0 / k)
        return self._is01(u, lower_bound, upper_bound)

    def gamma(self, alpha: float = 2.0, beta: float = 1.0, lower_bound: float | int = 0,
              upper_bound: float | int = 1) -> float:
        """
        Truncated Gamma(shape=alpha, scale=beta) on [0,1]. Uses rejection sampling (stdlib has no gamma CDF/invcdf).

        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :param alpha: Shape parameter of the Gamma distribution.
        :param beta: Scale parameter of the Gamma distribution.
        :return: Scaled random number from the Gamma distribution.
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha,beta must be > 0")
        u = self._truncated01(lambda: self.g.gammavariate(alpha, beta), self.TRUNCATE_MAX_TRIES)
        return self._is01(u, lower_bound, upper_bound)

    def cauchy(self, x0: float = 0.5, gamma: float = 0.1, lower_bound: float | int = 0,
               upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a Cauchy distribution and scale it within the specified bounds.

        :param x0: Median of the distribution.
        :param gamma: Scale parameter of the distribution.
        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the Cauchy distribution.
        """
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self._check_bounds(lower_bound, upper_bound)
        u = self._truncated01(lambda: self.g.cauchyvariate(x0, gamma), self.TRUNCATE_MAX_TRIES)
        return self._is01(u, lower_bound, upper_bound)

    def pareto(self, xm: float = 0.2, alpha: float = 2.0, lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        Generate a random number based on a Pareto distribution and scale it within the specified bounds.

        :param xm:
        :param alpha: Shape parameter of the Pareto distribution.
        :param lower_bound: Lower bound of the scaled range.
        :param upper_bound: Upper bound of the scaled range.
        :return: Scaled random number from the Pareto distribution.
        """
        if not (0.0 < xm < 1.0):
            raise ValueError("xm must be in (0,1) for truncation to [0,1]")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self._check_bounds(lower_bound, upper_bound)

        r = self.g.random()
        z = 1.0 - (xm ** alpha)  # 1 - (xm/1)^alpha
        u = xm / ((1.0 - r * z) ** (1.0 / alpha))  # u in [xm, 1]
        return self._is01(u, lower_bound, upper_bound)

    def u_quadratic(self, lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        U-quadratic distribution on [0,1].
        PDF: f(x) = 6x^2 - 6x + 3
        CDF: F(x) = 2x^3 - 3x^2 + 3x
        Sample via numerical inversion (bisection), stable and exact enough for testing.
        """
        r = self.g.random()

        def F(x: float) -> float:
            return x * x * x - 1.5 * x * x + 1.5 * x

        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if F(mid) < r:
                lo = mid
            else:
                hi = mid

        u = 0.5 * (lo + hi)
        return self._is01(u, lower_bound, upper_bound)

    def kumaraswamy(self, a: float = 2.0, b: float = 5.0,
                    lower_bound: float | int = 0, upper_bound: float | int = 1) -> float:
        """
        Kumaraswamy(a,b) on [0,1], a>0, b>0.
        Inverse CDF sampling (closed form).
        """
        if a <= 0.0 or b <= 0.0:
            raise ValueError("a and b must be > 0")

        u = self.g.random()
        x = (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)
        return self._is01(x, lower_bound, upper_bound)
