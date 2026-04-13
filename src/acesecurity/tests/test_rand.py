from ..rand import Random, SystemRandom, WeightedFunctions  # , OSRandom, SecretsRandom
from statistics import mean, pvariance
from functools import lru_cache
import collections.abc as _a
import typing as _ty
import bisect
import math

__all__ = ["test_random_generators", "test_weighted_functions"]


def integrate(y_vals, h):
    i = 1
    total = y_vals[0] + y_vals[-1]
    for y in y_vals[1:-1]:
        if i % 2 == 0:
            total += 2 * y
        else:
            total += 4 * y
        i += 1
    return total * (h / 3.0)


def linspace(start: float, stop: float, num: int):
        if num <= 0:
            return []
        if num == 1:
            return [float(start)]

        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]


# TODO: Use Pytest to make a bunch of different test from this one test
def test_random_generators(samples: int = 100_000, bins: int = 20):
    generators = [
        ("Random", Random()),
        ("SystemRandom", SystemRandom()),
        # ("OSRandom", OSRandom()),
        # ("SecretsRandom", SecretsRandom()),
    ]
    for name, gen in generators:
        data = [gen.random() for _ in range(samples)]

        # Range check
        in_range = all(0.0 <= x < 1.0 for x in data)

        # Mean and variance
        m = mean(data)
        v = pvariance(data)

        # Histogram
        counts = [0] * bins
        for x in data:
            i = min(bins - 1, int(x * bins))
            counts[i] += 1

        expected = samples / bins
        chi2 = sum((c - expected) ** 2 / expected for c in counts)

        print(f"\n{name}")
        print("  range ok:", in_range)
        print("  mean:", m)
        print("  variance:", v)
        print("  chi²:", chi2)

        # expected theoretical values
        print("  mean error:", abs(m - 0.5))
        print("  var error:", abs(v - 1/12))


# AI-Generated
class PDFs:
    @staticmethod
    def _phi(z: float) -> float:
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    @staticmethod
    def _Phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    @staticmethod
    def _beta_fn(a: float, b: float) -> float:
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    @staticmethod
    def _logit(x: float) -> float:
        return math.log(x / (1.0 - x))
    @staticmethod
    def pdf_uniform01(x: float) -> float:
        return 1.0 if 0.0 <= x <= 1.0 else 0.0
    @staticmethod
    def linear_trunc01(x: float, slope: float, intercept: float) -> float:
        # x = slope*r + intercept, r ~ U(0,1), then truncated to [0,1]
        # This is a truncated-uniform-on-an-interval density.
        if not (0.0 <= x <= 1.0):
            return 0.0
        if slope == 0.0:
            # point mass at intercept if in [0,1] (not representable as curve)
            return float("inf") if x == intercept else 0.0
        # support interval of slope*r+intercept:
        lo = min(intercept, intercept + slope)
        hi = max(intercept, intercept + slope)
        # truncated to [0,1]:
        a = max(0.0, lo)
        b = min(1.0, hi)
        if a >= b:
            return 0.0
        if a <= x <= b:
            # uniform on [a,b]
            return 1.0 / (b - a)
        return 0.0
    @staticmethod
    def power01(x: float, a: float) -> float:
        if a <= 0:
            return float("nan")
        return a * (x ** (a - 1.0)) if 0.0 <= x <= 1.0 else 0.0
    @classmethod
    def beta01(cls, x: float, alpha: float, beta: float) -> float:
        if alpha <= 0 or beta <= 0:
            return float("nan")
        if not (0.0 <= x <= 1.0):
            return 0.0
        return (x ** (alpha - 1.0)) * ((1.0 - x) ** (beta - 1.0)) / cls._beta_fn(alpha, beta)
    @classmethod
    def beta_mean_kappa01(cls, x: float, mean: float, kappa: float) -> float:
        if not (0.0 < mean < 1.0) or kappa <= 0:
            return float("nan")
        return cls.beta01(x, mean * kappa, (1.0 - mean) * kappa)
    @staticmethod
    def triangular01(x: float, mode: float) -> float:
        if not (0.0 <= mode <= 1.0):
            return float("nan")
        if not (0.0 <= x <= 1.0):
            return 0.0
        if mode == 0.0:
            return 2.0 * (1.0 - x)
        if mode == 1.0:
            return 2.0 * x
        if x < mode:
            return 2.0 * x / mode
        return 2.0 * (1.0 - x) / (1.0 - mode)
    @staticmethod
    def trapezoidal01(x: float, a: float, b: float, c: float, d: float) -> float:
        if not (0.0 <= a <= b <= c <= d <= 1.0):
            return float("nan")
        if not (0.0 <= x <= 1.0):
            return 0.0
        if x < a or x > d:
            return 0.0
        denom = (c - b) + 0.5 * (b - a) + 0.5 * (d - c)
        if denom <= 0:
            # degenerate: uniform on [a,d] or point mass
            if a == d:
                return float("inf") if x == a else 0.0
            return (1.0 / (d - a)) if (a <= x <= d) else 0.0
        h = 1.0 / denom
        if x <= b:
            return 0.0 if b == a else h * (x - a) / (b - a)
        if x <= c:
            return h
        return 0.0 if d == c else h * (d - x) / (d - c)
    @classmethod
    def trunc_normal01(cls, x: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return float("nan")
        if not (0.0 <= x <= 1.0):
            return 0.0
        z0 = (0.0 - mu) / sigma
        z1 = (1.0 - mu) / sigma
        Z = max(1e-15, cls._Phi(z1) - cls._Phi(z0))
        return (cls._phi((x - mu) / sigma) / sigma) / Z
    @staticmethod
    def trunc_exponential01(x: float, lambd: float) -> float:
        if not (0.0 <= x <= 1.0):
            return 0.0
        if lambd == 0.0:
            return 1.0
        if lambd > 0:
            Z = max(1e-15, 1.0 - math.exp(-lambd))
            return (lambd * math.exp(-lambd * x)) / Z
        # lambd < 0: increasing via mirror
        lp = -lambd
        Z = max(1e-15, 1.0 - math.exp(-lp))
        return (lp * math.exp(-lp * (1.0 - x))) / Z
    @staticmethod
    def trunc_weibull01(x: float, k: float, lam: float) -> float:
        if k <= 0 or lam <= 0:
            return float("nan")
        if not (0.0 <= x <= 1.0):
            return 0.0
        base = (k / lam) * ((x / lam) ** (k - 1.0)) * math.exp(-((x / lam) ** k)) if x >= 0 else 0.0
        Z = max(1e-15, 1.0 - math.exp(-((1.0 / lam) ** k)))
        return base / Z
    @classmethod
    def logit_normal01(cls, x: float, mean: float, sigma: float) -> float:
        if sigma <= 0:
            return float("nan")
        if not (0.0 < x < 1.0):
            return 0.0
        t = cls._logit(x)
        return (1.0 / (sigma * math.sqrt(2.0 * math.pi) * x * (1.0 - x))) * math.exp(
            -((t - mean) ** 2) / (2.0 * sigma * sigma))
    @staticmethod
    def arcsine01(x: float) -> float:
        if not (0.0 < x < 1.0):
            return 0.0
        return 1.0 / (math.pi * math.sqrt(x * (1.0 - x)))
    @staticmethod
    def trunc_cauchy01(x: float, x0: float, gamma: float) -> float:
        if gamma <= 0:
            return float("nan")
        if not (0.0 <= x <= 1.0):
            return 0.0
        base = 1.0 / (math.pi * gamma * (1.0 + ((x - x0) / gamma) ** 2))
        F = lambda t: 0.5 + math.atan((t - x0) / gamma) / math.pi
        Z = max(1e-15, F(1.0) - F(0.0))
        return base / Z
    @staticmethod
    def trunc_pareto01(x: float, xm: float, alpha: float) -> float:
        if not (0.0 < xm < 1.0) or alpha <= 0:
            return float("nan")
        if x < xm or x > 1.0:
            return 0.0
        base = alpha * (xm ** alpha) / (x ** (alpha + 1.0))
        Z = max(1e-15, 1.0 - (xm ** alpha))
        return base / Z
    @staticmethod
    def _gamma_pdf_base(t: float, alpha: float, beta: float, inv_norm: float) -> float:
        # inv_norm = 1 / (Gamma(alpha) * beta^alpha)
        # assumes t>=0
        return (t ** (alpha - 1.0)) * math.exp(-t / beta) * inv_norm
    @classmethod
    @lru_cache(maxsize=4096)
    def _trunc_gamma_Z(cls, alpha: float, beta: float, steps: int) -> float:
        # Normalize over [0,1] via trapezoid; cached.
        inv_norm = 1.0 / (math.gamma(alpha) * (beta ** alpha))
        dx = 1.0 / steps

        # trapezoid sum
        s = 0.5 * cls._gamma_pdf_base(0.0, alpha, beta, inv_norm) + 0.5 * cls._gamma_pdf_base(1.0, alpha, beta, inv_norm)
        for i in range(1, steps):
            t = i * dx
            s += cls._gamma_pdf_base(t, alpha, beta, inv_norm)

        return max(1e-15, s * dx)
    @classmethod
    def trunc_gamma01_numeric(cls, x: float, alpha: float, beta: float, steps: int = 2000) -> float:
        if alpha <= 0.0 or beta <= 0.0:
            return float("nan")
        if not (0.0 <= x <= 1.0):
            return 0.0

        inv_norm = 1.0 / (math.gamma(alpha) * (beta ** alpha))
        Z = cls._trunc_gamma_Z(alpha, beta, steps)
        return cls._gamma_pdf_base(x, alpha, beta, inv_norm) / Z
    @staticmethod
    def u_quadratic01(x: float) -> float:
        if 0.0 <= x <= 1.0:
            return 3.0 * x * x - 3.0 * x + 1.5
        return 0.0
    @staticmethod
    def u_quadratic01(x: float) -> float:  # Here to ensure testing / scaling works
        if 0.0 <= x <= 1.0:
            return 6.0 * x * x - 6.0 * x + 3.0
        return 0.0
    @staticmethod
    def kumaraswamy01(x: float, a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return float("nan")
        if 0.0 < x < 1.0:
            return a * b * (x ** (a - 1.0)) * ((1.0 - x ** a) ** (b - 1.0))
        # endpoints: depends on parameters; for histogram overlays return 0 at 0/1
        return 0.0


# TODO: Use Pytest to make a bunch of different test from this one test
def test_weighted_functions():
    """
            # TODO: Detect and remove high variance items? (Append scaled scales)
            # for boundry, bdata in zip(bin_boundaries, binned_data):
            #     number_of_samples: int = samples // bins
            #     avg_for_bin: float = sum(test_func(x, *args) for x in linspace(last_boundry, boundry, number_of_samples)) / number_of_samples
            #     # if avg_for_bin < 1 / (samples / 10):
            #     #     continue
            #     if avg_for_bin == 0.0:
            #         continue
            #     # elif len(scales) > 0 and len(bdata) / samples * bins / avg_for_bin > 1.5 * (sum(scales) / len(scales)):
            #     #     continue
            #     internal_scale: float = len(bdata) / samples * bins / avg_for_bin
            #     differences.append(abs(len(bdata) / samples * bins - avg_for_bin))
            #     scales.append(internal_scale)
            #     last_boundry = boundry
            #     scale += internal_scale * (len(bdata) / samples)
            # last_boundry: float = 0.0
            # for boundry, bdata in zip(bin_boundaries, binned_data):
            #     number_of_samples: int = len(bdata)
            #     if number_of_samples == 0: continue
            #     avg_for_bin: float = sum(test_func(x, *args) for x in linspace(last_boundry, boundry, number_of_samples)) / number_of_samples
            #     if avg_for_bin == 0.0: continue
            #     internal_scale: float = len(bdata) / samples * bins / avg_for_bin
            #     last_boundry = boundry
            #     scale += min(0.1, internal_scale * (len(bdata) / samples))
            # scale /= 2
            # max_test_func = max(vals)  # max(test_func(x, *args) for x in linspace(0, 1, 1000))
            # scale = max(len(x) for x in binned_data) / samples * bins / max_test_func
            # scale = sum(scales) / len(scales)
    """
    functions_to_test: dict[tuple[str, tuple[_ty.Any, ...]], _a.Callable[[float, _ty.Any], float]] = {
        ("g.random", tuple()): lambda x: 1,
        ("g.uniform", (0.0, 1.0)): lambda x, _, __: 1,
        ("linear", (1.0, 0.0)): PDFs.linear_trunc01,
        ("power", (2.0,)): PDFs.power01,
        ("quadratic", tuple()): lambda x: PDFs.power01(x, 3.0),
        ("cubic", tuple()): lambda x: PDFs.power01(x, 4.0),
        ("quartic", tuple()): lambda x: PDFs.power01(x, 5.0),
        ("gaussian", (0.5, 0.15)): PDFs.trunc_normal01,
        ("exponential", (2.5,)): PDFs.trunc_exponential01,
        ("beta_mean_kappa", (0.35, 6.0)): PDFs.beta_mean_kappa01,
        ("arcsine", tuple()): PDFs.arcsine01,
        ("triangular", (0.5,)): PDFs.triangular01,
        ("beta", (2.0, 5.0)): PDFs.beta01,
        ("logit_normal", (0.0, 1.0)): PDFs.logit_normal01,
        ("trapezoidal", (0.0, 0.3, 0.7, 1.0)): PDFs.trapezoidal01,
        ("weibull", (2.0, 1.0)): PDFs.trunc_weibull01,
        ("gamma", (2.0, 1.0)): PDFs.trunc_gamma01_numeric,
        ("cauchy", (0.0, 1.0)): PDFs.trunc_cauchy01,
        ("pareto", (0.2, 2.0)): PDFs.trunc_pareto01,
        ("u_quadratic", tuple()): PDFs.u_quadratic01,
        ("kumaraswamy", (2.0, 5.0)): PDFs.kumaraswamy01,
    }
    samples_lst = [100, 1_000, 10_000, 100_000]
    bins_lst = [int(x ** 0.5) for x in samples_lst]  # [25, 50, 100, 200]

    for samples, bins in zip(samples_lst, bins_lst):
        bin_segment_length: float = 1.0 / bins
        fbins: list[float] = [min(1.0, bin_segment_length * i) for i in range(bins+1)]
        bin_boundaries: list[float] = [min(1.0, bin_segment_length * i) for i in range(1, bins+1)]
        rng = WeightedFunctions(Random())  # We already test the random generators in the other function no need here.

        for (name, args), test_func in functions_to_test.items():
            vals: list[float] = [test_func(x, *args) for x in fbins]
            area: float = integrate(vals, bin_segment_length)

            func: _a.Callable[[_ty.Any], float]
            if name.startswith("g."):
                func = getattr(rng.g, name.removeprefix("g."))
            else:
                func = getattr(rng, name)

            data: list[float] = list(func(*args) for _ in range(samples))

            # Transform into bins
            data.sort()
            binned_data: list[list[float]] = [list() for _ in range(bins)]
            binned_i: int = 0

            for d in data:
                if d <= bin_boundaries[binned_i]:
                    binned_data[binned_i].append(d)
                else:
                    binned_i = min(bins-1, binned_i + 1)
                    binned_data[binned_i].append(d)

            # Calculate density scale
            hi_fi: float = 0.0
            fi_fi: float = 0.0
            last_boundry: float = 0.0
            averages: list[float] = list()
            normalized_binned: list[float] = list()

            for boundry, bdata in zip(bin_boundaries, binned_data):
                number_of_samples: int = samples // bins
                avg_for_bin: float = sum(test_func(x, *args) for x in linspace(last_boundry, boundry, number_of_samples)) / number_of_samples
                normalized_bdata = (len(bdata) / samples) * bins

                hi_fi += normalized_bdata * avg_for_bin
                fi_fi += avg_for_bin ** 2

                last_boundry = boundry
                averages.append(avg_for_bin)
                normalized_binned.append(normalized_bdata)

            scale = hi_fi / max(1e-12, fi_fi)
            distribution_area: float = sum(list((len(x) / (samples / bins)) * bin_segment_length for x in binned_data))

            if abs(1.0 - scale) > 0.05 and bins <= 60:
                print("[WARNING] Scaling safeguard crossed! Please investigate PDFs")  # scale = 1.0

            threshold: float = 20 / bins
            area_good: bool = abs((area * scale) - distribution_area) <= threshold
            print(f"{name}-Area", area_good)

            okay_num: int = 0
            for avg_for_bin, normalized_bdata in zip(averages, normalized_binned):
                if abs((avg_for_bin * scale) - normalized_bdata) <= 30 / bins:
                    okay_num += 1

            if area_good and okay_num >= len(averages) * 0.8:
                continue

            print(f"[ERROR] {name} failed with {okay_num}/{len(averages)} and area {'good' if area_good else 'bad'}.")
            try:
                import matplotlib.pyplot as plt

                counts, edges, _ = plt.hist(  # Histogram (density=True normalizes area to 1)
                    data,
                    bins=bins,
                    range=(0.0, 1.0),
                    density=True,
                    alpha=0.5,
                    label="Sampled distribution"
                )
                if max(vals) > 0:
                    for i, x in enumerate(vals):
                        vals[i] = x * scale

                plt.plot(fbins, vals, color="red", linewidth=2, label="Test function (scaled)")

                plt.title(name)
                plt.xlim(0.0, 1.0)
                plt.legend()
                plt.show()
            except ImportError:
                pass
