from typing import Union, Optional
from scipy.stats import t, norm
import numpy as np

Vector = Union[np.ndarray, list, tuple]


def pformat(values: Vector, precision) -> str:
    """
    Return string of comma-separated values with given number of digits after dot

    :param values: array of values
    :param precision: number of digits after dot
    :return: formatted string
    """
    return ", ".join((("{:.{}f}".format(num, precision)) for num in values))


def mean(values: Vector) -> float:
    """
    Compute mean of the vector

    :param values: array of values
    :return: mean of the array
    """
    return sum(values) / len(values)


def central_moment(values: Vector, k: int) -> float:
    """
    Compute k-th central moment of data

    :param values: array of values
    :param k: degree of moment
    :return: central moment of k-th degree
    """
    values = np.array(values)
    m = mean(values)
    return sum(((values - m) ** k)) / len(values)


def asymmetry(values: Vector) -> float:
    """
    Compute asymmetry coefficient of given data

    :param values: array of values
    :return: asymmetry coefficient
    """
    return central_moment(values, 3) / std(values) ** 3


def kurtosis(values: Vector) -> float:
    """
    Compute kurtosis coefficient of given data

    :param values: array of values
    :return: kurtosis coefficient
    """
    return central_moment(values, 4) / std(values) ** 4 - 3


def dispersion(values: Vector) -> float:
    """
    Compute dispersion of the vector

    :param values: array of values
    :return: dispersion of the array
    """
    return central_moment(values, 2)


def std(values: Vector) -> float:
    """
    Compute standard deviation of the vector

    :param values: array of values
    :return: standard deviation of the array
    """
    return np.sqrt(dispersion(values))


def quantile(values: Vector, p: float) -> float:
    """
    Compute quantile for given data

    :param values: array of values
    :param p: quantile value ([0..1])
    :return: quantile element
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Erroneous quantile value = {p}")

    values = sorted(values)
    i = int(np.trunc(len(values) * p))
    return float(values[i])


def median(values: Vector) -> float:
    """
    Return median of the array

    :param values: array of values
    :return: median of the given array
    """

    if len(values) % 2 == 1:
        return quantile(values, 0.5)
    else:
        values = sorted(values)
        x = len(values) // 2
        return (values[x-1] + values[x]) / 2


def correlation_coef(x: Vector, y: Vector) -> float:
    """
    Compute correlation coefficient for vectors x and y (lengths should be same)

    :param x: array of values
    :param y: another array of values
    :return: correlation coefficient
    """

    x = np.array(x)
    y = np.array(y)

    mx = mean(x)
    my = mean(y)

    sx = x - mx
    sy = y - my

    xy = sum(sx * sy)
    d = np.sqrt(sum(sx ** 2) * sum(sy ** 2))

    return xy / d


def get_student_k(s2x: float, s2y: float, m: int, n: int) -> int:
    """
    Calculate K value for Student distribution T(K) if variances are not equal and not known.

    :param s2x: Unshifted variance of X
    :param s2y: Unshifted variance of Y
    :param m: len of X sample
    :param n: len of Y sample
    :return: K degree of freedom for T(K)
    """

    k_dvd = (s2x/m + s2y/n)**2

    k_dvs_l = ((s2x/m)**2)/(m-1)
    k_dvs_r = ((s2y/n)**2)/(n-1)

    k = k_dvd / (k_dvs_l + k_dvs_r)
    return np.floor(k)


def norm_mean_pvalue(data: Vector, mean_x: Union[int, float], var: Optional[float] = None) -> float:
    """
    Calculate p-value for hypothesis of mean value of data sample = mean_x, where data sample is normally distributed

    :param data: sample of data
    :param mean_x: supposed mean value
    :param var: (optional parameter) true variance of distribution
    :return: p-value
    """

    n = len(data)

    m = data.mean()

    if var is None:
        var = sum((data - m) ** 2) / (n - 1)
        f = t(n-1)
    else:
        f = norm()

    Z = (m - mean_x) / (np.sqrt(var) / np.sqrt(n))

    p = 2 * min(f.cdf(Z), 1 - f.cdf(Z))
    return p

