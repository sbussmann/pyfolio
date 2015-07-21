"""
Historical Volatility Estimators

ref: http://www.todaysgroep.nl/media/236846/measuring_historic_volatility.pdf
"""


import numpy as np


def overnight_returns(O, C):
    """
    :param O: Series/DataFrame of portfolio values or prices at
        the open of the day
    :param C: Series/DataFrame of portfolio values or prices at
        the close of the day

    :returns: Series/Dataframe of overnight gap returns
    """
    return np.log(O / C.shift(1))


def intraday_returns(O, C):
    """
    :param O: Series/DataFrame of portfolio values or prices at
        the open of the day
    :param C: Series/DataFrame of portfolio values or prices at
        the close of the day

    :returns: Series/Dataframe of intraday returns
    """
    return np.log(C / O)


def intraday_range(H, L):
    """
    :param H: Series/DataFrame of daily high portfolio values or prices
    :param C: Series/DataFrame of daily low portfolio values or prices

    :returns: Series/Dataframe of intraday ranges
    """
    return np.log(H / L)


def log_returns(values):
    """
    :param values: Series/DataFrame of portfolio values or prices

    :returns: Series/Dataframe of log returns
    """
    return np.log(values / values.shift(1)).dropna()


def close_to_close(close_values):
    """
    :param close_values: Series/DataFrame of portfolio values or prices

    :returns: float/Series daily standard deviations.
    """
    log_return = np.log(close_values / close_values.shift(1))
    return np.std(log_return, ddof=1)


def parkinson_vol(hi, low):
    ranges = np.square(intraday_range(hi, low))
    var = np.sum(ranges) / (4 * ranges.shape[0] * np.log(2))
    return np.sqrt(var)


def garman_klass(O, H, L, C):
    ranges = np.square(intraday_range(H, L))
    intraday_R = np.square(intraday_returns(C, O))
    var = 0.5 * np.sum(ranges) - (2 * np.log(2) - 1) * np.sum(intraday_R)
    return np.sqrt(var / ranges.shape[0])


def garman_klass_ext(O, H, L, C):
    oR = np.square(overnight_returns(O, C))
    iR = np.square(intraday_returns(O, C))
    irange = np.square(intraday_range(H, L))
    var = oR.sum() + irange / 2.0 - (2 * np.log(2) - 1) * iR.sum()
    return np.sqrt(var / O.shape[0])


def rodgers_satchell(O, H, L, C):
    high_ref = np.log(H / C) * np.log(H / O)
    low_ref = np.log(L / C) * np.log(L / O)
    return np.sqrt((high_ref.sum() + low_ref.sum()) / O.shape[0])


def yang_zhang(O, H, L, C):
    N = O.shape[0]
    k = 0.34 / (1.34 + (N + 1) / (N - 1))
    overnight_var = overnight_returns(O, C).var()
    intraday_var = intraday_returns(O, C).var()
    rs_var = rodgers_satchell(O, H, L, C) ** 2
    var = overnight_var + k * intraday_var + (1 - k) * rs_var
    return np.sqrt(var)
