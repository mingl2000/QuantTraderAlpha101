"""
WorldQuant 101 Alphas — corrected implementation scaffold
- Keeps your interface: WorldQuant_101_Alphas(df_data) and alpha_###() methods.
- Fixes the *core* semantic bugs (rank/scale/ts_rank orientation, signed_power, pandas alias, NaN safety).
- Updates the alphas you posted to match the published formulas more closely.

Assumptions (same as your code):
- Each field in df_data (open/high/low/close/volume/returns/amount/vwap) is a pandas DataFrame:
    index = dates, columns = tickers
  (Series will also work in most places but DataFrame is the intended form.)
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ==================== Core operators (WorldQuant-style) ====================

def _to_df(x):
    if isinstance(x, pd.Series):
        return x.to_frame()
    return x

def correlation(x, y, window=10):
    x = _to_df(x)
    y = _to_df(y)
    return x.rolling(window).corr(y)

def covariance(x, y, window=10):
    x = _to_df(x)
    y = _to_df(y)
    return x.rolling(window).cov(y)

def rank(df):
    # Cross-sectional rank per date (across tickers)
    df = _to_df(df)
    return df.rank(axis=1, pct=True)

def scale(df, k=1):
    # Cross-sectional scaling per date
    df = _to_df(df)
    denom = df.abs().sum(axis=1).replace(0, np.nan)
    return df.mul(k).div(denom, axis=0)

def delta(df, period=1):
    df = _to_df(df)
    return df.diff(period)

def delay(df, period=1):
    df = _to_df(df)
    return df.shift(period)

def ts_sum(df, window=10):
    df = _to_df(df)
    return df.rolling(window).sum()

def sma(df, window=10):
    # rolling mean
    df = _to_df(df)
    return df.rolling(window).mean()

def stddev(df, window=10):
    df = _to_df(df)
    return df.rolling(window).std()

def ts_min(df, window=10):
    df = _to_df(df)
    return df.rolling(window).min()

def ts_max(df, window=10):
    df = _to_df(df)
    return df.rolling(window).max()

def ts_argmax(df, window=10):
    df = _to_df(df)
    return df.rolling(window).apply(lambda x: np.argmax(x) + 1, raw=True)

def ts_argmin(df, window=10):
    df = _to_df(df)
    return df.rolling(window).apply(lambda x: np.argmin(x) + 1, raw=True)

def ts_rank(df, window=10):
    # Rank of the *latest* value within the rolling window (1..N)
    df = _to_df(df)
    return df.rolling(window).apply(lambda x: rankdata(x)[-1], raw=True)

def signed_power(x, a):
    x = _to_df(x)
    return np.sign(x) * (np.abs(x) ** a)

def product(df, window=10):
    df = _to_df(df)
    return df.rolling(window).apply(np.prod, raw=True)

def decay_linear(df, period=10):
    """
    Linear Weighted Moving Average (LWMA), weights 1..period (more weight on recent).
    Returns DataFrame with same index/columns.

    Note: This implementation avoids in-place NaN filling (no side effects).
    """
    df = _to_df(df).copy()
    df = df.ffill().bfill().fillna(0)

    w = np.arange(1, period + 1, dtype=float)
    w /= w.sum()

    arr = df.values
    out = np.zeros_like(arr, dtype=float)

    # Fill the first (period-1) rows with original (common practice in many ref implementations)
    out[: period - 1, :] = arr[: period - 1, :]

    for i in range(period - 1, arr.shape[0]):
        window_slice = arr[i - period + 1 : i + 1, :]
        out[i, :] = (window_slice * w[:, None]).sum(axis=0)

    return pd.DataFrame(out, index=df.index, columns=df.columns)

def _clean(df):
    # Many Alpha101 references set inf->0 and NaN->0 at the end of each alpha
    return _to_df(df).replace([np.inf, -np.inf], 0).fillna(0)

# ==================== Alphas ====================

class WorldQuant_101_Alphas(object):
    def __init__(self, df_data):
        self.open = _to_df(df_data["open"])
        self.close = _to_df(df_data["close"])
        self.high = _to_df(df_data["high"])
        self.low = _to_df(df_data["low"])
        self.volume = _to_df(df_data["volume"])
        self.returns = _to_df(df_data["returns"])
        self.amount = _to_df(df_data["amount"])

        vwap = df_data.get("vwap", None)
        if vwap is not None and not _to_df(vwap).empty:
            self.vwap = _to_df(vwap)
        else:
            # your fallback (kept)
            self.vwap = self.amount * 1000.0 / (self.volume + 1e-8)

    # Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    def alpha_001(self):
        inner = self.close.where(self.returns >= 0, stddev(self.returns, 20))
        out = rank(ts_argmax(signed_power(inner, 2.0), 5)) - 0.5
        return _clean(out)

    # Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha_002(self):
        out = -1.0 * correlation(
            rank(delta(np.log(self.volume), 2)),
            rank((self.close - self.open) / self.open.replace(0, np.nan)),
            6,
        )
        return _clean(out)

    # Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    def alpha_003(self):
        out = -1.0 * correlation(rank(self.open), rank(self.volume), 10)
        return _clean(out)

    # Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    def alpha_004(self):
        out = -1.0 * ts_rank(rank(self.low), 9)
        return _clean(out)

    # Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha_005(self):
        out = rank(self.open - (ts_sum(self.vwap, 10) / 10.0)) * (-1.0 * np.abs(rank(self.close - self.vwap)))
        return _clean(out)

    # Alpha#6: (-1 * correlation(open, volume, 10))
    def alpha_006(self):
        out = -1.0 * correlation(self.open, self.volume, 10)
        return _clean(out)

    # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    def alpha_007(self):
        adv20 = sma(self.volume, 20)
        sig = -1.0 * ts_rank(np.abs(delta(self.close, 7)), 60) * np.sign(delta(self.close, 7))
        out = sig.where(adv20 < self.volume, -1.0)
        return _clean(out)

    # Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha_008(self):
        inner = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        out = -1.0 * rank(inner - delay(inner, 10))
        return _clean(out)

    # Alpha#9: conditional on ts_min/max of delta(close,1)
    def alpha_009(self):
        d = delta(self.close, 1)
        cond1 = ts_min(d, 5) > 0
        cond2 = ts_max(d, 5) < 0
        out = (-1.0 * d).where(~(cond1 | cond2), d)
        return _clean(out)

    # Alpha#10: rank(conditional)
    def alpha_010(self):
        d = delta(self.close, 1)
        cond1 = ts_min(d, 4) > 0
        cond2 = ts_max(d, 4) < 0
        inner = (-1.0 * d).where(~(cond1 | cond2), d)
        out = rank(inner)
        return _clean(out)

    # Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
    def alpha_011(self):
        out = (rank(ts_max(self.vwap - self.close, 3)) + rank(ts_min(self.vwap - self.close, 3))) * rank(delta(self.volume, 3))
        return _clean(out)

    # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha_012(self):
        out = np.sign(delta(self.volume, 1)) * (-1.0 * delta(self.close, 1))
        return _clean(out)

    # Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha_013(self):
        out = -1.0 * rank(covariance(rank(self.close), rank(self.volume), 5))
        return _clean(out)

    # Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha_014(self):
        corr = _clean(correlation(self.open, self.volume, 10))
        out = (-1.0 * rank(delta(self.returns, 3))) * corr
        return _clean(out)

    # Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha_015(self):
        corr = _clean(correlation(rank(self.high), rank(self.volume), 3))
        out = -1.0 * ts_sum(rank(corr), 3)
        return _clean(out)

    # Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha_016(self):
        out = -1.0 * rank(covariance(rank(self.high), rank(self.volume), 5))
        return _clean(out)

    # Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    def alpha_017(self):
        adv20 = sma(self.volume, 20)
        out = (-1.0 * rank(ts_rank(self.close, 10))) * rank(delta(delta(self.close, 1), 1)) * rank(ts_rank(self.volume / adv20.replace(0, np.nan), 5))
        return _clean(out)

    # Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    def alpha_018(self):
        corr = _clean(correlation(self.close, self.open, 10))
        out = -1.0 * rank(stddev(np.abs(self.close - self.open), 5) + (self.close - self.open) + corr)
        return _clean(out)

    # Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    def alpha_019(self):
        out = (-1.0 * np.sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) * (1.0 + rank(1.0 + ts_sum(self.returns, 250)))
        return _clean(out)

    # Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
    def alpha_020(self):
        out = -1.0 * rank(self.open - delay(self.high, 1)) * rank(self.open - delay(self.close, 1)) * rank(self.open - delay(self.low, 1))
        return _clean(out)

    # Alpha#21: keep same interface, but this alpha is easy to get wrong; below matches published ternaries more closely.
    def alpha_021(self):
        # ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1)
        # : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1
        # : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1))))
        ma8 = ts_sum(self.close, 8) / 8.0
        sd8 = stddev(self.close, 8)
        ma2 = ts_sum(self.close, 2) / 2.0
        adv20 = sma(self.volume, 20)
        v_over_adv = self.volume / adv20.replace(0, np.nan)

        out = pd.DataFrame(-1.0, index=self.close.index, columns=self.close.columns)
        cond_a = (ma8 + sd8) < ma2
        cond_b = ma2 < (ma8 - sd8)
        cond_c = v_over_adv >= 1.0

        out = out.where(cond_a, out)          # default -1
        out = out.mask(cond_b, 1.0)           # if cond_b -> 1
        out = out.mask(~cond_a & ~cond_b & cond_c, 1.0)  # else if cond_c -> 1, else -1
        return _clean(out)

    # Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha_022(self):
        corr = _clean(correlation(self.high, self.volume, 5))
        out = -1.0 * delta(corr, 5) * rank(stddev(self.close, 20))
        return _clean(out)

    # Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha_023(self):
        cond = (ts_sum(self.high, 20) / 20.0) < self.high
        out = pd.DataFrame(0.0, index=self.close.index, columns=self.close.columns)
        out = out.mask(cond, -1.0 * delta(self.high, 2))
        return _clean(out)

    # Alpha#24: conditional (as in paper)
    def alpha_024(self):
        cond = (delta(ts_sum(self.close, 100) / 100.0, 100) / delay(self.close, 100).replace(0, np.nan)) <= 0.05
        out = -1.0 * delta(self.close, 3)
        out = out.mask(cond, -1.0 * (self.close - ts_min(self.close, 100)))
        return _clean(out)

    # Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha_025(self):
        adv20 = sma(self.volume, 20)
        out = rank(((-1.0 * self.returns) * adv20) * self.vwap * (self.high - self.close))
        return _clean(out)

    # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha_026(self):
        corr = _clean(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5))
        out = -1.0 * ts_max(corr, 3)
        return _clean(out)

    # Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1) : 1)
    def alpha_027(self):
        inner = ts_sum(_clean(correlation(rank(self.volume), rank(self.vwap), 6)), 2) / 2.0
        r = rank(inner)
        out = pd.DataFrame(1.0, index=r.index, columns=r.columns)
        out = out.mask(r > 0.5, -1.0)
        return _clean(out)

    # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha_028(self):
        adv20 = sma(self.volume, 20)
        corr = _clean(correlation(adv20, self.low, 5))
        out = scale(corr + ((self.high + self.low) / 2.0) - self.close)
        return _clean(out)

    # Alpha#29: You had syntax issues; below is a faithful structural translation of the published formula.
    # (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5)
    #  + ts_rank(delay((-1 * returns), 6), 5))
    def alpha_029(self):
        inner = -1.0 * rank(delta(self.close - 1.0, 5))
        inner = rank(rank(inner))
        inner = ts_min(inner, 2)
        inner = ts_sum(inner, 1)
        inner = np.log(inner.replace(0, np.nan))
        inner = scale(inner)
        inner = rank(rank(inner))
        p1 = ts_min(product(inner, 1), 5)
        p2 = ts_rank(delay(-1.0 * self.returns, 6), 5)
        out = p1 + p2
        return _clean(out)

    # Alpha#30: (((1.0 - rank(sign(close-delay(close,1)) + sign(delay(close,1)-delay(close,2)) + sign(delay(close,2)-delay(close,3)))) * sum(volume,5)) / sum(volume,20))
    def alpha_030(self):
        s1 = np.sign(self.close - delay(self.close, 1))
        s2 = np.sign(delay(self.close, 1) - delay(self.close, 2))
        s3 = np.sign(delay(self.close, 2) - delay(self.close, 3))
        inner = s1 + s2 + s3
        out = (1.0 - rank(inner)) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20).replace(0, np.nan)
        return _clean(out)

    # Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha_031(self):
        adv20 = sma(self.volume, 20)
        corr = _clean(correlation(adv20, self.low, 12))
        p1 = rank(rank(rank(decay_linear((-1.0 * rank(rank(delta(self.close, 10)))), 10))))
        p2 = rank(-1.0 * delta(self.close, 3))
        p3 = np.sign(scale(corr))
        out = p1 + p2 + p3
        return _clean(out)

    # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha_032(self):
        p1 = scale((ts_sum(self.close, 7) / 7.0) - self.close)
        p2 = 20.0 * scale(_clean(correlation(self.vwap, delay(self.close, 5), 230)))
        out = p1 + p2
        return _clean(out)

    # Alpha#33: rank((-1 * ((1 - (open / close))^1)))  -> equals rank((open/close) - 1)
    def alpha_033(self):
        out = rank((self.open / self.close.replace(0, np.nan)) - 1.0)
        return _clean(out)

    # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha_034(self):
        ratio = stddev(self.returns, 2) / stddev(self.returns, 5).replace(0, np.nan)
        inner = 1.0 - rank(ratio)
        out = rank(inner + (1.0 - rank(delta(self.close, 1))))
        return _clean(out)

    # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha_035(self):
        out = ts_rank(self.volume, 32) * (1.0 - ts_rank((self.close + self.high) - self.low, 16)) * (1.0 - ts_rank(self.returns, 32))
        return _clean(out)

    # Alpha#36: long expression — keep structure; add cleaning around correlations
    def alpha_036(self):
        adv20 = sma(self.volume, 20)
        c1 = _clean(correlation(self.close - self.open, delay(self.volume, 1), 15))
        c2 = _clean(correlation(self.vwap, adv20, 6))
        p1 = 2.21 * rank(c1)
        p2 = 0.7 * rank(self.open - self.close)
        p3 = 0.73 * rank(ts_rank(delay(-1.0 * self.returns, 6), 5))
        p4 = rank(np.abs(c2))
        p5 = 0.6 * rank(((ts_sum(self.close, 200) / 200.0) - self.open) * (self.close - self.open))
        out = p1 + p2 + p3 + p4 + p5
        return _clean(out)

    # Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha_037(self):
        out = rank(_clean(correlation(delay(self.open - self.close, 1), self.close, 200))) + rank(self.open - self.close)
        return _clean(out)

    # Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha_038(self):
        out = (-1.0 * rank(ts_rank(self.close, 10))) * rank(self.close / self.open.replace(0, np.nan))
        return _clean(out)

    # Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha_039(self):
        adv20 = sma(self.volume, 20)
        inner = delta(self.close, 7) * (1.0 - rank(decay_linear(self.volume / adv20.replace(0, np.nan), 9)))
        out = (-1.0 * rank(inner)) * (1.0 + rank(ts_sum(self.returns, 250)))
        return _clean(out)

    # Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha_040(self):
        out = (-1.0 * rank(stddev(self.high, 10))) * _clean(correlation(self.high, self.volume, 10))
        return _clean(out)

    # Alpha#41: (((high * low)^0.5) - vwap)
    def alpha_041(self):
        out = np.sqrt(self.high * self.low) - self.vwap
        return _clean(out)

    # Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    def alpha_042(self):
        out = rank(self.vwap - self.close) / rank(self.vwap + self.close).replace(0, np.nan)
        return _clean(out)

    # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha_043(self):
        adv20 = sma(self.volume, 20)
        out = ts_rank(self.volume / adv20.replace(0, np.nan), 20) * ts_rank(-1.0 * delta(self.close, 7), 8)
        return _clean(out)

    # Alpha#44: (-1 * correlation(high, rank(volume), 5))
    def alpha_044(self):
        out = -1.0 * _clean(correlation(self.high, rank(self.volume), 5))
        return _clean(out)

    # Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha_045(self):
        p1 = rank(ts_sum(delay(self.close, 5), 20) / 20.0)
        c1 = _clean(correlation(self.close, self.volume, 2))
        c2 = _clean(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        out = -1.0 * (p1 * c1 * rank(c2))
        return _clean(out)

    # Alpha#46: as given in paper (ternary)
    def alpha_046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10.0) - ((delay(self.close, 10) - self.close) / 10.0)
        out = (-1.0 * (self.close - delay(self.close, 1)))
        out = out.mask(inner < 0, 1.0)
        out = out.mask(inner > 0.25, -1.0)
        return _clean(out)

    # Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    def alpha_047(self):
        adv20 = sma(self.volume, 20)
        denom = (ts_sum(self.high, 5) / 5.0).replace(0, np.nan)
        out = ((rank(1.0 / self.close.replace(0, np.nan)) * self.volume) / adv20.replace(0, np.nan)) * ((self.high * rank(self.high - self.close)) / denom) \
              - rank(self.vwap - delay(self.vwap, 5))
        return _clean(out)

    # Alpha#49
    def alpha_049(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10.0) - ((delay(self.close, 10) - self.close) / 10.0)
        out = (-1.0 * (self.close - delay(self.close, 1)))
        out = out.mask(inner < -0.1, 1.0)
        return _clean(out)

    # Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha_050(self):
        out = -1.0 * ts_max(rank(_clean(correlation(rank(self.volume), rank(self.vwap), 5))), 5)
        return _clean(out)

    # Alpha#51
    def alpha_051(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10.0) - ((delay(self.close, 10) - self.close) / 10.0)
        out = (-1.0 * (self.close - delay(self.close, 1)))
        out = out.mask(inner < -0.05, 1.0)
        return _clean(out)

    # Alpha#52
    def alpha_052(self):
        out = (-1.0 * (ts_min(self.low, 5) - delay(ts_min(self.low, 5), 5))) * rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220.0) * ts_rank(self.volume, 5)
        return _clean(out)

    # Alpha#53
    def alpha_053(self):
        denom = (self.close - self.low).replace(0, np.nan)
        out = -1.0 * delta(((self.close - self.low) - (self.high - self.close)) / denom, 9)
        return _clean(out)

    # Alpha#54
    def alpha_054(self):
        denom = (self.low - self.high).replace(0, np.nan)
        out = -1.0 * (self.low - self.close) * (self.open ** 5) / (denom * (self.close ** 5).replace(0, np.nan))
        return _clean(out)

    # Alpha#55
    def alpha_055(self):
        denom = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, np.nan)
        inner = (self.close - ts_min(self.low, 12)) / denom
        out = -1.0 * _clean(correlation(rank(inner), rank(self.volume), 6))
        return _clean(out)

    # Alpha#57
    def alpha_057(self):
        out = -1.0 * ((self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)), 2).replace(0, np.nan))
        return _clean(out)

    # Alpha#60
    def alpha_060(self):
        denom = (self.high - self.low).replace(0, np.nan)
        inner = (((self.close - self.low) - (self.high - self.close)) / denom) * self.volume
        out = -1.0 * ((2.0 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))
        return _clean(out)

    # Alpha#61
    def alpha_061(self):
        adv180 = sma(self.volume, 180)
        out = rank(self.vwap - ts_min(self.vwap, 16)) < rank(_clean(correlation(self.vwap, adv180, 18)))
        return _clean(out.astype(float))

    # Alpha#62
    def alpha_062(self):
        adv20 = sma(self.volume, 20)
        left = rank(_clean(correlation(self.vwap, sma(adv20, 22), 10)))
        right = rank((rank(self.open) + rank(self.open)) < (rank((self.high + self.low) / 2.0) + rank(self.high)))
        out = (left < right) * -1.0
        return _clean(out.astype(float))

    # Alpha#64
    def alpha_064(self):
        adv120 = sma(self.volume, 120)
        left = rank(_clean(correlation(sma(self.open * 0.178404 + self.low * (1 - 0.178404), 13), sma(adv120, 13), 17)))
        right = rank(delta(((self.high + self.low) / 2.0) * 0.178404 + self.vwap * (1 - 0.178404), 4))
        out = (left < right) * -1.0
        return _clean(out.astype(float))

    # Alpha#65
    def alpha_065(self):
        adv60 = sma(self.volume, 60)
        left = rank(_clean(correlation(self.open * 0.00817205 + self.vwap * (1 - 0.00817205), sma(adv60, 9), 6)))
        right = rank(self.open - ts_min(self.open, 14))
        out = (left < right) * -1.0
        return _clean(out.astype(float))

    # Alpha#66
    def alpha_066(self):
        p1 = rank(decay_linear(delta(self.vwap, 4), 7))
        denom = (self.open - (self.high + self.low) / 2.0).replace(0, np.nan)
        inner = ((self.low - self.vwap) / denom)
        p2 = ts_rank(decay_linear(inner, 11), 7)
        out = (p1 + p2) * -1.0
        return _clean(out)

    # Alpha#68
    def alpha_068(self):
        adv15 = sma(self.volume, 15)
        left = ts_rank(_clean(correlation(rank(self.high), rank(adv15), 9)), 14)
        right = rank(delta(self.close * 0.518371 + self.low * (1 - 0.518371), 1))
        out = (left < right) * -1.0
        return _clean(out.astype(float))

    # Alpha#71 (note: max should be elementwise; your prior version mixed shapes)
    def alpha_071(self):
        adv180 = sma(self.volume, 180)
        p1 = ts_rank(decay_linear(_clean(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18)), 4), 16)
        p2 = ts_rank(decay_linear(rank((self.low + self.open) - (self.vwap + self.vwap)) ** 2, 16), 4)
        out = np.maximum(_to_df(p1), _to_df(p2))
        return _clean(out)

    # Alpha#72
    def alpha_072(self):
        adv40 = sma(self.volume, 40)
        num = rank(decay_linear(_clean(correlation((self.high + self.low) / 2.0, adv40, 9)), 10))
        den = rank(decay_linear(_clean(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7)), 3)).replace(0, np.nan)
        out = num / den
        return _clean(out)

    # Alpha#73
    def alpha_073(self):
        p1 = rank(decay_linear(delta(self.vwap, 5), 3))
        base = self.open * 0.147155 + self.low * (1 - 0.147155)
        inner = (delta(base, 2) / base.replace(0, np.nan)) * -1.0
        p2 = ts_rank(decay_linear(inner, 3), 17)
        out = -1.0 * np.maximum(_to_df(p1), _to_df(p2))
        return _clean(out)

    # Alpha#74
    def alpha_074(self):
        adv30 = sma(self.volume, 30)
        left = rank(_clean(correlation(self.close, sma(adv30, 37), 15)))
        right = rank(_clean(correlation(rank(self.high * 0.0261661 + self.vwap * (1 - 0.0261661)), rank(self.volume), 11)))
        out = (left < right) * -1.0
        return _clean(out.astype(float))

    # Alpha#75
    def alpha_075(self):
        adv50 = sma(self.volume, 50)
        out = rank(_clean(correlation(self.vwap, self.volume, 4))) < rank(_clean(correlation(rank(self.low), rank(adv50), 12)))
        return _clean(out.astype(float))

    # Alpha#77
    def alpha_077(self):
        adv40 = sma(self.volume, 40)
        p1 = rank(decay_linear(((self.high + self.low) / 2.0 + self.high) - (self.vwap + self.high), 20))
        p2 = rank(decay_linear(_clean(correlation((self.high + self.low) / 2.0, adv40, 3)), 6))
        out = np.minimum(_to_df(p1), _to_df(p2))
        return _clean(out)

    # Alpha#78
    def alpha_078(self):
        adv40 = sma(self.volume, 40)
        left = rank(_clean(correlation(ts_sum(self.low * 0.352233 + self.vwap * (1 - 0.352233), 20), ts_sum(adv40, 20), 7)))
        right = rank(_clean(correlation(rank(self.vwap), rank(self.volume), 6)))
        out = left ** right
        return _clean(out)

    # Alpha#81
    def alpha_081(self):
        adv10 = sma(self.volume, 10)
        inner = rank(_clean(correlation(self.vwap, ts_sum(adv10, 50), 8))) ** 4
        inner = rank(inner)
        inner = product(inner, 15)
        left = rank(np.log(inner.replace(0, np.nan)))
        right = rank(_clean(correlation(rank(self.vwap), rank(self.volume), 5)))
        out = (left < right) * -1.0
        return _clean(out.astype(float))

    # Alpha#83
    def alpha_083(self):
        denom = (ts_sum(self.close, 5) / 5.0).replace(0, np.nan)
        a = (self.high - self.low) / denom
        out = (rank(delay(a, 2)) * rank(rank(self.volume))) / (a / (self.vwap - self.close).replace(0, np.nan))
        return _clean(out)

    # Alpha#84
    def alpha_084(self):
        out = signed_power(ts_rank(self.vwap - ts_max(self.vwap, 15), 21), delta(self.close, 5))
        return _clean(out)

    # Alpha#85
    def alpha_085(self):
        adv30 = sma(self.volume, 30)
        left = rank(_clean(correlation(self.high * 0.876703 + self.close * (1 - 0.876703), adv30, 10)))
        right = rank(_clean(correlation(ts_rank((self.high + self.low) / 2.0, 4), ts_rank(self.volume, 10), 7)))
        out = left ** right
        return _clean(out)

    # Alpha#86
    def alpha_086(self):
        adv20 = sma(self.volume, 20)
        left = ts_rank(_clean(correlation(self.close, sma(adv20, 15), 6)), 20)
        right = rank((self.open + self.close) - (self.vwap + self.open))
        out = (left < right) * -1.0
        return _clean(out.astype(float))

    # Alpha#88
    def alpha_088(self):
        adv60 = sma(self.volume, 60)
        p1 = rank(decay_linear((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close)), 8))
        p2 = ts_rank(decay_linear(_clean(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8)), 7), 3)
        out = np.minimum(_to_df(p1), _to_df(p2))
        return _clean(out)

    # Alpha#92
    def alpha_092(self):
        adv30 = sma(self.volume, 30)
        p1 = ts_rank(decay_linear(((self.high + self.low) / 2.0 + self.close) < (self.low + self.open), 15), 19)
        p2 = ts_rank(decay_linear(_clean(correlation(rank(self.low), rank(adv30), 8)), 7), 7)
        out = np.minimum(_to_df(p1), _to_df(p2))
        return _clean(out)

    # Alpha#94
    def alpha_094(self):
        adv60 = sma(self.volume, 60)
        left = rank(self.vwap - ts_min(self.vwap, 12))
        right = ts_rank(_clean(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18)), 3)
        out = (left ** right) * -1.0
        return _clean(out)

    # Alpha#95
    def alpha_095(self):
        adv40 = sma(self.volume, 40)
        left = rank(self.open - ts_min(self.open, 12))
        right = ts_rank(rank(_clean(correlation(sma((self.high + self.low) / 2.0, 19), sma(adv40, 19), 13))) ** 5, 12)
        out = left < right
        return _clean(out.astype(float))

    # Alpha#96
    def alpha_096(self):
        adv60 = sma(self.volume, 60)
        p1 = ts_rank(decay_linear(_clean(correlation(rank(self.vwap), rank(self.volume), 4)), 4), 8)
        p2 = ts_rank(decay_linear(ts_argmax(_clean(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4)), 13), 14), 13)
        out = -1.0 * np.maximum(_to_df(p1), _to_df(p2))
        return _clean(out)

    # Alpha#98
    def alpha_098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        left = rank(decay_linear(_clean(correlation(self.vwap, sma(adv5, 26), 5)), 7))
        right = rank(decay_linear(ts_rank(ts_argmin(_clean(correlation(rank(self.open), rank(adv15), 21)), 9), 7), 8))
        out = left - right
        return _clean(out)

    # Alpha#99
    def alpha_099(self):
        adv60 = sma(self.volume, 60)
        left = rank(_clean(correlation(ts_sum((self.high + self.low) / 2.0, 20), ts_sum(adv60, 20), 9)))
        right = rank(_clean(correlation(self.low, self.volume, 6)))
        out = (left < right) * -1.0
        return _clean(out.astype(float))

    # Alpha#101
    def alpha_101(self):
        out = (self.close - self.open) / ((self.high - self.low) + 0.001)
        return _clean(out)

    # Alpha#102
    def alpha_102(self):
        out = delta(self.close, 4) / delay(self.close, 4).replace(0, np.nan)
        return _clean(out)
    # Alpha#103
    def alpha_103(self):
        out = delta(self.close, 5) / delay(self.close, 5).replace(0, np.nan)

        return _clean(out)
    # Alpha#104
    def alpha_104(self):
        out = delta(self.close, 6) / delay(self.close, 6).replace(0, np.nan)
        return _clean(out)
    def alpha_105(self):
        out = delta(self.close, 4) / delay(self.close, 4).replace(0, np.nan)
        return _clean(out)

    # ---------------------------------------------------------------------
    # IMPORTANT:
    # You did not include alpha_056, 058-059, 063, 067, 069-070, 076, 079-080,
    # 082, 087, 089-091, 093, 097, 100, etc. in your snippet.
    # To keep the interface stable, you can add stubs like below:
    # def alpha_056(self): raise NotImplementedError("Alpha 56 not included in base snippet")
    # ---------------------------------------------------------------------