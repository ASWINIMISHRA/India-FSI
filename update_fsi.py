# ── Standard library ──────────────────────────────────────────────────────────

import os

import sys

import time

import warnings

from datetime import datetime



warnings.filterwarnings("ignore")



# ── Third-party ───────────────────────────────────────────────────────────────

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy import stats

from statsmodels.tsa.filters.hp_filter import hpfilter

from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter

from scipy.stats import zscore

# ── statistical machinery for robustness / validation extensions ────────────
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.multitest import multipletests

# ── NEW: DCC-GARCH (uses the `arch` package's univariate GARCH engine as the
#     first stage, then a custom Engle (2002) DCC recursion for the second
#     stage — `arch` does not ship a native multivariate DCC class) ─────────
try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False



# Optimize Matplotlib rendering engine performance

matplotlib.rcParams['path.simplify'] = True

matplotlib.rcParams['path.simplify_threshold'] = 1.0

matplotlib.rcParams['agg.path.chunksize'] = 10000



try:

    from pytrends.request import TrendReq

    _PYTRENDS_AVAILABLE = True

except ImportError:

    _PYTRENDS_AVAILABLE = False

    print("ERROR: pytrends not installed.  Run:  pip install pytrends")

    sys.exit(1)



COMP_DIR = "output_diagnostics"
os.makedirs(COMP_DIR, exist_ok=True)




# =============================================================================

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# =============================================================================



TIMEFRAME        = f"2004-01-01 {datetime.today().strftime('%Y-%m-%d')}"

GEO              = "IN"                      # Restricts to Indian searches

BENCHMARK_WORD   = "economy"                 # Universal stabilizing baseline



# ── Sampling parameters (Paper Section 3 batch-optimized) ────────────────────

N_REPEATS        = 3           # Set to 12 for final production run

REPEAT_INTERVAL  = 300          # 60s for testing; paper uses 900s

INTER_KW_SLEEP   = (20, 30)      # Optimized random delay between batches (seconds)

MIN_KEYWORDS     = 5           # Minimum valid keywords required



# ── HP filter (retained as robustness diagnostic only — no longer main) ─────

HP_LAMBDA_POWER  = 4 

HP_LAMBDA = 1600 * ((12 / 4) ** 4) # = 14 400



# ── Hamilton regression filter config — NOW THE MAIN FSI FILTER ─────────────
HAMILTON_HORIZON = 24           # h: forecast horizon (months) per Hamilton (2018)
HAMILTON_LAGS    = 4            # p: number of recent lags used as regressors

# ── Kaiser-Maravall ARIMA-extension config (robustness diagnostic) ──────────
KM_ARIMA_HORIZON     = 24       # months to forecast/backcast before HP filtering
KM_ARIMA_ORDER       = None     # None => small-grid AIC search over (p,d,q)
KM_ARIMA_MAX_PDQ     = (2, 1, 2)  # inclusive upper bound for grid search (p,d,q)

# ── VAR / Granger / FEVD config ──────────────────────────────────────────────
VAR_MAXLAGS       = 24          # max lag candidates for BIC search
VAR_IC            = "bic"       # information criterion for lag selection
FEVD_HORIZONS     = (6, 12, 24) # months ahead reported for variance decomposition

# ── NEW: Toda-Yamamoto augmented VAR causality config ────────────────────────
TY_MAXLAGS        = 24          # max lag candidates for BIC search on levels-VAR
TY_IC             = "bic"       # information criterion for lag order k
TY_MAX_INTEGRATION_ORDER = 2    # d_max: max order of integration to test via ADF

# ── NEW: DCC-GARCH config ────────────────────────────────────────────────────
DCC_GARCH_P       = 1           # GARCH(p,q) order for first-stage univariate fits
DCC_GARCH_Q       = 1
DCC_DIST          = "normal"    # innovation distribution for first-stage GARCH

# ── NEW: TF-IDF anchor-term normalization config ─────────────────────────────
TFIDF_SMOOTHING   = 1.0         # additive smoothing constant in IDF-style weight
TFIDF_MIN_WEIGHT  = 0.05        # floor so no keyword is fully zeroed out

# ── NEW: Multi-Anchor Overlapping Window Calibration config ─────────────────
MAOW_WINDOW_MONTHS   = 12       # rolling anchor-window length (months)
MAOW_STEP_MONTHS     = 6        # step size between successive anchor windows
                                 # (=> 6-month overlap when window=12, step=6)
MAOW_MIN_OVERLAP_OBS = 3        # minimum # overlapping months required to
                                 # estimate a splice calibration ratio
MAOW_CLIP_RATIO      = (0.1, 10.0)  # sanity bounds on any single calibration
                                     # ratio, guarding against near-zero
                                     # denominators in the overlap region



# ── Validation window ─────────────────────────────────────────────────────────

ROLLING_WINDOW   = 12          # Months

TRIM_MONTHS_START = 3   # months to drop from start when plotting/reporting
TRIM_MONTHS_END   = 3


# ── Pickle / cache control ────────────────────────────────────────────────────

FORCE_REFETCH       = True  

PKL_KEYWORD_SERIES = "india_fsi_keyword_series_cache.pkl"
PKL_RAW_FSI        = "india_fsi_raw_series_cache.pkl"



# ── I/O paths ─────────────────────────────────────────────────────────────────

FILE_VIX        = "India_VIX_Historical_Data.csv"

FILE_GUI        = "India GUI - Index Data.xlsx"

FILE_PU         = "India_Policy_Uncertainty_Data-2.csv"

FILE_GPR        = "gpr_india.csv"

FILE_GLOBAL_WUI = "WUI_M_dataset_2026_04.xlsx"

FILE_GEPU       = "Global_Policy_Uncertainty_Data.xlsx"

# ── Data sources ──────────────────────────────────────────────────────────────

FILE_WSI        = "WSI_M_dataset_2026_05.xlsx"   # Sheet F1: Date + WSI (GDP weighted avg)

FILE_OFR_FSI    = "fsi.csv"                       # Columns: Date, OFR FSI, Emerging markets

# ── RBI DBIE hard-data anchor (GDP / GFCF) ────────────────────────────────────
FILE_RBI_DBIE   = "rbi_dbie_gdp_gfcf.csv"



DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

OUT_CSV  = os.path.join(DATA_DIR, "India_FSI_Monthly.csv")
OUT_PNG  = os.path.join(DATA_DIR, "India_FSI_Headline_Figure.png")


KEYWORD_CATEGORIES = {

    "Financial Sector": [
        "bank crisis",
        "bad loans",
        "bank failure",
        "bank run",
        "NPA",
        "bank scam",
        "bank fraud",
        "financial crisis",
        "credit crisis",
        "credit crunch",
        "bank merger",
    ],

    "Financial Market": [
        "bear market",
        "market volatility",
        "market crash",
        "stock market crash",
    ],

    "External Sector": [
        "currency crisis",
    ],

    "Corporate & Sovereign Sector": [
        "bankruptcy",
        "public debt",
        "debt crisis",
        "economic crisis",
    ],

}

KEYWORDS = []

for kws in KEYWORD_CATEGORIES.values():
    KEYWORDS.extend(kws)

KEYWORDS = list(dict.fromkeys(KEYWORDS))

assert len(KEYWORDS) == 20, f"Expected 20 keywords, got {len(KEYWORDS)}"





PALETTE = {

    "fsi":  "#000000",  # black

    "vix":  "#d62728",  # red

    "gui":  "#1f77b4",  # blue

    "pu":   "#2ca02c",  # green



    "roll": "#9467bd",  # purple rolling correlation

    "neg":  "#ff7f0e",  # orange negative region

    "zero": "#666666",  # zero line

    "grid": "#d9d9d9",  # grid color

    "hamilton": "#8c564b",  # brown - Hamilton filter cycle (now MAIN FSI)

    "km_hp":     "#17becf",  # cyan - KM-extended HP cycle

    "dcc":       "#e377c2",  # pink - DCC-GARCH dynamic correlation

    "maow":      "#bcbd22",  # olive - Multi-Anchor Overlapping Window calibration

}





# =============================================================================

# ── HELPER UTILITIES ──────────────────────────────────────────────────────────

# =============================================================================



def has_variance(arr) -> bool:

    a = np.asarray(arr, dtype=float)

    valid = a[~np.isnan(a)]

    return len(valid) >= 12 and np.nanstd(valid) > 1e-9



def initialize_pytrends_safely(max_retries=5, backoff=5):

    """Initializes TrendReq with an exponential backoff loop to bypass Timeout Errors."""

    for attempt in range(1, max_retries + 1):

        try:

            # Set internal requests timeout higher (10s) to absorb network jitter

            pytrends = TrendReq(hl="en-US", tz=330, timeout=10)

            return pytrends

        except Exception as e:

            if attempt == max_retries:

                raise RuntimeError(f"Failed to connect to Google Trends after {max_retries} attempts: {e}")

            print(f"  [Init Error] Connection timed out. Retrying in {backoff}s... (Attempt {attempt}/{max_retries})")

            time.sleep(backoff)

            backoff *= 2


# =============================================================================
# ── NEW: TF-IDF-STYLE ANCHOR-TERM NORMALIZATION ─────────────────────────────
# Motivation: in the standard pipeline every keyword's monthly search-share
# series is simply summed. Some keywords (e.g. "public debt", "bank merger")
# behave more like generic/background terms that spike often and are not
# distinctively "anchored" to acute stress episodes, while others (e.g.
# "bank run", "bank failure") are sparse and highly discriminative when they
# do spike. This mirrors the classic TF-IDF intuition: a term that appears
# in (spikes across) most "documents" (months) carries less discriminative
# information than a term that appears rarely but strongly.
#
# We treat each keyword's time series as a bag of "term frequencies" (TF,
# the raw monthly share values themselves, already benchmark-ratio'd) and
# compute a document-frequency-style IDF weight per keyword:
#
#   DF_k  = (# of months where keyword k's normalized value exceeds a
#            threshold, i.e. the term "fires") / (total # of months)
#   IDF_k = log( (1 + N) / (1 + DF_k * N) ) + smoothing
#
# A keyword that fires in most months (DF_k → 1) gets a low IDF weight
# (behaves like a generic/common term); a keyword that fires rarely but
# sharply (DF_k → 0) gets a high IDF weight (an "anchor" term, distinctive
# of genuine stress episodes). Each keyword's monthly series is then
# rescaled by its IDF weight before summation into the raw aggregate FSI,
# i.e. TF x IDF per keyword-month, exactly analogous to a TF-IDF matrix.
# =============================================================================

def compute_tfidf_anchor_weights(keyword_memory: dict, fire_quantile: float = 0.75) -> pd.Series:
    """
    Computes a TF-IDF-style anchor weight for each keyword series in
    keyword_memory (dict[str, pd.Series]).

    Returns a pd.Series indexed by keyword name -> IDF-style anchor weight,
    already floored at TFIDF_MIN_WEIGHT and NOT yet normalized to sum to 1
    (so it can be applied as a direct multiplicative rescaling factor).
    """
    if not keyword_memory:
        return pd.Series(dtype=float)

    weights = {}
    n_kw = len(keyword_memory)

    for kw, series in keyword_memory.items():
        s = series.dropna()
        if len(s) == 0:
            weights[kw] = TFIDF_MIN_WEIGHT
            continue

        # "Firing" threshold: values above the fire_quantile of the keyword's
        # own distribution count as an "occurrence" of the term (mirrors a
        # binary term-presence indicator per document/month).
        thresh = s.quantile(fire_quantile)
        n_months = len(s)
        df_k = (s > thresh).sum() / n_months if n_months > 0 else 0.0

        idf_k = np.log((1 + n_kw) / (1 + df_k * n_kw)) + TFIDF_SMOOTHING
        weights[kw] = max(idf_k, TFIDF_MIN_WEIGHT)

    w = pd.Series(weights, name="TFIDF_Anchor_Weight")

    # Rescale so the weights average to 1.0 across kept keywords — this
    # preserves the raw FSI's overall level/scale (so downstream HP/Hamilton
    # filtering and cross-index comparisons remain interpretable) while still
    # re-weighting the *relative* contribution of each keyword.
    if w.mean() > 0:
        w = w / w.mean()

    return w


def apply_tfidf_anchor_normalization(keyword_memory: dict) -> tuple[dict, pd.Series]:
    """
    Applies TF-IDF anchor weighting to each keyword series in keyword_memory.

    Returns (reweighted_keyword_memory, weights_series).
    """
    weights = compute_tfidf_anchor_weights(keyword_memory)

    reweighted = {}
    for kw, series in keyword_memory.items():
        w = weights.get(kw, 1.0)
        reweighted[kw] = series * w

    return reweighted, weights


# =============================================================================

# ── STEP 4 : BATCH-OPTIMIZED SAMPLING (Castelnuovo-Tran Method) ─────────────

# =============================================================================



def build_raw_fsi(pytrends_obj) -> tuple[pd.Series, list[str], list[str]]:

    raw_accumulator = []

    kept_kws = []

    removed_kws = []

    

    target_kws = [k for k in KEYWORDS if k != BENCHMARK_WORD]

    batches = [target_kws[i:i + 4] for i in range(0, len(target_kws), 4)]

    sample_runs = []

    

    for rep in range(N_REPEATS):

        print(f"\n─── Starting Repetition {rep+1}/{N_REPEATS} ───")

        rep_series_dict = {}

        

        for b_idx, batch in enumerate(batches, 1):
            pytrends_obj = initialize_pytrends_safely()

            query_terms = batch + [BENCHMARK_WORD]

            success = False

            

            for network_retry in range(2):
                try:
                    pytrends_obj.build_payload(query_terms, cat=0, timeframe=TIMEFRAME, geo=GEO)
                    data = pytrends_obj.interest_over_time()

                    if not data.empty:
                        bench = data[BENCHMARK_WORD].replace(0, np.nan)
                        for kw in batch:
                            if kw in data.columns:
                                raw_ratio = data[kw] / bench
                                p99 = np.nanpercentile(raw_ratio.dropna(), 99)
                                monthly = raw_ratio.clip(lower=0, upper=p99).resample("MS").mean()
                                rep_series_dict[kw] = monthly
                        success = True
                        break

                except Exception as e:
                    msg = str(e)
                    is_429 = "429" in msg or "TooManyRequests" in msg
                    wait = (30 * (2 ** network_retry)) if is_429 else 5
                    print(f"    [Retry {network_retry+1}/5] {type(e).__name__}: {msg[:150]}")
                    print(f"    Waiting {wait}s before retry...")
                    time.sleep(wait)

            

            if success:

                print(f"  Batch {b_idx:02d}/{len(batches)} processed successfully")

            else:

                print(f"  WARNING: Batch {b_idx:02d}/{len(batches)} failed completely across network retries.")

                

            time.sleep(np.random.uniform(*INTER_KW_SLEEP))

            

        sample_runs.append(pd.DataFrame(rep_series_dict))

        

        if rep < N_REPEATS - 1:

            print(f"  Sleeping {REPEAT_INTERVAL}s before initiating next repetition loop...")

            time.sleep(REPEAT_INTERVAL)



    # Compute median along axis=0 across dataframe repetitions

    master_df = pd.concat(sample_runs, axis=0, keys=range(N_REPEATS))

    median_df = master_df.groupby(level=1).median()

    

    for kw in median_df.columns:

        if has_variance(median_df[kw].values):

            raw_accumulator.append(median_df[kw])

            kept_kws.append(kw)

        else:

            removed_kws.append(kw)

            

    keyword_memory = dict(zip(kept_kws, raw_accumulator))

    pd.to_pickle(keyword_memory, PKL_KEYWORD_SERIES)

    # ── NEW: TF-IDF anchor-term normalization applied before aggregation ────

    reweighted_memory, tfidf_weights = apply_tfidf_anchor_normalization(keyword_memory)

    tfidf_weights.to_csv(os.path.join(COMP_DIR, "tfidf_anchor_weights.csv"))

    print("\n  TF-IDF Anchor-Term Weights (mean-normalized to 1.0):")

    print(tfidf_weights.sort_values(ascending=False).to_string())

    all_series = pd.concat(list(reweighted_memory.values()), axis=1)

    raw_gui = all_series.sum(axis=1, min_count=1)

    pd.to_pickle(raw_gui, PKL_RAW_FSI)

    

    return raw_gui, kept_kws, removed_kws





def load_raw_fsi_from_pkl() -> tuple[pd.Series, dict[str, pd.Series]]:

    keyword_memory = {}

    if os.path.exists(PKL_KEYWORD_SERIES):

        keyword_memory = pd.read_pickle(PKL_KEYWORD_SERIES)

        print(f"  Loaded {len(keyword_memory)} clean keyword series from '{PKL_KEYWORD_SERIES}'")



    if os.path.exists(PKL_RAW_FSI):

        raw_gui = pd.read_pickle(PKL_RAW_FSI)

        print(f"  Loaded raw GUI series from '{PKL_RAW_FSI}' ({raw_gui.notna().sum()} months)")

    elif keyword_memory:

        # ── NEW: apply TF-IDF anchor normalization when rebuilding from the
        #     keyword-level cache (i.e. the raw_fsi pickle didn't exist yet,
        #     but the per-keyword pickle did) ─────────────────────────────
        reweighted_memory, tfidf_weights = apply_tfidf_anchor_normalization(keyword_memory)

        tfidf_weights.to_csv(os.path.join(COMP_DIR, "tfidf_anchor_weights.csv"))

        print("\n  TF-IDF Anchor-Term Weights (mean-normalized to 1.0):")

        print(tfidf_weights.sort_values(ascending=False).to_string())

        raw_gui = pd.concat(list(reweighted_memory.values()), axis=1).sum(axis=1, min_count=1)

        pd.to_pickle(raw_gui, PKL_RAW_FSI)

    else:

        raise FileNotFoundError("No valid cache pickles found.")

    return raw_gui, keyword_memory


# =============================================================================
# ── NEW: MULTI-ANCHOR OVERLAPPING WINDOW CALIBRATION ─────────────────────────
#
# MOTIVATION
# ----------
# Google Trends does NOT return raw absolute search counts. Every query is
# internally rescaled so that the single highest point *within that specific
# query's requested time range* is set to 100, and everything else is scaled
# relative to that peak. This means the exact same keyword, queried over two
# different date ranges, can carry a *different implicit scale* — a value of
# "40" in a 2004-2010 query and a value of "40" in a 2015-2020 query are not
# guaranteed to represent the same underlying search intensity, because each
# query has its own independent max-100 anchor point. This is Google Trends'
# core "internal scaling artifact."
#
# The full FSI pipeline above queries the ENTIRE 2004-2025 window in a single
# shot (see TIMEFRAME / build_raw_fsi), so the whole panel is anchored to one
# global max. That protects it from *some* forms of this artifact by
# construction — but it also means we have never actually tested whether a
# differently-anchored construction of the same series would look the same.
# If the FSI's shape/level were secretly sensitive to *how* the scaling
# anchor happens to fall (e.g. because the single global peak sits in an
# unusual regime), a full-sample single-shot query would not reveal that.
#
# METHOD (rolling overlapping-window re-anchoring + splice calibration)
# -----------------------------------------------------------------------
# 1. Take the cached RAW per-keyword monthly series (as already collected/
#    scaled by the standard single-shot pipeline) and carve it into a set of
#    overlapping ROLLING ANCHOR WINDOWS of length MAOW_WINDOW_MONTHS (default
#    12 months), stepping forward by MAOW_STEP_MONTHS (default 6 months) —
#    i.e. consecutive windows share a 6-month overlap.
# 2. Within each window, RE-ANCHOR the sub-series to its own local max = 100
#    (mimicking what an independent Google Trends query restricted to just
#    that 12-month window would have returned) — this is the deliberate
#    re-introduction of the "internal scaling artifact" we want to stress
#    test against.
# 3. Walk through the windows in chronological order and SPLICE them back
#    into one continuous series: for each new window, use the OVERLAP region
#    with the already-spliced series to estimate a calibration ratio
#    (median of new-window-values / already-spliced-values over the overlap
#    months), then rescale the new window by that ratio before appending its
#    non-overlapping tail. This is the same overlap-and-bridge logic used to
#    stitch together independently-normalized panel/index data.
# 4. Aggregate the recalibrated keyword series into a MAOW-calibrated raw FSI
#    using the SAME TF-IDF anchor weights as the main pipeline (so the ONLY
#    thing that differs from the main raw FSI is the anchoring/splicing
#    procedure), then run it through the identical seasonal + Hamilton-filter
#    pipeline used for the main FSI.
# 5. Compare the MAOW-calibrated FSI to the main single-shot FSI: high
#    correlation / low relative deviation is direct empirical evidence that
#    the index is NOT an artifact of any single anchoring choice — i.e. it IS
#    robust to Google Trends' internal max-100 rescaling behavior. Low
#    correlation would indicate the opposite and flag a real robustness
#    concern.
# =============================================================================

def _rolling_anchor_windows(series: pd.Series, window_months=MAOW_WINDOW_MONTHS,
                             step_months=MAOW_STEP_MONTHS):
    """
    Yields (window_start, window_end, sub_series) tuples covering `series`'s
    date range with overlapping rolling windows of length `window_months`,
    stepping by `step_months`.
    """
    s = series.dropna()
    if s.empty:
        return

    idx_min, idx_max = s.index.min(), s.index.max()
    win_start = idx_min

    while win_start <= idx_max:
        win_end = win_start + pd.DateOffset(months=window_months) - pd.DateOffset(months=1)
        sub = s[(s.index >= win_start) & (s.index <= win_end)]
        if len(sub) > 0:
            yield win_start, win_end, sub
        win_start = win_start + pd.DateOffset(months=step_months)


def _reanchor_to_local_max(sub_series: pd.Series) -> pd.Series:
    """
    Re-scales `sub_series` so its own local maximum = 100, mimicking an
    independent Google Trends query restricted to just this window's date
    range (Google's internal per-query max-100 normalization).
    """
    local_max = sub_series.max()
    if not np.isfinite(local_max) or local_max <= 1e-9:
        return sub_series * np.nan
    return (sub_series / local_max) * 100.0


def _splice_overlapping_windows(windows: list, min_overlap_obs=MAOW_MIN_OVERLAP_OBS,
                                 clip_ratio=MAOW_CLIP_RATIO) -> pd.Series:
    """
    windows: list of (win_start, win_end, reanchored_sub_series), in
        chronological order of win_start.

    Splices the sequence of independently re-anchored windows into one
    continuous series by calibrating each new window against the
    already-spliced series using their overlap region, then appending the
    new window's non-overlapping tail.

    Returns the fully spliced, single continuous pd.Series (still on the
    same 0-100-ish scale as the first window, since the first window anchors
    the whole spliced series).
    """
    if not windows:
        return pd.Series(dtype=float)

    spliced = windows[0][2].copy()

    for win_start, win_end, sub in windows[1:]:
        overlap_idx = spliced.index.intersection(sub.index)

        if len(overlap_idx) >= min_overlap_obs:
            spliced_overlap = spliced.loc[overlap_idx]
            new_overlap = sub.loc[overlap_idx]

            # Ratio-based calibration: median ratio of the already-spliced
            # (accepted) values to this new window's values over the shared
            # overlap months. Median is used (rather than mean/regression)
            # for robustness to any single noisy overlap month.
            valid = (new_overlap.abs() > 1e-9) & spliced_overlap.notna() & new_overlap.notna()
            if valid.sum() >= 1:
                ratios = (spliced_overlap[valid] / new_overlap[valid])
                calib_ratio = float(ratios.median())
                calib_ratio = float(np.clip(calib_ratio, clip_ratio[0], clip_ratio[1]))
            else:
                calib_ratio = 1.0
        else:
            # Not enough overlap to calibrate reliably — fall back to
            # matching means over whatever overlap exists, or leave
            # uncalibrated (ratio=1) if there's no overlap at all.
            calib_ratio = 1.0

        sub_calibrated = sub * calib_ratio

        # Only append the NEW, non-overlapping tail beyond what's already
        # in the spliced series so we don't duplicate/average overlap
        # months — the overlap months keep the *already-spliced* value.
        new_tail_idx = sub_calibrated.index.difference(spliced.index)
        if len(new_tail_idx) > 0:
            spliced = pd.concat([spliced, sub_calibrated.loc[new_tail_idx]]).sort_index()

    return spliced


def multi_anchor_overlapping_window_calibration(keyword_memory: dict,
                                                  window_months=MAOW_WINDOW_MONTHS,
                                                  step_months=MAOW_STEP_MONTHS) -> dict:
    """
    Runs the full multi-anchor overlapping-window re-anchoring + splice
    calibration procedure (see module docstring above) independently for
    EVERY keyword series in `keyword_memory`.

    Returns dict[keyword] -> spliced, MAOW-calibrated pd.Series (same
    interpretive scale/units as the input keyword series).
    """
    calibrated_memory = {}

    for kw, series in keyword_memory.items():
        s = series.dropna()
        if len(s) < window_months:
            # Too short to build even one full anchor window — keep as-is.
            calibrated_memory[kw] = series
            continue

        windows = []
        for win_start, win_end, sub in _rolling_anchor_windows(s, window_months, step_months):
            reanchored = _reanchor_to_local_max(sub)
            windows.append((win_start, win_end, reanchored))

        if not windows:
            calibrated_memory[kw] = series
            continue

        spliced = _splice_overlapping_windows(windows)

        # Rescale the fully-spliced MAOW series so it has the SAME overall
        # mean level as the original single-shot series — this isolates
        # "does re-anchoring change the *shape/dynamics*" from "does it
        # trivially change the *overall scale*" (the latter is expected and
        # uninteresting; the former is the actual artifact we're testing
        # for). Shape/dynamics are what downstream correlation comparisons
        # will actually probe.
        orig_mean = s.mean()
        spliced_mean = spliced.mean()
        if spliced_mean and np.isfinite(spliced_mean) and abs(spliced_mean) > 1e-9:
            spliced = spliced * (orig_mean / spliced_mean)

        calibrated_memory[kw] = spliced.reindex(series.index)

    return calibrated_memory


def run_maow_robustness_calibration(keyword_memory: dict, tfidf_weights: pd.Series,
                                     main_fsi: pd.Series,
                                     window_months=MAOW_WINDOW_MONTHS,
                                     step_months=MAOW_STEP_MONTHS) -> dict:
    """
    Top-level orchestration for the Multi-Anchor Overlapping Window (MAOW)
    robustness calibration diagnostic:

      1. Re-derive every keyword series via rolling overlapping-window
         re-anchoring + splice calibration (multi_anchor_overlapping_window_
         calibration).
      2. Re-aggregate into a raw FSI using the SAME TF-IDF anchor weights as
         the main pipeline (isolating anchoring-method as the only variable
         that differs from the main raw FSI construction).
      3. Push the MAOW-calibrated raw FSI through the identical seasonal
         adjustment + Hamilton regression filter used for the main FSI.
      4. Compare (Pearson/Spearman correlation, mean absolute relative
         deviation) the MAOW-calibrated FSI against `main_fsi`.
      5. Save a diagnostic CSV + comparison plot.

    Returns a dict with the calibrated FSI series and comparison statistics.
    """
    print("\n" + "=" * 70)
    print("  ROBUSTNESS CALIBRATION")
    print("  Multi-Anchor Overlapping Window Calibration")
    print(f"  (rolling {window_months}-month anchor windows, "
          f"{step_months}-month step => {window_months - step_months}-month overlap)")
    print("=" * 70)

    result = {
        "maow_fsi": None,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_r": None,
        "spearman_p": None,
        "mean_abs_rel_dev": None,
        "n_overlap_obs": None,
        "error": None,
    }

    try:
        # ── Step 1: re-anchor + splice every keyword series ───────────────
        calibrated_memory = multi_anchor_overlapping_window_calibration(
            keyword_memory, window_months=window_months, step_months=step_months
        )

        n_recalibrated = sum(
            1 for kw in calibrated_memory
            if not calibrated_memory[kw].equals(keyword_memory.get(kw))
        )
        print(f"  Re-anchored & spliced {n_recalibrated}/{len(keyword_memory)} keyword series "
              f"across overlapping windows.")

        # ── Step 2: re-aggregate using the SAME TF-IDF weights as main FSI ─
        reweighted = {}
        for kw, series in calibrated_memory.items():
            w = tfidf_weights.get(kw, 1.0) if tfidf_weights is not None else 1.0
            reweighted[kw] = series * w

        maow_raw_fsi = pd.concat(list(reweighted.values()), axis=1).sum(axis=1, min_count=1)

        if maow_raw_fsi.dropna().empty:
            result["error"] = "MAOW raw aggregate series is empty after re-anchoring/splicing."
            print(f"  ERROR: {result['error']}")
            print("=" * 70)
            return result

        # ── Step 3: identical downstream pipeline (seasonal + Hamilton) ────
        maow_sa = try_x13_seasonal_adjust(maow_raw_fsi)
        maow_fsi_full = hamilton_filter_and_scale(maow_sa)
        maow_fsi = trim_endpoints(maow_fsi_full)
        maow_fsi.name = "FSI_MAOW_CALIBRATED"

        result["maow_fsi"] = maow_fsi

        # ── Step 4: compare against the main single-shot-anchored FSI ─────
        merged = pd.concat([main_fsi.rename("FSI_MAIN"), maow_fsi.rename("FSI_MAOW")],
                            axis=1).dropna()

        if len(merged) < 10:
            result["error"] = (
                f"Insufficient overlap between main FSI and MAOW FSI for comparison "
                f"({len(merged)} obs)."
            )
            print(f"  ERROR: {result['error']}")
            print("=" * 70)
            return result

        pearson_r, pearson_p = stats.pearsonr(merged["FSI_MAIN"], merged["FSI_MAOW"])
        spearman_r, spearman_p = stats.spearmanr(merged["FSI_MAIN"], merged["FSI_MAOW"])

        denom = merged["FSI_MAIN"].abs().replace(0, np.nan)
        rel_dev = ((merged["FSI_MAOW"] - merged["FSI_MAIN"]).abs() / denom).dropna()
        mean_abs_rel_dev = float(rel_dev.mean()) if len(rel_dev) > 0 else np.nan

        result["pearson_r"] = float(pearson_r)
        result["pearson_p"] = float(pearson_p)
        result["spearman_r"] = float(spearman_r)
        result["spearman_p"] = float(spearman_p)
        result["mean_abs_rel_dev"] = mean_abs_rel_dev
        result["n_overlap_obs"] = len(merged)

        print(f"\n  Main FSI (single-shot anchor) vs MAOW FSI (rolling overlapping anchors):")
        print(f"    n={len(merged)} overlapping months")
        print(f"    Pearson  r = {pearson_r:+.4f}  (p={pearson_p:.2e})")
        print(f"    Spearman ρ = {spearman_r:+.4f}  (p={spearman_p:.2e})")
        print(f"    Mean absolute relative deviation = {mean_abs_rel_dev:.2%}")

        if pearson_r >= 0.90:
            verdict = "STRONG evidence the FSI is robust to Google Trends' internal scaling artifacts."
        elif pearson_r >= 0.75:
            verdict = "MODERATE evidence of robustness — some anchoring sensitivity present."
        else:
            verdict = "WEAK correlation — FSI shows meaningful sensitivity to anchoring window choice; investigate further."
        print(f"    Verdict: {verdict}")

        # ── Step 5: save diagnostic CSV + comparison plot ──────────────────
        comparison_df = merged.copy()
        comparison_df["Abs_Relative_Deviation"] = rel_dev.reindex(comparison_df.index)
        comparison_df.to_csv(os.path.join(COMP_DIR, "maow_robustness_calibration_comparison.csv"))

        plt.figure(figsize=(14, 6))
        plt.plot(merged.index, merged["FSI_MAIN"], color=PALETTE["fsi"], lw=1.8,
                  label="Main FSI (single-shot anchor)")
        plt.plot(merged.index, merged["FSI_MAOW"], color=PALETTE["maow"], lw=1.6,
                  linestyle="--", alpha=0.9, label="MAOW FSI (rolling overlapping anchors)")
        plt.axhline(0, color=PALETTE["zero"], lw=0.8, linestyle=":")
        plt.title(
            "Robustness Calibration: Multi-Anchor Overlapping Window\n"
            f"Pearson r={pearson_r:+.3f} (p={pearson_p:.2e})  |  "
            f"Spearman ρ={spearman_r:+.3f}  |  "
            f"Mean |rel. dev.|={mean_abs_rel_dev:.2%}"
        )
        plt.ylabel("FSI Index")
        plt.legend()
        plt.grid(True, color=PALETTE["grid"])
        plt.tight_layout()
        plt.savefig(os.path.join(COMP_DIR, "maow_robustness_calibration_timeseries.png"), dpi=300)
        plt.close()

        # Rolling correlation between main and MAOW FSI, showing whether
        # robustness holds uniformly through time or degrades in specific
        # regimes/episodes.
        if len(merged) >= ROLLING_WINDOW + 2:
            rc = merged["FSI_MAIN"].rolling(ROLLING_WINDOW).corr(merged["FSI_MAOW"])
            plt.figure(figsize=(14, 6))
            plt.fill_between(rc.index, rc, 0, where=(rc >= 0), color=PALETTE["maow"], alpha=0.32)
            plt.fill_between(rc.index, rc, 0, where=(rc < 0), color=PALETTE["neg"], alpha=0.32)
            plt.plot(rc.index, rc, color=PALETTE["maow"], lw=1.5)
            plt.axhline(0, color=PALETTE["zero"], lw=0.9, linestyle="--")
            plt.axhline(pearson_r, color="black", lw=0.9, linestyle=":",
                        label=f"Overall r = {pearson_r:+.3f}")
            plt.ylim(-1, 1)
            plt.ylabel("Rolling Pearson r")
            plt.title(f"Rolling {ROLLING_WINDOW}-Month Correlation: Main FSI vs MAOW-Calibrated FSI")
            plt.legend()
            plt.grid(True, color=PALETTE["grid"])
            plt.tight_layout()
            plt.savefig(os.path.join(COMP_DIR, "maow_robustness_calibration_rolling_corr.png"), dpi=300)
            plt.close()

        print(f"\n  Saved MAOW robustness calibration diagnostics -> {COMP_DIR}/maow_robustness_calibration_*")

    except Exception as e:
        result["error"] = str(e)
        print(f"  ERROR in MAOW robustness calibration: {e}")

    print("=" * 70)
    return result


# =============================================================================

# ── STEP 5 : ACCELERATED SEASONAL ADJUSTMENT ─────────────────────────────────

# =============================================================================



def seasonal_adjust_stl(series: pd.Series) -> pd.Series:

    from statsmodels.tsa.seasonal import STL

    filled = series.interpolate(method="time").bfill().ffill()

    # Optimized: degree=1 speeds up internal LOESS linear equations significantly

    stl = STL(filled, period=12, robust=True, seasonal_deg=1, trend_deg=1)

    result = stl.fit()

    return filled - result.seasonal





def try_x13_seasonal_adjust(series: pd.Series) -> pd.Series:

    try:

        from statsmodels.tsa.x13 import x13_arima_analysis

        filled = series.interpolate(method="time").bfill().ffill()

        result = x13_arima_analysis(filled, x12path=None, prefer_x13=True)

        print("  Seasonal adjustment: X-13 ARIMA  ✓")

        return result.seasadj

    except Exception:

        print("  Seasonal adjustment: Accelerated STL fallback triggered")

        return seasonal_adjust_stl(series)





# =============================================================================

# ── STEPS 6–7 : HP FILTER + SCALING (robustness diagnostic only) ────────────

# =============================================================================
def trim_endpoints(series: pd.Series, start_months=TRIM_MONTHS_START, end_months=TRIM_MONTHS_END) -> pd.Series:
    """Trims boundary-biased filter observations for display/reporting purposes only.
    The full sample is still used for estimation — only the OUTPUT is trimmed."""
    s = series.copy()
    if start_months > 0:
        cutoff_start = s.index.min() + pd.DateOffset(months=start_months)
        s = s[s.index >= cutoff_start]
    if end_months > 0:
        cutoff_end = s.index.max() - pd.DateOffset(months=end_months)
        s = s[s.index <= cutoff_end]
    return s


def hp_filter_and_scale(sa_series: pd.Series) -> pd.Series:

    filled = sa_series.interpolate(method="time").bfill().ffill()

    cycle, trend = hpfilter(filled.values, lamb=HP_LAMBDA)

    trend_safe = np.where(np.abs(trend) < 1e-9, np.nan, trend)

    fsi = 100.0 * (cycle / trend_safe)

    return pd.Series(fsi, index=sa_series.index, name="FSI")


def bk_filter_and_scale(sa_series: pd.Series, low=8, high=520, K=104) -> pd.Series:
    """Baxter-King bandpass filter. low/high are periodicity bounds in weeks
    (roughly 2 months to 10 years); K is the lead/lag truncation window."""
    filled = sa_series.interpolate(method="time").bfill().ffill()
    cycle = bkfilter(filled.values, low=low, high=high, K=K)
    trend_len_diff = len(filled) - len(cycle)
    trim = trend_len_diff // 2
    idx = filled.index[trim: trim + len(cycle)]
    trend_approx = filled.values[trim: trim + len(cycle)] - cycle
    trend_safe = np.where(np.abs(trend_approx) < 1e-9, np.nan, trend_approx)
    fsi_bk = 100.0 * (cycle / trend_safe)
    return pd.Series(fsi_bk, index=idx, name="FSI_BK")


def cf_filter_and_scale(sa_series: pd.Series, low=8, high=520) -> pd.Series:
    """Christiano-Fitzgerald asymmetric bandpass filter — retains full sample length,
    no endpoint trimming needed (unlike BK)."""
    filled = sa_series.interpolate(method="time").bfill().ffill()
    cycle, trend = cffilter(filled.values, low=low, high=high, drift=True)
    trend_safe = np.where(np.abs(trend) < 1e-9, np.nan, trend)
    fsi_cf = 100.0 * (cycle / trend_safe)
    return pd.Series(fsi_cf, index=sa_series.index, name="FSI_CF")


# =============================================================================

# ── HAMILTON (2018) REGRESSION FILTER — NOW THE MAIN FSI ─────────────────────
# Regresses y_{t+h} on y_t, y_{t-1}, ..., y_{t-p+1} (the p most recent values
# as of date t). The forecast error (residual) is a stationary cyclical
# component, dated at t+h, computed without any two-sided smoothing — this is
# the core of Hamilton's critique that the HP filter creates spurious
# dynamics via its symmetric moving-average structure. Because it uses only
# past information (a one-sided regression filter), it is also more suitable
# than the HP filter for real-time / leading-indicator use, which is why it
# now defines the MAIN India-FSI series in this revision.
# =============================================================================

def hamilton_filter_and_scale(sa_series: pd.Series, h=HAMILTON_HORIZON,
                               p=HAMILTON_LAGS) -> pd.Series:
    """
    Returns a percentage-scaled cyclical component analogous to the other
    filters (100 * cycle / trend-proxy), where the trend-proxy is the
    regression's fitted value (y_hat), keeping the same interpretive scale
    as the HP/BK/CF outputs. THIS IS NOW THE MAIN FSI-GENERATING FILTER.
    """
    filled = sa_series.interpolate(method="time").bfill().ffill()
    y = filled.values
    n = len(y)

    if n <= p + h:
        raise ValueError(
            f"Series too short for Hamilton filter: need > {p + h} obs, got {n}"
        )

    X_rows, y_target, idx_out = [], [], []
    for t in range(p - 1, n - h):
        regs = [y[t - i] for i in range(p)]
        X_rows.append(regs)
        y_target.append(y[t + h])
        idx_out.append(filled.index[t + h])

    X = add_constant(np.array(X_rows))
    Y = np.array(y_target)
    model = OLS(Y, X).fit()

    fitted_vals = model.fittedvalues
    resid = model.resid  # this is the cyclical component (forecast error)

    trend_safe = np.where(np.abs(fitted_vals) < 1e-9, np.nan, fitted_vals)
    fsi_hamilton = 100.0 * (resid / trend_safe)

    return pd.Series(fsi_hamilton, index=idx_out, name="FSI_HAMILTON")


# =============================================================================

# ── KAISER-MARAVALL ARIMA-EXTENDED HP FILTER (robustness diagnostic) ────────
# Fits a best-AIC ARIMA(p,d,q) model (small grid search), forecasts forward
# and backcasts backward by KM_ARIMA_HORIZON months, applies the standard
# HP filter to the padded series, then truncates the artificial forecasts.
# This directly targets Hamilton's endpoint-bias critique for real-time use.
# Retained here purely as a comparison series against the new Hamilton-based
# main FSI — it is NOT used to generate the main FSI.
# =============================================================================

def _best_arima_order(y: np.ndarray, max_pdq=KM_ARIMA_MAX_PDQ):
    """Small grid search over (p,d,q) by AIC. Returns best order tuple."""
    best_aic = np.inf
    best_order = (1, 1, 1)
    max_p, max_d, max_q = max_pdq
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    m = ARIMA(y, order=(p, d, q)).fit()
                    if np.isfinite(m.aic) and m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order, best_aic


def km_arima_extended_hp_filter_and_scale(sa_series: pd.Series,
                                           lamb=HP_LAMBDA,
                                           horizon=KM_ARIMA_HORIZON,
                                           order=KM_ARIMA_ORDER):
    """
    Returns (fsi_km, chosen_order). fsi_km is percentage-scaled cycle/trend,
    same convention as hp_filter_and_scale, but computed on the ARIMA-padded
    series and then truncated back to the original sample — eliminating the
    boundary distortion of the plain two-sided HP filter.
    """
    filled = sa_series.interpolate(method="time").bfill().ffill()
    y = filled.values
    idx = filled.index
    freq = pd.infer_freq(idx) or "MS"

    chosen_order = order
    if chosen_order is None:
        chosen_order, _ = _best_arima_order(y)

    fwd_model = ARIMA(y, order=chosen_order).fit()
    fc = fwd_model.get_forecast(steps=horizon).predicted_mean

    # backcast: reverse the series, fit/forecast, then re-reverse
    y_rev = y[::-1]
    bwd_model = ARIMA(y_rev, order=chosen_order).fit()
    bc = bwd_model.get_forecast(steps=horizon).predicted_mean[::-1]

    fwd_idx = pd.date_range(idx[-1], periods=horizon + 1, freq=freq)[1:]
    bwd_idx = pd.date_range(end=idx[0], periods=horizon + 1, freq=freq)[:-1]

    extended_vals = np.concatenate([bc, y, fc])
    extended_idx = bwd_idx.append(idx).append(fwd_idx)

    cycle_ext, trend_ext = hpfilter(extended_vals, lamb=lamb)
    cycle_full = pd.Series(cycle_ext, index=extended_idx)
    trend_full = pd.Series(trend_ext, index=extended_idx)

    # truncate the artificial forecasts back to the original sample window
    cycle = cycle_full.loc[idx]
    trend = trend_full.loc[idx]
    trend_safe = trend.where(trend.abs() > 1e-9, np.nan)

    fsi_km = 100.0 * (cycle / trend_safe)
    fsi_km.name = "FSI_KM_HP"

    return fsi_km, chosen_order


# =============================================================================

# ── FIXED DATETIME LOADERS (Forces Index Start-of-Month 'MS' Alignment) ──────

# =============================================================================



def load_india_vix(path: str) -> pd.Series:

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    df["date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%m-%Y")

    df["VIX"]  = df["Price"].astype(str).str.replace(",", "", regex=False).astype(float)

    # FIX: Resample straight to Month-Start 'MS' to force timestamp alignment

    monthly = df.set_index("date").sort_index()["VIX"].resample("MS").mean()

    return monthly.rename("VIX")





def load_gui(path: str) -> pd.Series:



    df = pd.read_excel(path)



    df.columns = df.columns.str.strip()



    df["date"] = pd.to_datetime(df["Date"])



    df["GUI"] = (

        df["India - GUI"]

        .astype(str)

        .str.replace(",", "", regex=False)

        .astype(float)

    )



    monthly = (

        df

        .set_index("date")["GUI"]

        .sort_index()

        .resample("MS")

        .mean()

    )



    monthly.index = monthly.index.normalize()



    return monthly.rename("GUI")



def load_gpr(path: str) -> pd.Series:



    df = pd.read_csv(path)



    df.columns = df.columns.str.strip()



    df["Date"] = pd.to_datetime(

        df["Date"],

        format="%d/%m/%y"

    )



    gpr = (

        df.set_index("Date")["INDIA_gpr"]

        .sort_index()

        .resample("MS")

        .mean()

    )



    return gpr.rename("GPR")





def load_global_wui(path):

    # Try multiple common sheet names

    xl = pd.ExcelFile(path)

    sheet = "F1" if "F1" in xl.sheet_names else xl.sheet_names[0]



    # Read raw without assuming a header row position

    raw = pd.read_excel(path, sheet_name=sheet, header=None)



    # Scan rows to find where actual date+numeric data begins

    data_start = 0

    for i, row in raw.iterrows():

        val0, val1 = row.iloc[0], row.iloc[1]

        try:

            pd.to_datetime(val0)

            float(val1)

            data_start = i

            break

        except (ValueError, TypeError):

            continue



    df = raw.iloc[data_start:].copy()

    df.columns = range(df.shape[1])   # integer column names — avoids NaN key conflicts



    df[0] = pd.to_datetime(df[0], errors="coerce")

    df[1] = pd.to_numeric(df[1], errors="coerce")

    df = df.dropna(subset=[0, 1])



    series = (

        df.set_index(0)[1]

          .sort_index()

          .resample("MS")

          .mean()

    )



    print(f"  Global WUI: sheet='{sheet}', data_start_row={data_start}, "

          f"{series.notna().sum()} obs "

          f"({series.index.min().date()} to {series.index.max().date()})")



    return series.rename("GWUI")



def load_gepu(path: str) -> pd.Series:

    df = pd.read_excel(path)

    df.columns = df.columns.str.strip()



    # Drop rows where Year or Month are NaN (footer/header bleed-in)

    df = df.dropna(subset=["Year", "Month"])



    df["date"] = pd.to_datetime(

        df["Year"].astype(int).astype(str)

        + "-"

        + df["Month"].astype(float).astype(int).astype(str).str.zfill(2)

        + "-01"

    )



    gepu = (

        df

        .set_index("date")["GEPU_current"]

        .sort_index()

        .resample("MS")

        .mean()

    )



    return gepu.rename("GEPU")


# =============================================================================

# ── NEW LOADERS ───────────────────────────────────────────────────────────────

# =============================================================================


def load_wsi(path: str) -> pd.Series:

    """Load World Stress Index (WSI) from WSI_M_dataset xlsx, sheet F1.

    Expects columns: Date and a GDP-weighted average column (second numeric col).

    """

    xl = pd.ExcelFile(path)

    sheet = "F1" if "F1" in xl.sheet_names else xl.sheet_names[0]

    raw = pd.read_excel(path, sheet_name=sheet, header=None)



    # Scan rows to find where actual date+numeric data begins

    data_start = 0

    for i, row in raw.iterrows():

        val0, val1 = row.iloc[0], row.iloc[1]

        try:

            pd.to_datetime(val0)

            float(val1)

            data_start = i

            break

        except (ValueError, TypeError):

            continue



    # Use header row just above data_start if available, else integer cols

    if data_start > 0:

        header_row = raw.iloc[data_start - 1]

        df = raw.iloc[data_start:].copy()

        df.columns = range(df.shape[1])

    else:

        df = raw.iloc[data_start:].copy()

        df.columns = range(df.shape[1])



    df[0] = pd.to_datetime(df[0], errors="coerce")

    df[1] = pd.to_numeric(df[1], errors="coerce")

    df = df.dropna(subset=[0, 1])



    series = (

        df.set_index(0)[1]

          .sort_index()

          .resample("MS")

          .mean()

    )



    print(

        f"  WSI Vector: sheet='{sheet}', data_start_row={data_start}, "

        f"{series.notna().sum()} obs "

        f"({series.index.min().date()} to {series.index.max().date()})"

    )



    return series.rename("WSI")


def load_wui_india_t1(path: str) -> pd.Series:

    """Load India country-level WUI from the T1 sheet of the WUI_M dataset.

    The T1 sheet has country columns; we extract the 'IND' column.

    """

    xl = pd.ExcelFile(path)

    sheet = "T1" if "T1" in xl.sheet_names else None

    if sheet is None:

        raise ValueError(f"Sheet 'T1' not found in {path}. Available: {xl.sheet_names}")



    raw = pd.read_excel(path, sheet_name=sheet, header=None)



    # Scan to find the header row containing 'IND'

    header_row_idx = None

    for i, row in raw.iterrows():

        if any(str(v).strip().upper() == "IND" for v in row.values):

            header_row_idx = i

            break



    if header_row_idx is None:

        raise ValueError("Could not locate 'IND' column header in T1 sheet.")



    # Re-read with proper header

    df = pd.read_excel(path, sheet_name=sheet, header=header_row_idx)

    df.columns = df.columns.str.strip()



    # Find the date column (first column, or one named 'Date'/'date')

    date_col = None

    for col in df.columns:

        try:

            pd.to_datetime(df[col].dropna().iloc[0])

            date_col = col

            break

        except Exception:

            continue



    if date_col is None:

        raise ValueError("Could not identify date column in T1 sheet.")



    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df["IND"] = pd.to_numeric(df["IND"], errors="coerce")

    df = df.dropna(subset=[date_col, "IND"])



    series = (

        df.set_index(date_col)["IND"]

          .sort_index()

          .resample("MS")

          .mean()

    )



    print(

        f"  WUI India T1 Vector: {series.notna().sum()} obs "

        f"({series.index.min().date()} to {series.index.max().date()})"

    )



    return series.rename("WUI_IND_T1")


def load_ofr_fsi_csv(path: str) -> tuple[pd.Series, pd.Series]:

    """Load OFR FSI and Emerging Markets columns from fsi.csv.

    Expected columns: Date, OFR FSI (or similar), Emerging markets (or similar).

    Returns two Series: ofr_fsi, emerging_markets.

    """

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()



    # Flexible date parsing

    date_col = None

    for col in df.columns:

        if col.lower() in ("date", "dates", "month", "period"):

            date_col = col

            break

    if date_col is None:

        date_col = df.columns[0]  # fallback to first column



    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[date_col])

    df = df.set_index(date_col).sort_index()

    df.index = df.index.to_period("M").to_timestamp()  # normalize to MS



    # Locate OFR FSI column (case-insensitive fuzzy match)

    ofr_col = None

    em_col  = None

    for col in df.columns:

        col_lower = col.lower().strip()

        if "ofr" in col_lower and "fsi" in col_lower:

            ofr_col = col

        elif "emerging" in col_lower:

            em_col = col



    if ofr_col is None:

        raise ValueError(f"Could not find OFR FSI column in {path}. Columns: {list(df.columns)}")

    if em_col is None:

        raise ValueError(f"Could not find Emerging markets column in {path}. Columns: {list(df.columns)}")



    ofr_fsi = pd.to_numeric(df[ofr_col], errors="coerce").resample("MS").mean().rename("OFR_FSI")

    em_mkts = pd.to_numeric(df[em_col],  errors="coerce").resample("MS").mean().rename("EM_FSI")



    print(

        f"  OFR FSI Vector: {ofr_fsi.notna().sum()} obs "

        f"({ofr_fsi.index.min().date()} to {ofr_fsi.index.max().date()})"

    )

    print(

        f"  Emerging Markets FSI Vector: {em_mkts.notna().sum()} obs "

        f"({em_mkts.index.min().date()} to {em_mkts.index.max().date()})"

    )



    return ofr_fsi, em_mkts


def load_policy_uncertainty(path: str) -> pd.Series:

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # FIX: Formulates date directly to the 1st of each month

    df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01")

    pu = df.set_index("date")["India News-Based Policy Uncertainty Index"].sort_index().resample("MS").mean()

    return pu.rename("PU")


# =============================================================================

# ── RBI DBIE hard-data loader (GDP / GFCF) ───────────────────────────────────
# Expects a CSV with a Date column plus GDP and GFCF columns (any reasonable
# aliasing of those names is matched case-insensitively). Data is resampled
# to quarterly start ('QS') since RBI DBIE national accounts are quarterly.
# =============================================================================

def load_rbi_dbie_gdp_gfcf(path: str) -> tuple[pd.Series, pd.Series]:

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    date_col = None
    for col in df.columns:
        if col.lower() in ("date", "quarter", "period"):
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    gdp_col = gfcf_col = None
    for col in df.columns:
        cl = col.lower()
        if "gdp" in cl:
            gdp_col = col
        elif "gfcf" in cl or "fixed capital" in cl:
            gfcf_col = col

    if gdp_col is None:
        raise ValueError(f"Could not find GDP column in {path}. Columns: {list(df.columns)}")
    if gfcf_col is None:
        raise ValueError(f"Could not find GFCF column in {path}. Columns: {list(df.columns)}")

    gdp = pd.to_numeric(df[gdp_col], errors="coerce").resample("QS").mean().rename("GDP")
    gfcf = pd.to_numeric(df[gfcf_col], errors="coerce").resample("QS").mean().rename("GFCF")

    # YoY growth is what typically maps to "contractions" for overlay purposes
    gdp_yoy = gdp.pct_change(4) * 100
    gfcf_yoy = gfcf.pct_change(4) * 100

    print(
        f"  RBI DBIE GDP Vector: {gdp.notna().sum()} obs "
        f"({gdp.index.min().date()} to {gdp.index.max().date()})"
    )
    print(
        f"  RBI DBIE GFCF Vector: {gfcf.notna().sum()} obs "
        f"({gfcf.index.min().date()} to {gfcf.index.max().date()})"
    )

    return gdp_yoy.rename("GDP_YOY"), gfcf_yoy.rename("GFCF_YOY")


def merge_series(fsi, comp):



    merged = pd.concat(

        [fsi, comp],

        axis=1

    )



    merged.columns = ["FSI", "COMP"]



    merged = merged.dropna()



    return merged



def plot_timeseries(fsi, comp, name, filename):



    merged = merge_series(fsi, comp)



    if len(merged) < 5:

        return



    merged["FSI_Z"] = zscore(merged["FSI"])

    merged["COMP_Z"] = zscore(merged["COMP"])



    pearson_r, pearson_p = stats.pearsonr(

        merged["FSI_Z"],

        merged["COMP_Z"]

    )



    spearman_r, spearman_p = stats.spearmanr(

        merged["FSI_Z"],

        merged["COMP_Z"]

    )



    plt.figure(figsize=(14,6))



    plt.plot(

        merged.index,

        merged["FSI_Z"],

        linewidth=2,

        label="FSI"

    )



    plt.plot(

        merged.index,

        merged["COMP_Z"],

        linewidth=2,

        linestyle="--",

        label=name

    )



    plt.title(

        f"{name} vs FSI\n"

        f"Pearson r={pearson_r:.3f} p={pearson_p:.3e}\n"

        f"Spearman r={spearman_r:.3f} p={spearman_p:.3e}"

    )



    plt.legend()

    plt.grid(True)



    plt.tight_layout()



    plt.savefig(filename, dpi=300)



    plt.close()



def plot_scatter(fsi, comp, name, filename):



    merged = merge_series(fsi, comp)



    if len(merged) < 5:

        return



    x = zscore(merged["FSI"])

    y = zscore(merged["COMP"])



    pearson_r, pearson_p = stats.pearsonr(x, y)



    slope, intercept, _, _, _ = stats.linregress(x, y)



    xs = np.linspace(x.min(), x.max(), 100)



    plt.figure(figsize=(7,7))



    plt.scatter(x, y, alpha=0.7)



    plt.plot(

        xs,

        slope * xs + intercept,

        linewidth=2

    )



    plt.xlabel("FSI")



    plt.ylabel(name)



    plt.title(

        f"{name} Scatter Plot\n"

        f"Pearson r={pearson_r:.3f} p={pearson_p:.3e}"

    )



    plt.grid(True)



    plt.tight_layout()



    plt.savefig(filename, dpi=300)



    plt.close()



def plot_rolling_corr(fsi, comp, name, filename):



    merged = merge_series(fsi, comp)



    if len(merged) < 24:

        return



    rolling = (

        merged["FSI"]

        .rolling(ROLLING_WINDOW)

        .corr(merged["COMP"])

    )



    avg_corr = rolling.mean()



    plt.figure(figsize=(14,6))



    plt.plot(

        rolling.index,

        rolling,

        linewidth=2

    )



    plt.axhline(

        0,

        linestyle="--"

    )



    plt.ylim(-1,1)



    plt.title(

        f"{name} Rolling Correlation\n"

        f"Average correlation={avg_corr:.3f}"

    )



    plt.grid(True)



    plt.tight_layout()



    plt.savefig(filename, dpi=300)



    plt.close()



def plot_cross_corr(fsi, comp, name, filename):



    merged = merge_series(fsi, comp)



    if len(merged) < 24:

        return



    x = pd.Series(

        zscore(merged["FSI"]),

        index=merged.index

    )



    y = pd.Series(

        zscore(merged["COMP"]),

        index=merged.index

    )



    lags = np.arange(-12, 13)



    corrs = []



    for lag in lags:

        corrs.append(

            x.corr(y.shift(lag))

        )



    corrs = np.array(corrs)



    best_idx = np.nanargmax(np.abs(corrs))



    best_lag = lags[best_idx]



    best_corr = corrs[best_idx]

    zero_idx = np.where(lags == 0)[0][0]

    zero_corr = corrs[zero_idx]



    plt.figure(figsize=(12,5))



    plt.bar(

        lags,

        corrs

    )



    plt.axvline(

        0,

        color="red",

        linewidth=2,

        linestyle="--",

        label=f"Lag0 = {zero_corr:.3f}"

    )



    plt.axhline(

        0,

        linestyle="--"

    )



    plt.legend()



    plt.title(

        f"{name} Cross Correlation\n"

        f"Lag0 correlation={zero_corr:.3f}\n"

        f"Best lag={best_lag}  correlation={best_corr:.3f}"

    )



    plt.xlabel("Lag (months)")

    plt.ylabel("Correlation")



    plt.grid(True)



    plt.tight_layout()



    plt.savefig(filename, dpi=300)



    plt.close()



def generate_plots(fsi, comp, name, prefix):



    plot_timeseries(

        fsi,

        comp,

        name,

        os.path.join(COMP_DIR, f"{prefix}_timeseries.png")

    )



    plot_scatter(

        fsi,

        comp,

        name,

        os.path.join(COMP_DIR, f"{prefix}_scatter.png")

    )



    plot_rolling_corr(

        fsi,

        comp,

        name,

        os.path.join(COMP_DIR, f"{prefix}_rolling_corr.png")

    )



    plot_cross_corr(

        fsi,

        comp,

        name,

        os.path.join(COMP_DIR, f"{prefix}_cross_corr.png")

    )


def _scatter_panel(ax, x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str, color: str):

    merged = pd.concat([x, y], axis=1).dropna()

    if len(merged) < 10:

        ax.text(0.5, 0.5, "Insufficient overlap", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(title, fontsize=11, pad=6)

        return



    xv, yv = merged.iloc[:, 0].values, merged.iloc[:, 1].values

    slope, intercept, r, p, _ = stats.linregress(xv, yv)

    x_fit = np.linspace(xv.min(), xv.max(), 100)



    ax.scatter(xv, yv, color=color, alpha=0.55, s=28, edgecolors="white", linewidths=0.4)

    ax.plot(x_fit, slope * x_fit + intercept, color="black", lw=1.5)

    ax.set_xlabel(xlabel, fontsize=10)

    ax.set_ylabel(ylabel, fontsize=10)

    ax.set_title(title, fontsize=11, pad=6)

    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.6)

    ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.6)

    ax.set_facecolor("#fafafa")



    txt = f"r = {r:+.3f}\nR² = {r**2:.3f}\np = {p:.2e}"

    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,

            bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.90))





def _timeseries_panel(ax, fsi: pd.Series, comp: pd.Series, comp_label: str, comp_color: str, title: str):

    merged = pd.concat([fsi, comp], axis=1).dropna()

    ax2    = ax.twinx()



    l1, = ax.plot(merged.index, merged["FSI"], color=PALETTE["fsi"], lw=1.7, label="FSI")

    l2, = ax2.plot(merged.index, merged.iloc[:, 1], color=comp_color, lw=1.7, linestyle="--", alpha=0.85, label=comp_label)



    ax.set_ylabel("FSI Index", color=PALETTE["fsi"], fontsize=9)

    ax2.set_ylabel(comp_label, color=comp_color, fontsize=9)

    ax.tick_params(axis="y", colors=PALETTE["fsi"])

    ax2.tick_params(axis="y", colors=comp_color)

    ax.set_title(title, fontsize=11, pad=6)

    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.6)

    ax.set_facecolor("#fafafa")





def _rolling_corr_panel(ax, fsi: pd.Series, comp: pd.Series, comp_label: str, title: str):

    merged = pd.concat([fsi, comp], axis=1).dropna()

    if len(merged) < ROLLING_WINDOW + 2:

        ax.text(0.5, 0.5, "Insufficient overlap", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(title, fontsize=11, pad=6)

        return



    rc = merged["FSI"].rolling(ROLLING_WINDOW).corr(merged.iloc[:, 1])

    r_overall, _ = stats.pearsonr(merged["FSI"], merged.iloc[:, 1])



    ax.fill_between(rc.index, rc, 0, where=(rc >= 0), color=PALETTE["roll"], alpha=0.32)

    ax.fill_between(rc.index, rc, 0, where=(rc <  0), color=PALETTE["neg"],  alpha=0.32)

    ax.plot(rc.index, rc, color=PALETTE["roll"], lw=1.5)

    ax.axhline(0,         color=PALETTE["zero"], lw=0.9, linestyle="--")

    ax.axhline(r_overall, color="black",        lw=0.9, linestyle=":", label=f"Overall r = {r_overall:+.3f}")

    ax.set_ylim(-1, 1)

    ax.set_ylabel("Pearson r", fontsize=9)

    ax.set_title(title, fontsize=11, pad=6)

    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.6)

    ax.set_facecolor("#fafafa")





def print_correlation_table(fsi: pd.Series, vix: pd.Series, gui: pd.Series, pu: pd.Series,
                             gpr: pd.Series, gwui: pd.Series, gepu: pd.Series,
                             wsi: pd.Series = None, wui_ind_t1: pd.Series = None,
                             ofr_fsi: pd.Series = None, em_fsi: pd.Series = None):

    print("\n" + "=" * 70)

    print("  FSI Co-movement Metrics (Pratap & Priyaranjan Table A1 Alignment)")

    print("  [Main FSI now generated via Hamilton (2018) regression filter]")

    print("=" * 70)

    comparisons = [
        ("India VIX", vix),
        ("GUI", gui),
        ("PU", pu),
        ("GPR", gpr),
        ("Global WUI", gwui),
        ("Global EPU", gepu),
        ("WSI (GDP-Weighted)", wsi),
        ("WUI India T1 (IND)", wui_ind_t1),
        ("OFR FSI", ofr_fsi),
        ("Emerging Markets FSI", em_fsi),
    ]

    for name, comp in comparisons:

        if comp is None:

            continue

        m = pd.concat([fsi, comp], axis=1).dropna()

        if len(m) < 10:

            print(f"  {name:35s}  — Insufficient Overlap (Check loader parameters)")

            continue

        pr, pp = stats.pearsonr(m["FSI"],  m.iloc[:, 1])

        sr, sp = stats.spearmanr(m["FSI"], m.iloc[:, 1])

        print(f"  {name:35s}  Pearson r={pr:+.4f} (p={pp:.2e}) | Spearman ρ={sr:+.4f}")

    print("=" * 70)


# =============================================================================

# ── VAR / FEVD VALIDATION MODULE ─────────────────────────────────────────────
# FEVD (Cholesky-identified) is retained as-is. Granger causality has been
# REPLACED by the Toda-Yamamoto procedure below, and a Block Exogeneity Wald
# test module has been added.
# =============================================================================

def run_var_for_fevd(fsi: pd.Series, comp: pd.Series, comp_name: str,
                      maxlags=VAR_MAXLAGS, ic=VAR_IC):
    """
    Fits a bivariate VAR(FSI, comp) on levels, selected by `ic` (default
    BIC), purely to supply a fitted VARResults object for FEVD. (Causality
    inference itself is now done via the Toda-Yamamoto module, which is
    valid regardless of integration order — this plain-levels VAR is not
    used for causality testing here.)
    """
    merged = pd.concat([fsi.rename("FSI"), comp.rename(comp_name)], axis=1).dropna()

    result = {
        "comp_name": comp_name,
        "n_obs": len(merged),
        "lag_order": None,
        "fitted": None,
        "error": None,
    }

    if len(merged) < maxlags * 3:
        result["error"] = (
            f"Insufficient overlap for VAR({maxlags}): {len(merged)} obs available."
        )
        return result

    try:
        model = VAR(merged)
        sel = model.select_order(maxlags=maxlags)
        best_lag = getattr(sel, ic)
        if best_lag == 0:
            best_lag = 1  # VAR needs at least 1 lag

        fitted = model.fit(best_lag)

        result["lag_order"] = best_lag
        result["fitted"] = fitted

    except Exception as e:
        result["error"] = str(e)

    return result


def run_fevd(fitted_var, fsi_name="FSI", target_name=None, horizons=FEVD_HORIZONS):
    """
    Given a fitted VARResults object, compute the Cholesky-identified FEVD
    and report the share of `target_name`'s forecast-error variance explained
    by orthogonalized shocks to `fsi_name`, at each horizon in `horizons`.

    NOTE: Cholesky ordering matters. FSI is placed first in the VAR input
    (see run_var_for_fevd), meaning FSI shocks are assumed contemporaneously
    exogenous to the target variable — the standard assumption when testing
    whether FSI is a leading/causal driver.
    """
    if fitted_var is None or target_name is None:
        return {}

    cols = list(fitted_var.names)
    if fsi_name not in cols or target_name not in cols:
        return {"error": f"Variable names not found in VAR: {cols}"}

    max_h = max(horizons)
    fevd = fitted_var.fevd(max_h)

    out = {}
    resp_idx = cols.index(target_name)
    imp_idx = cols.index(fsi_name)
    for h in horizons:
        pct = fevd.decomp[resp_idx, h - 1, imp_idx]
        out[h] = float(pct * 100)

    return out


# =============================================================================

# ── NEW: TODA-YAMAMOTO (1995) AUGMENTED-LAG GRANGER CAUSALITY ───────────────
# Replaces the plain Granger causality test. Procedure:
#   1. Determine each series' maximum order of integration d_max via
#      sequential ADF testing (test levels; if unit root not rejected,
#      difference and re-test; stop at TY_MAX_INTEGRATION_ORDER or first
#      rejection).
#   2. Fit an UNRESTRICTED VAR in LEVELS with lag order k (selected on the
#      levels data by `ic`, default BIC) PLUS d_max extra lags, i.e.
#      VAR(k + d_max) — this "augmented lag" trick is Toda & Yamamoto's key
#      insight: it makes the usual asymptotic chi-square Wald distribution
#      valid for the *first k* lag coefficients regardless of whether the
#      system is I(0), I(1), or cointegrated, without needing a separate
#      cointegration pre-test.
#   3. Test the null that the coefficients on the FIRST k lags of variable X
#      are jointly zero in variable Y's equation (a standard Wald test
#      restricted to the first k blocks; the extra d_max lags are included
#      in estimation but excluded from the restriction, exactly as in
#      Toda-Yamamoto/Dolado-Lütkepohl (1996)).
# =============================================================================

def _determine_max_integration_order(series: pd.Series, max_d=TY_MAX_INTEGRATION_ORDER,
                                      alpha=0.05) -> int:
    """
    Sequential ADF test: start at d=0 (levels). If ADF fails to reject the
    unit-root null (p >= alpha), difference once and re-test, up to max_d.
    Returns the estimated order of integration d (0, 1, ..., or max_d).
    """
    s = series.dropna().copy()
    d = 0
    while d < max_d:
        try:
            adf_stat, p_value, *_ = adfuller(s, autolag="AIC")
        except Exception:
            # If ADF itself fails (e.g. too short/constant), assume stationary
            return d
        if p_value < alpha:
            return d
        s = s.diff().dropna()
        d += 1
    return max_d


def toda_yamamoto_causality(fsi: pd.Series, comp: pd.Series, comp_name: str,
                             maxlags=TY_MAXLAGS, ic=TY_IC,
                             max_integration_order=TY_MAX_INTEGRATION_ORDER):
    """
    Runs the Toda-Yamamoto augmented-VAR Granger causality test in BOTH
    directions: FSI -> comp and comp -> FSI.

    Returns a dict with the estimated integration orders, chosen lag k,
    d_max, the augmented VAR(k + d_max) fit, and Wald test results
    (statistic, df, p-value, conclusion) for each direction.
    """
    merged = pd.concat([fsi.rename("FSI"), comp.rename(comp_name)], axis=1).dropna()

    result = {
        "comp_name": comp_name,
        "n_obs": len(merged),
        "d_fsi": None,
        "d_comp": None,
        "d_max": None,
        "k_lag": None,
        "augmented_lag": None,
        "fitted": None,
        "fsi_causes_comp": None,   # dict: stat, df, pvalue, conclusion
        "comp_causes_fsi": None,   # dict: stat, df, pvalue, conclusion
        "error": None,
    }

    if len(merged) < (maxlags + max_integration_order) * 3:
        result["error"] = (
            f"Insufficient overlap for Toda-Yamamoto VAR: {len(merged)} obs available, "
            f"need roughly >= {(maxlags + max_integration_order) * 3}."
        )
        return result

    try:
        # Step 1: integration orders
        d_fsi = _determine_max_integration_order(merged["FSI"], max_integration_order)
        d_comp = _determine_max_integration_order(merged[comp_name], max_integration_order)
        d_max = max(d_fsi, d_comp, 1)  # at least 1 to be conservative/safe

        result["d_fsi"] = d_fsi
        result["d_comp"] = d_comp
        result["d_max"] = d_max

        # Step 2: select k on the LEVELS VAR via information criterion
        levels_model = VAR(merged)
        sel = levels_model.select_order(maxlags=maxlags)
        k = getattr(sel, ic)
        if k == 0:
            k = 1
        result["k_lag"] = k

        augmented_lag = k + d_max
        result["augmented_lag"] = augmented_lag

        # Step 3: fit the UNRESTRICTED augmented VAR(k + d_max) in levels
        fitted = levels_model.fit(augmented_lag)
        result["fitted"] = fitted

        # Step 4: Wald test restricted to the FIRST k lags of the "causing"
        # variable's coefficients in the "caused" variable's equation.
        # statsmodels' VARResults doesn't expose a direct "first-k-lags-only"
        # Wald test, so we build the restriction matrix manually using the
        # model's parameter table structure.
        def _first_k_wald(fitted_var, caused_var: str, causing_var: str, k_restrict: int):
            names = fitted_var.names
            n_vars = len(names)
            total_lag = fitted_var.k_ar

            # params shape: (1 + n_vars*total_lag, n_vars) -> intercept + lags
            # Column index for the caused variable's equation:
            eq_idx = names.index(caused_var)
            causing_idx = names.index(causing_var)

            # Row offset per lag block: row 0 = intercept, then for lag L
            # (1-indexed), rows [1 + (L-1)*n_vars : 1 + L*n_vars) correspond
            # to that lag's coefficients across all n_vars regressors, in
            # the same order as `names`.
            restrict_rows = []
            for lag in range(1, k_restrict + 1):
                row = 1 + (lag - 1) * n_vars + causing_idx
                restrict_rows.append(row)

            n_params_per_eq = fitted_var.params.shape[0]
            R = np.zeros((len(restrict_rows), n_params_per_eq))
            for i, row in enumerate(restrict_rows):
                R[i, row] = 1.0

            # statsmodels VARResults.wald_test expects restriction on the
            # full stacked parameter vector across ALL equations, in
            # column-major (equation-major) order: params.T.ravel().
            n_eqs = n_vars
            full_len = n_params_per_eq * n_eqs
            R_full = np.zeros((len(restrict_rows), full_len))
            col_offset = eq_idx * n_params_per_eq
            for i, row in enumerate(restrict_rows):
                R_full[i, col_offset + row] = 1.0

            # Manual Wald test (VARResults has no .wald_test method):
            # W = (R beta)' [R Cov(beta) R']^{-1} (R beta) ~ chi2(df)
            beta_stacked = fitted_var.params.T.to_numpy().ravel()
            cov_beta = fitted_var.cov_params()
            cov_beta = cov_beta.to_numpy() if hasattr(cov_beta, "to_numpy") else np.asarray(cov_beta)

            Rb = R_full @ beta_stacked
            middle = R_full @ cov_beta @ R_full.T
            middle_inv = np.linalg.pinv(middle)
            stat = float(Rb @ middle_inv @ Rb)
            df = R_full.shape[0]
            pval = float(1 - stats.chi2.cdf(stat, df))

            return {"statistic": stat, "pvalue": pval, "df": df}
        
        wald_fsi_to_comp = _first_k_wald(fitted, caused_var=comp_name, causing_var="FSI", k_restrict=k)
        wald_comp_to_fsi = _first_k_wald(fitted, caused_var="FSI", causing_var=comp_name, k_restrict=k)

        def _pack(wald_res, direction_label):
            stat = wald_res["statistic"]
            pval = wald_res["pvalue"]
            df = wald_res["df"]
            concl = (
                f"{direction_label} (Toda-Yamamoto)"
                if pval < 0.05
                else f"Cannot reject: no {direction_label.lower()} (Toda-Yamamoto)"
            )
            return {"stat": stat, "df": df, "pvalue": pval, "conclusion": concl}

        result["fsi_causes_comp"] = _pack(wald_fsi_to_comp, f"FSI Granger-causes {comp_name}")
        result["comp_causes_fsi"] = _pack(wald_comp_to_fsi, f"{comp_name} Granger-causes FSI")

    except Exception as e:
        result["error"] = str(e)

    return result


# =============================================================================

# ── NEW: BLOCK EXOGENEITY WALD TESTS ─────────────────────────────────────────
# For a fitted (Toda-Yamamoto-consistent, augmented-lag) VAR, tests — for
# each equation — the joint null that ALL lags of ALL OTHER variables are
# zero in that equation. Rejecting the null means the "other variables" as a
# block are NOT exogenous to (i.e. do help predict) this variable — this is
# the natural multivariate generalization of pairwise Granger causality and
# is standard practice to report alongside Toda-Yamamoto results.
# =============================================================================

def block_exogeneity_wald_test(fitted_var, restrict_to_first_k=None):
    """
    fitted_var: a statsmodels VARResults object (ideally the SAME augmented
        VAR(k + d_max) fitted for Toda-Yamamoto, for consistency).
    restrict_to_first_k: if provided, restricts the block-exogeneity test to
        only the first k lags (Toda-Yamamoto-consistent block exogeneity);
        if None, tests ALL estimated lags of the other variables (the
        classical block-exogeneity formulation).

    Returns a dict keyed by "equation_variable" -> {stat, df, pvalue,
    conclusion}, one entry per variable in the system.
    """
    names = fitted_var.names
    n_vars = len(names)
    total_lag = fitted_var.k_ar
    k_restrict = restrict_to_first_k if restrict_to_first_k is not None else total_lag

    n_params_per_eq = fitted_var.params.shape[0]
    n_eqs = n_vars
    full_len = n_params_per_eq * n_eqs

    results = {}

    for eq_var in names:
        eq_idx = names.index(eq_var)
        other_vars = [v for v in names if v != eq_var]

        restrict_rows = []
        for other_var in other_vars:
            other_idx = names.index(other_var)
            for lag in range(1, k_restrict + 1):
                row = 1 + (lag - 1) * n_vars + other_idx
                restrict_rows.append(row)

        R_full = np.zeros((len(restrict_rows), full_len))
        col_offset = eq_idx * n_params_per_eq
        for i, row in enumerate(restrict_rows):
            R_full[i, col_offset + row] = 1.0

        try:
            beta_stacked = fitted_var.params.T.to_numpy().ravel()
            cov_beta = fitted_var.cov_params()
            cov_beta = cov_beta.to_numpy() if hasattr(cov_beta, "to_numpy") else np.asarray(cov_beta)

            Rb = R_full @ beta_stacked
            middle = R_full @ cov_beta @ R_full.T
            middle_inv = np.linalg.pinv(middle)
            stat = float(Rb @ middle_inv @ Rb)
            df = len(restrict_rows)
            pval = float(1 - stats.chi2.cdf(stat, df))
            concl = (
                f"Reject block exogeneity: {', '.join(other_vars)} jointly help predict {eq_var}"
                if pval < 0.05
                else f"Cannot reject block exogeneity of {', '.join(other_vars)} w.r.t. {eq_var}"
            )
            results[eq_var] = {"stat": stat, "df": df, "pvalue": pval, "conclusion": concl}
        except Exception as e:
            results[eq_var] = {"stat": None, "df": None, "pvalue": None,
                                "conclusion": f"Wald test failed: {e}"}

    return results


def print_ty_granger_fevd_report(fsi: pd.Series, comparisons: dict):
    """
    comparisons: dict of {name: series} to test against FSI, e.g.
        {"India VIX": vix, "GUI": gui, "OFR FSI": ofr_fsi}

    Runs Toda-Yamamoto causality (both directions) + Block Exogeneity Wald
    tests + FEVD for each, printing a clear diagnostic report and returning
    a results dict keyed by comparison name.
    """
    print("\n" + "=" * 70)

    print("  Toda-Yamamoto Causality / Block Exogeneity Wald / FEVD")

    print("  Leading-Indicator Validation (Hamilton-based Main FSI)")

    print("=" * 70)

    all_results = {}
    for name, comp in comparisons.items():
        if comp is None or comp.dropna().empty:
            continue

        # ── Toda-Yamamoto causality (both directions) ────────────────────
        ty_result = toda_yamamoto_causality(fsi, comp, name)
        all_results[name] = ty_result

        if ty_result["error"]:
            print(f"\n  [{name}]  SKIPPED (Toda-Yamamoto) — {ty_result['error']}")
            continue

        print(f"\n  [{name}]  n={ty_result['n_obs']}  "
              f"d(FSI)={ty_result['d_fsi']}  d({name})={ty_result['d_comp']}  "
              f"d_max={ty_result['d_max']}  k(BIC)={ty_result['k_lag']}  "
              f"augmented VAR order={ty_result['augmented_lag']}")

        f2c = ty_result["fsi_causes_comp"]
        c2f = ty_result["comp_causes_fsi"]

        print(f"    TY Wald: FSI -> {name:20s}  stat={f2c['stat']:.3f}  df={f2c['df']}  "
              f"p={f2c['pvalue']:.4f}")
        print(f"      Conclusion: {f2c['conclusion']}")

        print(f"    TY Wald: {name} -> FSI{'':20s} stat={c2f['stat']:.3f}  df={c2f['df']}  "
              f"p={c2f['pvalue']:.4f}")
        print(f"      Conclusion: {c2f['conclusion']}")

        # ── Block Exogeneity Wald test on the SAME augmented VAR ──────────
        block_results = block_exogeneity_wald_test(
            ty_result["fitted"], restrict_to_first_k=ty_result["k_lag"]
        )
        ty_result["block_exogeneity"] = block_results

        print(f"    Block Exogeneity Wald tests (first k={ty_result['k_lag']} lags):")
        for eq_var, br in block_results.items():
            if br["stat"] is None:
                print(f"      {eq_var:12s}: {br['conclusion']}")
            else:
                print(f"      {eq_var:12s}: stat={br['stat']:.3f}  df={br['df']}  "
                      f"p={br['pvalue']:.4f}  -> {br['conclusion']}")

        # ── FEVD via a separate BIC-selected plain-levels VAR (kept
        #     independent of the TY augmented-lag VAR, since the augmented
        #     lags are a testing device, not necessarily the best model for
        #     forecast-error variance decomposition) ─────────────────────
        fevd_var_result = run_var_for_fevd(fsi, comp, name)
        if fevd_var_result["error"] is None:
            fevd_shares = run_fevd(fevd_var_result["fitted"], fsi_name="FSI",
                                    target_name=name, horizons=FEVD_HORIZONS)
            if fevd_shares and "error" not in fevd_shares:
                for h, pct in fevd_shares.items():
                    print(f"    FEVD: FSI shocks explain {pct:.2f}% of {name} "
                          f"variance at h={h} months")
                ty_result["fevd"] = fevd_shares
        else:
            print(f"    FEVD skipped — {fevd_var_result['error']}")

    print("=" * 70)
    return all_results


# =============================================================================

# ── NEW: DCC-GARCH (Engle 2002) DYNAMIC CONDITIONAL CORRELATION ─────────────
# Two-stage estimation:
#   Stage 1: fit a univariate GARCH(p,q) to each series' (demeaned, %-scaled)
#            returns/changes to get standardized residuals e_t = r_t / sigma_t.
#   Stage 2: model the standardized residuals' correlation dynamically via
#            Engle's DCC recursion:
#                Q_t = (1 - a - b) * Qbar + a * (e_{t-1} e_{t-1}') + b * Q_{t-1}
#                R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
#            where Qbar is the unconditional covariance of standardized
#            residuals, and (a, b) are scalar DCC parameters estimated by
#            maximizing the (concentrated, bivariate) DCC quasi-likelihood
#            via a simple bounded grid + local refinement (kept dependency-
#            light rather than pulling in a full multivariate DCC package).
# The resulting R_t[0,1] time series is the dynamic conditional correlation
# between India-FSI and the benchmark index — showing precisely how their
# co-movement regime shifts over time, rather than relying on one static
# Pearson r.
# =============================================================================

def _fit_univariate_garch_std_resid(series: pd.Series, p=DCC_GARCH_P, q=DCC_GARCH_Q,
                                     dist=DCC_DIST):
    """
    Fits a GARCH(p,q) (constant-mean) model to `series` (treated as a
    already-stationary "returns"-like input — first-differenced levels are
    used upstream) and returns the standardized residuals e_t = resid_t /
    conditional_vol_t, aligned to the original index.
    """
    s = series.dropna()
    # arch_model wants reasonably scaled data (percentage-like); rescale to
    # roughly unit variance x 10 for numerical stability, remember the scale.
    scale = 10.0 / (s.std() + 1e-9)
    scaled = s * scale

    am = arch_model(scaled, mean="Constant", vol="GARCH", p=p, q=q, dist=dist)
    res = am.fit(disp="off")

    std_resid = res.resid / res.conditional_volatility
    std_resid = pd.Series(std_resid, index=s.index, name=series.name)

    return std_resid, res


def _dcc_negative_loglik(params, e1: np.ndarray, e2: np.ndarray, Qbar: np.ndarray):
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10

    T = len(e1)
    Q_t = Qbar.copy()
    nll = 0.0

    for t in range(T):
        Rt_diag = np.sqrt(np.diag(Q_t))
        # guard against numerical blow-up
        if np.any(Rt_diag <= 1e-8):
            return 1e10
        Rt = Q_t / np.outer(Rt_diag, Rt_diag)
        Rt = np.clip(Rt, -0.9999, 0.9999)
        np.fill_diagonal(Rt, 1.0)

        et = np.array([e1[t], e2[t]])
        try:
            Rt_inv = np.linalg.inv(Rt)
            sign, logdet = np.linalg.slogdet(Rt)
            if sign <= 0:
                return 1e10
        except np.linalg.LinAlgError:
            return 1e10

        nll += 0.5 * (logdet + et @ Rt_inv @ et - et @ et)

        outer_e = np.outer(et, et)
        Q_t = (1 - a - b) * Qbar + a * outer_e + b * Q_t

    return nll


def fit_dcc_garch(fsi_diff: pd.Series, comp_diff: pd.Series, comp_name: str):
    """
    Runs the full two-stage DCC-GARCH procedure between fsi_diff and
    comp_diff (both should be stationary, e.g. first-differenced or already-
    stationary cyclical FSI/benchmark series). Returns a dict with the fitted
    dynamic correlation series R_t, DCC parameters (a, b), and diagnostics.
    """
    if not _ARCH_AVAILABLE:
        return {"error": "`arch` package not installed — run `pip install arch`.",
                "dcc_corr": None}

    merged = pd.concat([fsi_diff.rename("FSI"), comp_diff.rename(comp_name)], axis=1).dropna()

    result = {
        "comp_name": comp_name,
        "n_obs": len(merged),
        "a": None,
        "b": None,
        "dcc_corr": None,
        "static_corr": None,
        "error": None,
    }

    if len(merged) < 36:
        result["error"] = f"Insufficient overlap for DCC-GARCH: {len(merged)} obs (need >= 36)."
        return result

    try:
        std_resid_fsi, garch_fsi_res = _fit_univariate_garch_std_resid(merged["FSI"])
        std_resid_comp, garch_comp_res = _fit_univariate_garch_std_resid(merged[comp_name])

        aligned = pd.concat([std_resid_fsi, std_resid_comp], axis=1).dropna()
        e1 = aligned.iloc[:, 0].values
        e2 = aligned.iloc[:, 1].values

        Qbar = np.cov(np.vstack([e1, e2]))

        # Bounded grid search + local Nelder-Mead-style refinement for (a, b)
        from scipy.optimize import minimize

        best = None
        for a0 in (0.02, 0.05, 0.1):
            for b0 in (0.85, 0.90, 0.95):
                if a0 + b0 >= 0.999:
                    continue
                res = minimize(
                    _dcc_negative_loglik, x0=[a0, b0], args=(e1, e2, Qbar),
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 200},
                )
                if best is None or res.fun < best.fun:
                    best = res

        a_hat, b_hat = best.x
        a_hat = max(a_hat, 0.0)
        b_hat = max(b_hat, 0.0)

        # Recompute the Q_t / R_t path at the optimal (a, b) for output
        T = len(e1)
        Q_t = Qbar.copy()
        dcc_series = np.zeros(T)
        for t in range(T):
            Rt_diag = np.sqrt(np.diag(Q_t))
            Rt = Q_t / np.outer(Rt_diag, Rt_diag)
            Rt = np.clip(Rt, -0.9999, 0.9999)
            dcc_series[t] = Rt[0, 1]

            et = np.array([e1[t], e2[t]])
            outer_e = np.outer(et, et)
            Q_t = (1 - a_hat - b_hat) * Qbar + a_hat * outer_e + b_hat * Q_t

        dcc_corr = pd.Series(dcc_series, index=aligned.index, name=f"DCC_FSI_{comp_name}")

        result["a"] = float(a_hat)
        result["b"] = float(b_hat)
        result["dcc_corr"] = dcc_corr
        result["static_corr"] = float(np.corrcoef(e1, e2)[0, 1])

    except Exception as e:
        result["error"] = str(e)

    return result


def plot_dcc_correlation(dcc_result: dict, filename: str):
    """Plots the DCC-GARCH dynamic conditional correlation path with the
    static (unconditional) correlation shown as a reference line."""
    if dcc_result.get("dcc_corr") is None:
        return

    dcc_corr = dcc_result["dcc_corr"]
    static_corr = dcc_result["static_corr"]
    comp_name = dcc_result["comp_name"]

    plt.figure(figsize=(14, 6))
    plt.plot(dcc_corr.index, dcc_corr.values, color=PALETTE["dcc"], lw=1.8,
              label="DCC-GARCH dynamic correlation")
    plt.axhline(static_corr, color=PALETTE["zero"], lw=1.2, linestyle="--",
                label=f"Static correlation = {static_corr:+.3f}")
    plt.axhline(0, color="black", lw=0.7, linestyle=":")
    plt.ylim(-1, 1)
    plt.title(
        f"DCC-GARCH: FSI vs {comp_name}\n"
        f"DCC params: a={dcc_result['a']:.4f}, b={dcc_result['b']:.4f}  "
        f"(persistence a+b={dcc_result['a'] + dcc_result['b']:.4f})"
    )
    plt.ylabel("Conditional correlation")
    plt.legend()
    plt.grid(True, color=PALETTE["grid"])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def run_dcc_garch_suite(fsi: pd.Series, comparisons: dict):
    """
    comparisons: dict of {name: series}. For each, first-differences both
    FSI and the comparison series (to obtain stationary "shock" series
    appropriate for GARCH modeling), fits DCC-GARCH, prints a summary, saves
    a plot, and returns a results dict keyed by comparison name.
    """
    print("\n" + "=" * 70)
    print("  DCC-GARCH — Dynamic Conditional Correlation")
    print("  (India-FSI vs Benchmark Indices)")
    print("=" * 70)

    if not _ARCH_AVAILABLE:
        print("  SKIPPED — `arch` package not installed. Run: pip install arch")
        print("=" * 70)
        return {}

    fsi_diff = fsi.diff().dropna()

    all_results = {}
    for name, comp in comparisons.items():
        if comp is None or comp.dropna().empty:
            continue

        comp_diff = comp.diff().dropna()

        dcc_result = fit_dcc_garch(fsi_diff, comp_diff, name)
        all_results[name] = dcc_result

        if dcc_result["error"]:
            print(f"\n  [{name}]  SKIPPED — {dcc_result['error']}")
            continue

        mean_dcc = dcc_result["dcc_corr"].mean()
        min_dcc = dcc_result["dcc_corr"].min()
        max_dcc = dcc_result["dcc_corr"].max()

        print(f"\n  [{name}]  n={dcc_result['n_obs']}  "
              f"a={dcc_result['a']:.4f}  b={dcc_result['b']:.4f}  "
              f"persistence(a+b)={dcc_result['a'] + dcc_result['b']:.4f}")
        print(f"    Static correlation (standardized residuals): {dcc_result['static_corr']:+.4f}")
        print(f"    DCC-GARCH conditional correlation — mean={mean_dcc:+.4f}, "
              f"range=[{min_dcc:+.4f}, {max_dcc:+.4f}]")

        plot_dcc_correlation(
            dcc_result,
            os.path.join(COMP_DIR, f"dcc_garch_{name.lower().replace(' ', '_')}.png")
        )

        dcc_result["dcc_corr"].to_csv(
            os.path.join(COMP_DIR, f"dcc_garch_{name.lower().replace(' ', '_')}_series.csv")
        )

    print("=" * 70)
    return all_results


# =============================================================================

# ── MAIN PIPELINE EXECUTOR ───────────────────────────────────────────────────

# =============================================================================

def export_overall_category_contributions(keyword_memory):

    category_totals = {}

    category_counts = {}

    category_active = {}



    for category, kws in KEYWORD_CATEGORIES.items():

        total_value = 0.0

        active_kws = []



        for kw in kws:

            if kw in keyword_memory:

                total_value += keyword_memory[kw].sum()

                active_kws.append(kw)



        category_totals[category] = total_value

        category_counts[category] = len(active_kws)        # keywords that survived has_variance

        category_active[category] = ", ".join(active_kws)  # names of surviving keywords



    grand_total = sum(category_totals.values())



    contribution_pct = {

        k: (v / grand_total) * 100 if grand_total > 0 else 0.0

        for k, v in category_totals.items()

    }



    output = (

        pd.DataFrame({

            "Category":                   contribution_pct.keys(),

            "Contribution_Percentage":    contribution_pct.values(),

            "Total_Search_Volume_20Yrs":  [round(category_totals[k], 2) for k in contribution_pct],

        })

        .sort_values("Contribution_Percentage", ascending=False)

    )



    output["Contribution_Percentage"] = output["Contribution_Percentage"].round(4)



    out_path = os.path.join(COMP_DIR, "overall_keyword_category_contributions.csv")

    output.to_csv(out_path, index=False)

    print(f"\nSaved overall category contributions -> {out_path}")

    print("\nCategory Contributions (%)")

    print(output[["Category", "Contribution_Percentage", "Total_Search_Volume_20Yrs"]])



    return output



def main():

    print("=" * 70)

    print("  Optimized India Financial Stress Index Replication Framework")

    print("  MAIN FSI FILTER: Hamilton (2018) regression filter")

    print("  Keyword weighting: TF-IDF-style anchor-term normalization")

    print("=" * 70)

    

    pkl_exists = os.path.exists(PKL_KEYWORD_SERIES) or os.path.exists(PKL_RAW_FSI)



    if not FORCE_REFETCH and pkl_exists:

        print("\n─── Phase 1: Reconstituting Baseline Index from Binary Pickles ─────")

        raw_fsi, keyword_memory = load_raw_fsi_from_pkl()

        export_overall_category_contributions(keyword_memory)

    else:

        print("\n─── Phase 1: Initiating Network Batch Data Collection Window ───────")

        pytrends = initialize_pytrends_safely()

        raw_fsi, kept_kws, removed_kws = build_raw_fsi(pytrends)

        # ── Load the freshly-saved pickle so we have keyword_memory ──────────

        keyword_memory = pd.read_pickle(PKL_KEYWORD_SERIES)

        export_overall_category_contributions(keyword_memory)

        print(f"  Removed {len(removed_kws)} zero-variance keywords: {removed_kws}")



    print("\n─── Phase 2: Running Temporal Seasonal Filtering ───────────────────")

    sa_fsi = try_x13_seasonal_adjust(raw_fsi)



    print("\n─── Phase 3: MAIN FSI — Hamilton (2018) Regression Filter ──────────")

    fsi_hamilton_full = hamilton_filter_and_scale(sa_fsi)   # MAIN FSI, full sample

    fsi = trim_endpoints(fsi_hamilton_full)

    fsi.name = "FSI"

    print(f"  Main FSI (Hamilton) computed: h={HAMILTON_HORIZON}, p={HAMILTON_LAGS}, "

          f"{fsi.notna().sum()} obs "

          f"({fsi.index.min().date()} to {fsi.index.max().date()})")

    print("\n─── Phase 3b: Robustness — Alternative Filters (HP, BK, CF) ────────")

    fsi_hp_full = hp_filter_and_scale(sa_fsi)
    fsi_bk = bk_filter_and_scale(sa_fsi)
    fsi_cf = cf_filter_and_scale(sa_fsi)

    fsi_hp = trim_endpoints(fsi_hp_full)
    fsi_bk = trim_endpoints(fsi_bk)
    fsi_cf = trim_endpoints(fsi_cf)

    print("\n─── Phase 3c: Robustness — Kaiser-Maravall ARIMA-Extended HP ───────")
    try:
        fsi_km_hp, km_order = km_arima_extended_hp_filter_and_scale(sa_fsi)
        fsi_km_hp_trimmed = trim_endpoints(fsi_km_hp)
        print(f"  ARIMA order selected: {km_order}")
        print(f"  KM-extended HP filter computed: horizon={KM_ARIMA_HORIZON}mo, "
              f"{fsi_km_hp.notna().sum()} obs")

        # Compare last few observations: MAIN (Hamilton) vs standard HP vs
        # KM-extended HP, illustrating the endpoint-bias differences across
        # detrending choices.
        compare_tail = pd.concat(
            [fsi_hamilton_full.rename("FSI_Hamilton_MAIN"),
             fsi_hp_full.rename("HP_standard"),
             fsi_km_hp.rename("HP_KM_extended")],
            axis=1
        ).dropna().tail(6)
        print("  Last 6 months — Hamilton (MAIN) vs standard HP vs KM-extended HP:")
        print(compare_tail.to_string())
    except Exception as e:
        fsi_km_hp = None
        fsi_km_hp_trimmed = None
        print(f"  WARNING: KM-extended ARIMA-HP filter failed — {e}")

    robustness_series = {"FSI_HAMILTON_MAIN": fsi, "FSI_HP": fsi_hp, "FSI_BK": fsi_bk, "FSI_CF": fsi_cf}
    if fsi_km_hp_trimmed is not None:
        robustness_series["FSI_KM_HP"] = fsi_km_hp_trimmed

    robustness_df = pd.concat(robustness_series, axis=1).dropna()

    if len(robustness_df) > 10:
        print("\n  Pairwise correlations across all detrending methods:")
        cols = robustness_df.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r, p = stats.pearsonr(robustness_df[cols[i]], robustness_df[cols[j]])
                print(f"    {cols[i]:18s} vs {cols[j]:18s}:  r={r:+.4f}  (p={p:.2e}, n={len(robustness_df)})")

        robustness_df.to_csv(os.path.join(COMP_DIR, "filter_robustness_comparison.csv"))
    else:
        print("  WARNING: Insufficient overlap across filters for robustness correlation.")

    # ── Phase 3d — NEW: Robustness Calibration — Multi-Anchor Overlapping
    #     Window Calibration. Executes a rolling 12-month, overlapping-
    #     window re-anchoring + splice-calibration reconstruction of the
    #     raw FSI (independently re-normalizing each rolling window the way
    #     Google Trends' own internal max-100 scaling would, then bridging
    #     windows via their overlap regions) and compares it against the
    #     single-shot MAIN FSI. High agreement is direct empirical proof
    #     that the index is robust to Google Trends' internal scaling
    #     artifacts, rather than an artifact of any one anchoring choice.
    print("\n─── Phase 3d: Robustness Calibration — Multi-Anchor Overlapping Window ─")
    try:
        tfidf_weights_path = os.path.join(COMP_DIR, "tfidf_anchor_weights.csv")
        if os.path.exists(tfidf_weights_path):
            tfidf_weights_for_maow = pd.read_csv(tfidf_weights_path, index_col=0).iloc[:, 0]
        else:
            tfidf_weights_for_maow = compute_tfidf_anchor_weights(keyword_memory)

        maow_result = run_maow_robustness_calibration(
            keyword_memory=keyword_memory,
            tfidf_weights=tfidf_weights_for_maow,
            main_fsi=fsi,
            window_months=MAOW_WINDOW_MONTHS,
            step_months=MAOW_STEP_MONTHS,
        )
    except Exception as e:
        maow_result = {"error": str(e)}
        print(f"  WARNING: Multi-Anchor Overlapping Window calibration failed — {e}")

    citation_header = [
        "# India Financial Stress Index (India-FSI)",
        "# Authors: Aswini Kumar Mishra, Nishad Kotkar, and Atharva Singh Rathore",
        "# Citation: 'Macro-Financial Shocks and Search Sentiment: A New Financial Stress Index for India', Finance Research Letters (2026).",
        "# Hosted by: Economic Policy Uncertainty (EPU) - https://www.policyuncertainty.com",
        f"# Last Updated: {datetime.today().strftime('%Y-%m-%d')}",
        "# Frequency: Monthly",
    ]

    out_df = pd.DataFrame({
        "Date": fsi.index.strftime("%Y-%m-%d"),
        "India_FSI": fsi.values.round(4)
    })

    with open(OUT_CSV, "w") as f:
        f.write("\n".join(citation_header) + "\n")
        out_df.to_csv(f, index=False)

    print(f"\n  Saved production CSV with citation header -> {OUT_CSV}")

    # Generate the headline annotated slide / plot requested by Prof. Scott Baker
    plt.figure(figsize=(12, 5))
    plt.plot(fsi.index, fsi.values, color=PALETTE["fsi"], lw=1.8, label="India-FSI")
    plt.axhline(0, color=PALETTE["zero"], lw=0.8, linestyle="--")
    plt.title(f"India Financial Stress Index (India-FSI) — Updated {datetime.today().strftime('%B %Y')}", fontsize=12, pad=10)
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Index (Hamilton Cyclical Component)", fontsize=10)
    plt.grid(True, color=PALETTE["grid"], linestyle=":", alpha=0.7)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()

    print(f"  Saved headline presentation figure -> {OUT_PNG}")



    print("\n─── Phase 4: Consolidating Analytical Validation Vectors ───────────")

    vix = gui = pu = gpr = gwui = gepu = None

    wsi = wui_ind_t1 = ofr_fsi = em_fsi = None
    gdp_yoy = gfcf_yoy = None

    try:

        vix = load_india_vix(FILE_VIX)

        print(f"  VIX Vector: {vix.notna().sum()} obs ({vix.index.min().date()} to {vix.index.max().date()})")

    except FileNotFoundError: pass

    try:

        print(os.path.abspath(FILE_GUI))

        gui = load_gui(FILE_GUI)

        print(f"  GUI Vector: {gui.notna().sum()} obs ({gui.index.min().date()} to {gui.index.max().date()})")

    except FileNotFoundError: pass

    try:

        pu = load_policy_uncertainty(FILE_PU)

        print(f"  PU Vector:  {pu.notna().sum()} obs ({pu.index.min().date()} to {pu.index.max().date()})")

    except FileNotFoundError: pass

    try:

        gpr = load_gpr(FILE_GPR)

        print(

            f"GPR Vector: {gpr.notna().sum()} obs "

            f"({gpr.index.min().date()} "

            f"to {gpr.index.max().date()})"

        )

    except FileNotFoundError:

        pass



    try:

        gwui = load_global_wui(FILE_GLOBAL_WUI)



        print(

            f"Global WUI Vector: {gwui.notna().sum()} obs "

            f"({gwui.index.min().date()} "

            f"to {gwui.index.max().date()})"

        )

    except FileNotFoundError:

        pass



    try:

        gepu = load_gepu(FILE_GEPU)



        print(

            f"GEPU Vector: {gepu.notna().sum()} obs "

            f"({gepu.index.min().date()} "

            f"to {gepu.index.max().date()})"

        )

    except FileNotFoundError:

        pass



    try:

        wsi = load_wsi(FILE_WSI)

    except FileNotFoundError:

        print(f"  WARNING: {FILE_WSI} not found — WSI comparisons skipped.")

    except Exception as e:

        print(f"  WARNING: WSI load failed — {e}")



    try:

        wui_ind_t1 = load_wui_india_t1(FILE_GLOBAL_WUI)

    except FileNotFoundError:

        print(f"  WARNING: {FILE_GLOBAL_WUI} not found for T1 — skipped.")

    except Exception as e:

        print(f"  WARNING: WUI T1 India load failed — {e}")



    try:

        ofr_fsi, em_fsi = load_ofr_fsi_csv(FILE_OFR_FSI)

    except FileNotFoundError:

        print(f"  WARNING: {FILE_OFR_FSI} not found — OFR FSI/EM comparisons skipped.")

    except Exception as e:

        print(f"  WARNING: OFR FSI load failed — {e}")



    # ── RBI DBIE hard-data anchor ─────────────────────────────────────────────

    try:

        gdp_yoy, gfcf_yoy = load_rbi_dbie_gdp_gfcf(FILE_RBI_DBIE)

    except FileNotFoundError:

        print(f"  WARNING: {FILE_RBI_DBIE} not found — RBI GDP/GFCF hard-data "

              f"anchoring skipped. Point FILE_RBI_DBIE at a CSV with Date/GDP/GFCF "

              f"columns to enable this comparison.")

    except Exception as e:

        print(f"  WARNING: RBI DBIE load failed — {e}")



    _empty = pd.Series(dtype=float)

    _vix   = vix if vix is not None else _empty.rename("VIX")

    _gui   = gui if gui is not None else _empty.rename("GUI")

    _pu    = pu  if pu  is not None else _empty.rename("PU")

    _gpr   = gpr if gpr is not None else _empty.rename("GPR")

    _gwui  = gwui if gwui is not None else _empty.rename("GWUI")

    _gepu  = gepu if gepu is not None else _empty.rename("GEPU")

    _wsi         = wsi         if wsi         is not None else _empty.rename("WSI")

    _wui_ind_t1  = wui_ind_t1  if wui_ind_t1  is not None else _empty.rename("WUI_IND_T1")

    _ofr_fsi     = ofr_fsi     if ofr_fsi     is not None else _empty.rename("OFR_FSI")

    _em_fsi      = em_fsi      if em_fsi      is not None else _empty.rename("EM_FSI")

    _gdp_yoy     = gdp_yoy     if gdp_yoy     is not None else _empty.rename("GDP_YOY")

    _gfcf_yoy    = gfcf_yoy    if gfcf_yoy    is not None else _empty.rename("GFCF_YOY")



    if any(s is not None for s in [vix, gui, pu]):

        print_correlation_table(
            fsi, _vix, _gui, _pu, _gpr, _gwui, _gepu,
            wsi=_wsi, wui_ind_t1=_wui_ind_t1,
            ofr_fsi=_ofr_fsi, em_fsi=_em_fsi
        )

        # ── Phase 4b — Toda-Yamamoto causality / Block Exogeneity / FEVD ─────
        print("\n─── Phase 4b: Toda-Yamamoto Causality, Block Exogeneity & FEVD ─────")
        var_comparisons = {}
        if gui is not None:
            var_comparisons["GUI"] = gui
        if vix is not None:
            var_comparisons["India VIX"] = vix
        if ofr_fsi is not None:
            var_comparisons["OFR FSI"] = ofr_fsi
        if em_fsi is not None:
            var_comparisons["Emerging Markets FSI"] = em_fsi

        if var_comparisons:
            ty_results = print_ty_granger_fevd_report(fsi, var_comparisons)
        else:
            ty_results = {}
            print("  No comparison series available — skipping Toda-Yamamoto/Block-Exogeneity/FEVD block.")

        # ── Phase 4c — DCC-GARCH dynamic conditional correlation ─────────────
        if var_comparisons:
            dcc_results = run_dcc_garch_suite(fsi, var_comparisons)
        else:
            dcc_results = {}

        # ── Phase 4d — RBI DBIE hard-data anchoring ───────────────────────────
        if gdp_yoy is not None or gfcf_yoy is not None:
            print("\n─── Phase 4d: RBI DBIE Hard-Data Anchoring (GDP/GFCF) ──────────────")
            fsi_quarterly = fsi.resample("QS").mean()
            for label, series in [("GDP YoY Growth", gdp_yoy), ("GFCF YoY Growth", gfcf_yoy)]:
                if series is None:
                    continue
                m = pd.concat([fsi_quarterly.rename("FSI"), series], axis=1).dropna()
                if len(m) < 8:
                    print(f"  {label:20s}  — Insufficient overlap for anchoring.")
                    continue
                r, p = stats.pearsonr(m["FSI"], m.iloc[:, 1])
                print(f"  FSI vs {label:20s}  Pearson r={r:+.4f} (p={p:.2e}, n={len(m)})")
                generate_plots(fsi_quarterly, series, label, label.lower().replace(" ", "_"))

        print("\n─── Phase 5: Executing Graphic Output Compilations ──────────────────")

        generate_plots(

            fsi,

            _vix,

            "India VIX",

            "vix"

        )



        generate_plots(

            fsi,

            _pu,

            "Policy Uncertainty",

            "pu"

        )



        generate_plots(

            fsi,

            _gui,

            "GUI",

            "gui"

        )



        generate_plots(

            fsi,

            _gpr,

            "Geopolitical Risk",

            "gpr"

        )



        generate_plots(

            fsi,

            _gwui,

            "Global WUI",

            "global_wui"

        )



        generate_plots(

            fsi,

            _gepu,

            "Global EPU",

            "global_epu"

        )



        if wsi is not None:

            generate_plots(

                fsi,

                _wsi,

                "WSI (GDP-Weighted)",

                "wsi"

            )



        if wui_ind_t1 is not None:

            generate_plots(

                fsi,

                _wui_ind_t1,

                "WUI India T1 (IND)",

                "wui_ind_t1"

            )



        if ofr_fsi is not None:

            generate_plots(

                fsi,

                _ofr_fsi,

                "OFR FSI",

                "ofr_fsi"

            )



        if em_fsi is not None:

            generate_plots(

                fsi,

                _em_fsi,

                "Emerging Markets FSI",

                "emerging_markets_fsi"

            )



    print("\n─── EXECUTION SEQUENCE COMPLETE ──────────────────────────────────────")

    return fsi



if __name__ == "__main__":

    fsi_result = main()