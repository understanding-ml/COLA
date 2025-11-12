import numpy as np
import pandas as pd
import ot  # POT
from cem.coarsen import coarsen
from cem.match import match
from cem.imbalance import L1
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from .base_matcher import BaseMatcher




#########################################
# Utility Functions: DataFrame conversion, numeralization and missing value imputation
#########################################
def _ensure_df(x, prefix: str = "x") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    x = np.asarray(x)
    cols = [f"{prefix}_{i}" for i in range(x.shape[1])]
    return pd.DataFrame(x, columns=cols)

def _to_numeric_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Convert each column to numeric (set NaN for non-convertible values), then impute NaN with column median."""
    out = df.apply(lambda col: col if np.issubdtype(col.dtype, np.number)
                   else pd.to_numeric(col, errors="coerce"))
    med = out.median(numeric_only=True)
    out = out.fillna(med)
    return out


#########################################
# Diagnostics: Treatment/control count within strata (explicit observed=True)
#########################################
def strata_diagnostics(X_coarse: pd.DataFrame, T_name: str) -> pd.DataFrame:
    cols_no_T = [c for c in X_coarse.columns if c != T_name]
    strata = pd.MultiIndex.from_frame(X_coarse[cols_no_T].astype("object"))
    sid, _ = pd.factorize(strata)
    T = X_coarse[T_name].to_numpy()

    df = pd.DataFrame({"sid": sid, "T": T})
    stat = (
        df.groupby("sid", observed=True)["T"]
          .agg(n="size",
               n_treat=lambda s: int((s == 1).sum()),
               n_ctrl=lambda s: int((s == 0).sum()))
    )
    stat["has_both"] = (stat["n_treat"] > 0) & (stat["n_ctrl"] > 0)
    return stat.sort_values("n", ascending=False)


#########################################
# Main Class: Non-strict same-stratum + OT补齐
#########################################
class CounterfactualSoftCEMPolicy(BaseMatcher):
    """
    Pipeline:
    1) Merge factual/counterfactual, covariates + T column; coarsen(X_all, T, "l1") to get coarsened table;
    2) Define strata by "all coarsened columns except T", assign uniform probability to counterfactuals within same stratum;
    3) For unmatched rows (same stratum has no counterfactuals), use OT to fill from global counterfactuals;
    4) Only normalize rows with non-zero sum to avoid division by zero warnings.
    """

    def __init__(self,
                 x_factual,
                 x_counterfactual,
                 treatment_name: str = "T",
                 scale: bool = True,
                 use_sinkhorn: bool = False,
                 sinkhorn_eps: float = 0.05,
                 cost_clip_quantile: Optional[float] = 0.9):
        """
        Parameters
        ----------
        x_factual, x_counterfactual : array-like or DataFrame
            Covariate matrix (excluding outcome).
        treatment_name : str
            Treatment indicator column name, passed to coarsen/match/L1.
        scale : bool
            Whether to apply StandardScaler normalization to covariates before constructing OT cost.
        use_sinkhorn : bool
            If True, use Sinkhorn-OT (GPU-compatible/more robust), if False use EMD.
        sinkhorn_eps : float
            Entropy regularization parameter for Sinkhorn.
        cost_clip_quantile : float or None
            Clip OT cost matrix by quantile (e.g., 0.9) to suppress extreme distance extrapolation; None means no clipping.
        """
        super().__init__(x_factual, x_counterfactual)
        self.treatment_name = treatment_name
        self.scale = scale
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.cost_clip_quantile = cost_clip_quantile

    ############################
    # Core: Construct coarsened table and strata id
    ############################
    def _coarsen_and_strata(self,
                            xf: pd.DataFrame,
                            xr: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, int, int]:
        # Align columns (union + fill missing)
        all_cols = sorted(set(xf.columns).union(set(xr.columns)))
        xf = xf.reindex(columns=all_cols)
        xr = xr.reindex(columns=all_cols)

        # Concatenate and add T
        X_all = pd.concat([xf, xr], axis=0, ignore_index=True)
        T = np.r_[np.zeros(len(xf), dtype=int), np.ones(len(xr), dtype=int)]
        X_all[self.treatment_name] = T

        # Coarsen
        Xc = coarsen(X_all, self.treatment_name, "l1")

        # strata id (all coarsened columns except T)
        cols_no_T = [c for c in Xc.columns if c != self.treatment_name]
        tuples = pd.MultiIndex.from_frame(Xc[cols_no_T].astype("object"))
        strata_ids, _ = pd.factorize(tuples)

        return Xc, strata_ids, len(xf), len(xr)

    ############################
    # Same-stratum priority uniform allocation (without normalization)
    ############################
    def _same_stratum_assign(self,
                             strata_ids: np.ndarray,
                             n_f: int,
                             n_r: int,
                             T: np.ndarray) -> np.ndarray:
        p = np.zeros((n_f, n_r), dtype=float)
        # Factual part indices: 0..n_f-1; Counterfactual part indices: n_f..n_f+n_r-1
        for sid in np.unique(strata_ids):
            idx_in_s = np.where(strata_ids == sid)[0]
            f_in_s = [i for i in idx_in_s if (i < n_f) and (T[i] == 0)]
            r_in_s = [i - n_f for i in idx_in_s if (n_f <= i < n_f + n_r) and (T[i] == 1)]
            if len(f_in_s) == 0 or len(r_in_s) == 0:
                continue
            w = 1.0 / len(r_in_s)
            for fi in f_in_s:
                for rj in r_in_s:
                    p[fi, rj] = w
        return p

    ############################
    # Construct numerical feature matrix (with standardization)
    ############################
    def _build_numeric_arrays(self,
                              xf: pd.DataFrame,
                              xr: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.concat([xf, xr], axis=0, ignore_index=True)
        df_num = _to_numeric_and_impute(df)
        X = df_num.to_numpy(dtype=float)
        n_f = len(xf)
        n_r = len(xr)
        Xf = X[:n_f]
        Xr = X[n_f:n_f + n_r]
        if self.scale:
            scaler = StandardScaler()
            X_all_scaled = scaler.fit_transform(np.vstack([Xf, Xr]))
            Xf = X_all_scaled[:n_f]
            Xr = X_all_scaled[n_f:n_f + n_r]
        return Xf, Xr

    ############################
    # Fill unmatched rows with OT (row-wise EMD / Sinkhorn)
    ############################
    def _fill_unmatched_rows_with_ot(self,
                                     p: np.ndarray,
                                     Xf: np.ndarray,
                                     Xr: np.ndarray) -> np.ndarray:
        n_f, n_r = p.shape
        row_sum = p.sum(axis=1)
        unmatched = np.where(row_sum == 0)[0]
        if len(unmatched) == 0:
            return p

        # Cost matrix (can be clipped by quantile)
        C = ot.dist(Xf[unmatched], Xr, metric="euclidean")  # (n_unmatched, n_r)
        if self.cost_clip_quantile is not None:
            finite_C = C[np.isfinite(C)]
            if finite_C.size > 0:
                thr = np.quantile(finite_C, self.cost_clip_quantile)
                C = np.minimum(C, thr)

        # Solve transport plan row-by-row and fill back
        for k, i in enumerate(unmatched):
            a = np.array([1.0])             # Mass 1 for this factual row
            b = np.ones(n_r) / n_r          # Target uniform (can be changed to prior column weights)
            c_i = C[k:k+1, :]               # (1, n_r)

            if self.use_sinkhorn:
                G = ot.sinkhorn(a, b, c_i, reg=self.sinkhorn_eps)
            else:
                G = ot.emd(a, b, c_i)

            gi = G[0]
            s = gi.sum()
            if s > 0 and np.isfinite(s):
                p[i, :] = gi / s
            else:
                p[i, :] = np.ones(n_r) / n_r  # Fallback
        return p

    ############################
    # Safe row normalization (only for non-zero rows)
    ############################
    @staticmethod
    def _row_normalize_safe(p: np.ndarray) -> np.ndarray:
        row_sum = p.sum(axis=1, keepdims=True)
        mask = (row_sum > 0) & np.isfinite(row_sum)
        out = np.zeros_like(p)
        out[mask[:, 0]] = p[mask[:, 0]] / row_sum[mask]
        return out

    #########################################################
    # External call: Compute factual-counterfactual probability matrix (non-strict same-stratum + OT)
    #########################################################
    def compute_prob_matrix_of_factual_and_counterfactual(self) -> np.ndarray:
        # 1) Unify to DataFrame
        xf = _ensure_df(self.x_factual, prefix="f")
        xr = _ensure_df(self.x_counterfactual, prefix="r")

        # 2) Coarsen + strata
        Xc, strata_ids, n_f, n_r = self._coarsen_and_strata(xf, xr)
        T = Xc[self.treatment_name].to_numpy()

        # 3) Same-stratum priority uniform allocation (unmatched rows sum to 0 at this point)
        p = self._same_stratum_assign(strata_ids, n_f, n_r, T)

        # 4) Construct numerical features for OT (with missing imputation and optional standardization)
        Xf, Xr = self._build_numeric_arrays(xf, xr)

        # 5) Fill unmatched rows with OT (row-wise), then safe normalize
        p = self._fill_unmatched_rows_with_ot(p, Xf, Xr)
        p = self._row_normalize_safe(p)
        return p

    #########################################################
    # Baseline imbalance assessment (optional, for checking coarsening effect)
    #########################################################
    def baseline_imbalance(self):
        xf = _ensure_df(self.x_factual, prefix="f")
        xr = _ensure_df(self.x_counterfactual, prefix="r")
        all_cols = sorted(set(xf.columns).union(set(xr.columns)))
        xf = xf.reindex(columns=all_cols)
        xr = xr.reindex(columns=all_cols)

        X_all = pd.concat([xf, xr], axis=0, ignore_index=True)
        T = np.r_[np.zeros(len(xf), dtype=int), np.ones(len(xr), dtype=int)]
        X_all[self.treatment_name] = T

        Xc = coarsen(X_all, self.treatment_name, "l1")
        w = match(Xc, self.treatment_name)
        l1_unw = L1(Xc, self.treatment_name)
        l1_w = L1(Xc, self.treatment_name, w)
        return l1_unw, l1_w, w

    #########################################################
    # Strata diagnostics (optional, for understanding common support situation)
    #########################################################
    def strata_report(self) -> pd.DataFrame:
        xf = _ensure_df(self.x_factual, prefix="f")
        xr = _ensure_df(self.x_counterfactual, prefix="r")
        # Reuse coarsening
        Xc, _, _, _ = self._coarsen_and_strata(xf, xr)
        return strata_diagnostics(Xc, self.treatment_name)
