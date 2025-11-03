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
# 工具函数：DataFrame 化、数值化与缺失填充
#########################################
def _ensure_df(x, prefix: str = "x") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    x = np.asarray(x)
    cols = [f"{prefix}_{i}" for i in range(x.shape[1])]
    return pd.DataFrame(x, columns=cols)

def _to_numeric_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """将各列转为数值（无法转换的置 NaN），再用列中位数填充 NaN。"""
    out = df.apply(lambda col: col if np.issubdtype(col.dtype, np.number)
                   else pd.to_numeric(col, errors="coerce"))
    med = out.median(numeric_only=True)
    out = out.fillna(med)
    return out


#########################################
# 诊断：分层内处理/对照计数（显式 observed=True）
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
# 主类：不严格同层 + OT 补齐
#########################################
class CounterfactualSoftCEMPolicy(BaseMatcher):
    """
    流程：
    1) 合并事实/反事实，自变量 + T 列；coarsen(X_all, T, "l1") 得到粗化表；
    2) 按“除 T 外所有粗化列”定义 strata，同层内对反事实均匀分配概率；
    3) 对未匹配行（同层没有任何反事实者），使用 OT 在全局反事实上补齐；
    4) 仅对非零行做行归一化，避免除零告警。
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
            协变量矩阵（不包含 outcome）。
        treatment_name : str
            处理标记列名，传入 coarsen/match/L1。
        scale : bool
            是否在构造 OT 成本前对协变量做 StandardScaler 标准化。
        use_sinkhorn : bool
            True 则用 Sinkhorn-OT（可 GPU/更稳健），False 用 EMD。
        sinkhorn_eps : float
            Sinkhorn 的熵正则参数。
        cost_clip_quantile : float or None
            对 OT 成本矩阵按分位数截断（如 0.9），抑制极远距离外推；None 则不截断。
        """
        super().__init__(x_factual, x_counterfactual)
        self.treatment_name = treatment_name
        self.scale = scale
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.cost_clip_quantile = cost_clip_quantile

    ############################
    # 核心：构造粗化表与 strata id
    ############################
    def _coarsen_and_strata(self,
                            xf: pd.DataFrame,
                            xr: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, int, int]:
        # 对齐列（并集 + 补缺）
        all_cols = sorted(set(xf.columns).union(set(xr.columns)))
        xf = xf.reindex(columns=all_cols)
        xr = xr.reindex(columns=all_cols)

        # 拼接并加 T
        X_all = pd.concat([xf, xr], axis=0, ignore_index=True)
        T = np.r_[np.zeros(len(xf), dtype=int), np.ones(len(xr), dtype=int)]
        X_all[self.treatment_name] = T

        # 粗化
        Xc = coarsen(X_all, self.treatment_name, "l1")

        # strata id（除 T 外所有粗化列）
        cols_no_T = [c for c in Xc.columns if c != self.treatment_name]
        tuples = pd.MultiIndex.from_frame(Xc[cols_no_T].astype("object"))
        strata_ids, _ = pd.factorize(tuples)

        return Xc, strata_ids, len(xf), len(xr)

    ############################
    # 同层优先的均匀分配（不归一）
    ############################
    def _same_stratum_assign(self,
                             strata_ids: np.ndarray,
                             n_f: int,
                             n_r: int,
                             T: np.ndarray) -> np.ndarray:
        p = np.zeros((n_f, n_r), dtype=float)
        # 事实部分索引：0..n_f-1；反事实部分索引：n_f..n_f+n_r-1
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
    # 构造数值特征矩阵（含标准化）
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
    # OT 补齐未匹配行（逐行 EMD / Sinkhorn）
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

        # 成本矩阵（可按分位数截断）
        C = ot.dist(Xf[unmatched], Xr, metric="euclidean")  # (n_unmatched, n_r)
        if self.cost_clip_quantile is not None:
            finite_C = C[np.isfinite(C)]
            if finite_C.size > 0:
                thr = np.quantile(finite_C, self.cost_clip_quantile)
                C = np.minimum(C, thr)

        # 逐行求运输计划并回填
        for k, i in enumerate(unmatched):
            a = np.array([1.0])             # 该事实行的质量 1
            b = np.ones(n_r) / n_r          # 目标均匀（也可改为先验列权重）
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
                p[i, :] = np.ones(n_r) / n_r  # 兜底
        return p

    ############################
    # 安全的行归一化（仅对非零行）
    ############################
    @staticmethod
    def _row_normalize_safe(p: np.ndarray) -> np.ndarray:
        row_sum = p.sum(axis=1, keepdims=True)
        mask = (row_sum > 0) & np.isfinite(row_sum)
        out = np.zeros_like(p)
        out[mask[:, 0]] = p[mask[:, 0]] / row_sum[mask]
        return out

    #########################################################
    # 外部调用：计算事实-反事实概率矩阵（非严格同层 + OT）
    #########################################################
    def compute_prob_matrix_of_factual_and_counterfactual(self) -> np.ndarray:
        # 1) 统一为 DataFrame
        xf = _ensure_df(self.x_factual, prefix="f")
        xr = _ensure_df(self.x_counterfactual, prefix="r")

        # 2) 粗化 + strata
        Xc, strata_ids, n_f, n_r = self._coarsen_and_strata(xf, xr)
        T = Xc[self.treatment_name].to_numpy()

        # 3) 同层优先均匀分配（此时未匹配行和为 0）
        p = self._same_stratum_assign(strata_ids, n_f, n_r, T)

        # 4) 为 OT 构造数值特征（含缺失填充与可选标准化）
        Xf, Xr = self._build_numeric_arrays(xf, xr)

        # 5) 用 OT 补齐未匹配行（逐行），然后安全归一化
        p = self._fill_unmatched_rows_with_ot(p, Xf, Xr)
        p = self._row_normalize_safe(p)
        return p

    #########################################################
    # 基线不平衡评估（可选，用于检查粗化效果）
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
    # 分层诊断（可选，用于了解共同支持情况）
    #########################################################
    def strata_report(self) -> pd.DataFrame:
        xf = _ensure_df(self.x_factual, prefix="f")
        xr = _ensure_df(self.x_counterfactual, prefix="r")
        # 复用粗化
        Xc, _, _, _ = self._coarsen_and_strata(xf, xr)
        return strata_diagnostics(Xc, self.treatment_name)
