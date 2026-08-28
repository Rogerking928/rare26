"""
RARE26 離線評分器。

`official_score` 是 RARE25-Baselines/evaluation_Grand-Challenge.py 的逐行複製
（bootstrap_metrics），行為必須完全一致 —— 這是唯一可信的目標函數。

其餘是診斷工具，用來回答「該優化什麼」：
  - PPV@90R 的分母幾乎全是偽陽性，見 ppv_from_fpr()
  - 門檻由抽樣正樣本的第 10 百分位決定，見 threshold_at_recall()
  - 模型比較要用配對 bootstrap，見 paired_compare()
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


# ---------------------------------------------------------------- 官方 metric

def official_score(y_true, y_pred, n_iterations=1000, imbalance_ratio=100, rng=None):
    """官方 bootstrap_metrics 的複製，回傳完整 summary dict。

    與官方唯一的差別：接受 rng 以便重現。官方沒給種子，所以每次跑分數會抖，
    這正是我們做 model selection 時必須自己固定的東西。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng() if rng is None else rng

    ndbe_indices = np.where(y_true == 0)[0]
    neoplasia_indices = np.where(y_true == 1)[0]

    auc_full = roc_auc_score(y_true, y_pred)
    auprc_full = average_precision_score(y_true, y_pred)
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    ppv_90_full = np.interp(0.9, recalls[::-1], precisions[::-1])

    n_ndbe = len(ndbe_indices)
    n_neoplasia_to_sample = max(1, int(n_ndbe / imbalance_ratio))

    boot = np.empty((n_iterations, 3))
    for i in range(n_iterations):
        sampled = np.concatenate([
            ndbe_indices,                                    # 負樣本固定全取，不重抽
            rng.choice(neoplasia_indices, size=n_neoplasia_to_sample, replace=True),
        ])
        yt, yp = y_true[sampled], y_pred[sampled]
        p, r, _ = precision_recall_curve(yt, yp)
        boot[i] = (roc_auc_score(yt, yp),
                   average_precision_score(yt, yp),
                   np.interp(0.9, r[::-1], p[::-1]))

    return {
        "Score": float(np.median(boot[:, 2])),
        "PPV@90RECALL": float(np.median(boot[:, 2])),
        "PPV@90RECALL 95% CI Lower Bound": float(np.percentile(boot[:, 2], 2.5)),
        "PPV@90RECALL 95% CI Upper Bound": float(np.percentile(boot[:, 2], 97.5)),
        "AUROC": float(np.median(boot[:, 0])),
        "AUPRC": float(np.median(boot[:, 1])),
        "AUROC Full Dataset": float(auc_full),
        "AUPRC Full Dataset": float(auprc_full),
        "PPV@90RECALL Full Dataset": float(ppv_90_full),
        "_boot": boot,
    }


# ------------------------------------------------------------------- 代數

def ppv_from_fpr(fpr, recall=0.9, prevalence=0.01):
    """PPV = recall*p / (recall*p + fpr*(1-p))。

    盛行率 1%、recall 0.9 時 ≈ 0.009 / (0.009 + fpr)。
    這說明分數是 FPR 的函數，AUROC 不是。
    """
    fpr = np.asarray(fpr, dtype=float)
    tp = recall * prevalence
    fp = fpr * (1.0 - prevalence)
    return tp / (tp + fp)


def fpr_at_sensitivity(y_true, y_scores, sensitivity=0.9):
    """在指定敏感度下的偽陽性率 —— 真正要壓的那個數字。"""
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    i = np.searchsorted(tpr, sensitivity, side="left")
    i = min(i, len(tpr) - 1)
    return float(fpr[i]), float(thr[i])


def threshold_at_recall(y_true, y_scores, recall=0.9):
    """達到指定 recall 所需的門檻 = 正樣本分數的第 (1-recall) 百分位。"""
    pos = np.asarray(y_scores)[np.asarray(y_true) == 1]
    return float(np.quantile(pos, 1.0 - recall))


# ------------------------------------------------------------------- 診斷

def hard_positives(y_true, y_scores, recall=0.9, n=20):
    """回傳分數最低的正樣本索引 —— 門檻就是被這些人拉低的。

    因為官方對正樣本做「放回」重抽，池子裡任何一個爛正樣本都會被反覆抽中，
    對中位數分數的影響遠大於它在原始資料裡的一份權重。
    """
    y_true = np.asarray(y_true); y_scores = np.asarray(y_scores)
    pos_idx = np.where(y_true == 1)[0]
    order = pos_idx[np.argsort(y_scores[pos_idx])]
    thr = threshold_at_recall(y_true, y_scores, recall)
    return {
        "threshold_at_recall": thr,
        "worst_positive_indices": order[:n].tolist(),
        "worst_positive_scores": y_scores[order[:n]].round(4).tolist(),
        "n_positives_below_threshold": int((y_scores[pos_idx] < thr).sum()),
    }


def report(y_true, y_scores, rng=None, n_iterations=1000):
    """一次印出所有該看的數字。"""
    s = official_score(y_true, y_scores, n_iterations=n_iterations, rng=rng)
    fpr, thr = fpr_at_sensitivity(y_true, y_scores, 0.9)
    lines = [
        f"Score (median PPV@90R, 1% prev) : {s['Score']:.4f}",
        f"  95% CI                        : [{s['PPV@90RECALL 95% CI Lower Bound']:.4f}, "
        f"{s['PPV@90RECALL 95% CI Upper Bound']:.4f}]",
        f"AUROC (full)                    : {s['AUROC Full Dataset']:.4f}   <- 不計名次",
        f"AUPRC (full)                    : {s['AUPRC Full Dataset']:.4f}   <- 不計名次",
        "",
        f"FPR @ 90% sensitivity           : {fpr:.5f}   <- 真正要壓的數字",
        f"  預測 PPV from FPR             : {float(ppv_from_fpr(fpr)):.4f}",
        f"  門檻                          : {thr:.4f}",
    ]
    hp = hard_positives(y_true, y_scores)
    lines += [
        "",
        f"正樣本第 10 百分位分數          : {hp['threshold_at_recall']:.4f}",
        f"最差的 10 個正樣本分數          : "
        f"{[round(v, 3) for v in hp['worst_positive_scores'][:10]]}",
    ]
    return "\n".join(lines), s


# --------------------------------------------------------------- 模型比較

def paired_compare(y_true, scores_a, scores_b, n_iterations=1000, seed=0,
                   imbalance_ratio=100):
    """用同一批 bootstrap 抽樣比較兩個模型。

    官方評分每次重抽，兩個模型各跑一次官方評分再相減，差值會被抽樣噪音淹沒。
    配對抽樣（兩者共用同一組 sampled indices）把那個噪音消掉，
    才看得出 +0.005 這種量級的真實差異。
    """
    y_true = np.asarray(y_true)
    a = np.asarray(scores_a); b = np.asarray(scores_b)
    rng = np.random.default_rng(seed)

    ndbe = np.where(y_true == 0)[0]
    neo = np.where(y_true == 1)[0]
    m = max(1, int(len(ndbe) / imbalance_ratio))

    da = np.empty(n_iterations); db = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = np.concatenate([ndbe, rng.choice(neo, size=m, replace=True)])
        yt = y_true[idx]
        for arr, out in ((a, da), (b, db)):
            p, r, _ = precision_recall_curve(yt, arr[idx])
            out[i] = np.interp(0.9, r[::-1], p[::-1])

    diff = da - db
    return {
        "score_a": float(np.median(da)),
        "score_b": float(np.median(db)),
        "median_diff": float(np.median(diff)),
        "diff_95CI": (float(np.percentile(diff, 2.5)),
                      float(np.percentile(diff, 97.5))),
        "P(a>b)": float((diff > 0).mean()),
    }
