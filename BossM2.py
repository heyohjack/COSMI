# -*- coding: utf-8 -*-
import sys
import time
import random
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from causallearn.score.LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import csv
import os


# =========================================================
# ===============  GST / GSTNode（原理同你现有版本） ======
# =========================================================
class GSTNode:
    def __init__(self, tree, add=None, score=None):
        if score is None:
            score = -tree.score.score_nocache(tree.vertex, [])
        self.tree = tree
        self.add = add
        self.grow_score = score
        self.shrink_score = score
        self.branches = None
        self.remove = None

    def __lt__(self, other):
        return self.grow_score < other.grow_score

    def grow(self, available, parents):
        self.branches = []
        for add in available:
            parents.append(add)
            score = -self.tree.score.score_nocache(self.tree.vertex, parents)
            parents.remove(add)
            branch = GSTNode(self.tree, add, score)
            if score > self.grow_score:
                self.branches.append(branch)
        self.branches.sort()

    def shrink(self, parents):
        self.remove = []
        while True:
            best = None
            for remove in [parent for parent in parents]:
                parents.remove(remove)
                score = -self.tree.score.score_nocache(self.tree.vertex, parents)
                parents.append(remove)
                if score > self.shrink_score:
                    self.shrink_score = score
                    best = remove
            if best is None:
                break
            self.remove.append(best)
            parents.remove(best)

    def trace(self, prefix, available, parents):
        if self.branches is None:
            self.grow(available, parents)
        for branch in self.branches:
            available.remove(branch.add)
            if branch.add in prefix:
                parents.append(branch.add)
                return branch.trace(prefix, available, parents)
        if self.remove is None:
            self.shrink(parents)
            return self.shrink_score
        for remove in self.remove:
            parents.remove(remove)
        return self.shrink_score


class GST:
    def __init__(self, vertex, score):
        self.vertex = vertex
        self.score = score
        self.root = GSTNode(self)
        self.forbidden = [vertex]
        self.required = []

    def trace(self, prefix, parents=None):
        if parents is None:
            parents = []
        available = [i for i in range(self.score.data.shape[1]) if i not in self.forbidden]
        return self.root.trace(prefix, available, parents)

    def set_knowledge(self, forbidden, required):
        self.forbidden = forbidden
        self.required = required
        self.reset()

    def reset(self):
        self.root = GSTNode(self)


# =========================================================
# ==================== 打分构造器 =========================
# =========================================================
def _make_score_class(X: np.ndarray, score_func: str, parameters: Optional[Dict[str, Any]]):
    if score_func == "local_score_CV_general":
        if parameters is None:
            parameters = {"kfold": 10, "lambda": 0.01}
        return LocalScoreClass(data=X, local_score_fun=local_score_cv_general, parameters=parameters)
    elif score_func == "local_score_marginal_general":
        return LocalScoreClass(data=X, local_score_fun=local_score_marginal_general, parameters={})
    elif score_func == "local_score_CV_multi":
        if parameters is None:
            parameters = {"kfold": 10, "lambda": 0.01, "dlabel": {}}
            for i in range(X.shape[1]):
                parameters["dlabel"][f"{i}"] = i
        return LocalScoreClass(data=X, local_score_fun=local_score_cv_multi, parameters=parameters)
    elif score_func == "local_score_marginal_multi":
        if parameters is None:
            parameters = {"dlabel": {}}
            for i in range(X.shape[1]):
                parameters["dlabel"][f"{i}"] = i
        return LocalScoreClass(data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)
    elif score_func in ("local_score_BIC", "local_score_BIC_from_cov"):
        if score_func == "local_score_BIC":
            warnings.warn("Using 'local_score_BIC_from_cov' instead for efficiency")
        if parameters is None:
            parameters = {"lambda_value": 2}
        return LocalScoreClass(data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters)
    elif score_func == "local_score_BDeu":
        return LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    else:
        raise Exception("Unknown score function!")


# =========================================================
# ============== 排列搜索：单调上升 + 约束 =================
# =========================================================
def global_order_score(order: List[int], gsts: List[GST]) -> float:
    """整条排列的总得分。"""
    prefix = []
    total = 0.0
    for w in order:
        total += gsts[w].trace(prefix)
        prefix.append(w)
    return float(total)


def _better_mutation_constrained(
    v: int,
    order: List[int],
    gsts: List[GST],
    must_before: Optional[tuple],
    tol: float = 1e-6,
) -> bool:
    """
    在变量 v 上尝试改变其在排列中的位置：
    - 维持 X<Y 约束（must_before=(x_idx,y_idx)）
    - 只有当全局得分严格提升才接受移动（避免死循环）
    """
    i = order.index(v)
    p = len(order)
    scores = np.zeros(p + 1)

    prefix = []
    part = 0.0
    for j, w in enumerate(order):
        scores[j] = gsts[v].trace(prefix) + part
        if v != w:
            part += gsts[w].trace(prefix)
            prefix.append(w)
    scores[p] = gsts[v].trace(prefix) + part

    curr_local = scores[i]
    old_global = global_order_score(order, gsts)

    bests = np.argsort(scores)[::-1]  # 从高到低尝试
    for j in bests:
        if scores[j] <= curr_local + tol:
            break

        tmp = order.copy()
        tmp.remove(v)
        ins = j - int(j > i)
        tmp.insert(ins, v)

        if must_before is not None:
            x_var, y_var = must_before
            if tmp.index(x_var) >= tmp.index(y_var):
                continue

        new_global = global_order_score(tmp, gsts)
        if new_global > old_global + tol:
            order[:] = tmp
            return True

    return False


def best_order_with_constraint(
    X: np.ndarray,
    node_names: List[str],
    score_func: str = "local_score_BIC_from_cov",
    parameters: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    X_name: Optional[str] = None,
    Y_name: Optional[str] = None,
    max_passes: int = 200,
):
    """仅搜索最优排列，强制 X 在 Y 之前，且每次更新需提升全局分数。"""
    n, p = X.shape
    score = _make_score_class(X, score_func, parameters)

    name_to_idx = {name: i for i, name in enumerate(node_names)}
    x_var = name_to_idx[X_name] if X_name is not None else None
    y_var = name_to_idx[Y_name] if Y_name is not None else None
    must_before = (x_var, y_var) if (x_var is not None and y_var is not None) else None

    order = list(range(p))
    gsts = [GST(v, score) for v in order]
    variables = list(order)

    t0 = time.perf_counter()
    prev_global = global_order_score(order, gsts)

    for pass_id in range(1, max_passes + 1):
        improved = False
        random.shuffle(variables)

        for v in variables:
            if _better_mutation_constrained(v, order, gsts, must_before):
                improved = True

        curr_global = global_order_score(order, gsts)
        if verbose:
            sys.stdout.write(f"\rOrder search pass #{pass_id:3d} | score: {curr_global:.6f}  ")
            sys.stdout.flush()

        if not improved or curr_global <= prev_global + 1e-9:
            break
        prev_global = curr_global

    if verbose:
        dt = time.perf_counter() - t0
        sys.stdout.write(f"\nOrder search completed in {dt:.2f}s\n")
        sys.stdout.flush()

    return order, name_to_idx


# =========================================================
# ================= 中介检验（只做“有/无”） ================
# =========================================================
def _ols_bic(y, X):
    """计算 OLS 的 BIC（含截距），返回 (bic, coef)。"""
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)
    n = y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xc = np.column_stack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    rss = float((resid.T @ resid))
    k = Xc.shape[1]
    rss = max(rss, 1e-12)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic, beta.flatten()


def find_mediators_on_order(
    df: pd.DataFrame,
    node_names: List[str],
    order: List[int],
    X_name: str,
    Y_name: str,
    min_coef: float = 1e-3,
    bic_threshold: float = 2.0,   # 达到该ΔBIC即认为“存在中介作用”
    standardize: bool = True,
) -> pd.DataFrame:
    """
    仅在 'order' 中位于 X 与 Y 之间的变量里识别中介。
    只输出“是否有中介作用（Yes/No）”，不区分部分/完全。
    """
    Z = df.copy()
    if standardize:
        for c in df.columns:
            s = Z[c].std(ddof=0)
            if s > 0:
                Z[c] = (Z[c] - Z[c].mean()) / s

    idx = {name: i for i, name in enumerate(node_names)}
    px = order.index(idx[X_name])
    py = order.index(idx[Y_name])
    if px >= py:
        return pd.DataFrame([])

    between_vars = [node_names[order[k]] for k in range(px + 1, py)]
    if not between_vars:
        return pd.DataFrame([], columns=[
            "candidate", "a (X→M)", "b|X (M→Y|X)",
            "BIC(Y~X)", "BIC(M~X)+BIC(Y~M)", "BIC(Y~X+M)",
            "ΔBIC_path", "ΔBIC_add",
            "mediator?"
        ])

    X_col, Y_col = X_name, Y_name
    # 基线：Y ~ X
    bic_Y_X, beta_A = _ols_bic(Z[Y_col].values, Z[X_col].values)

    rows = []

    bic_path_threshold = 1000
    for M in between_vars:
        # a: X -> M
        bic_M_X, beta_M_X = _ols_bic(Z[M].values, Z[X_col].values)
        a = float(beta_M_X[1])

        # Y ~ M（路径证据第二段）
        bic_Y_M, beta_Y_M = _ols_bic(Z[Y_col].values, Z[M].values)

        # b|X: Y ~ X + M
        bic_Y_XM, beta_Y_XM = _ols_bic(Z[Y_col].values, Z[[X_col, M]].values)
        b_withX = float(beta_Y_XM[2])

        # 两条证据：结构型 与 增量型
        bic_B1_total = bic_M_X + bic_Y_M
        delta_bic_path = bic_Y_X - bic_B1_total          # 结构：A vs (M~X + Y~M)
        delta_bic_add = bic_Y_X - bic_Y_XM              # 增量：A vs (Y~X+M)

        is_mediator = (
            abs(a) >= min_coef and
            abs(b_withX) >= min_coef and
            -delta_bic_path <= bic_path_threshold and
            delta_bic_add > bic_threshold
        )
        print(M)
        print("bic_Y_X", bic_Y_X)
        print("bic_B1_total", bic_B1_total)
        print("bic_Y_XM", bic_Y_XM)
        rows.append({
            "candidate": M,
            "a (X→M)": a,
            "b|X (M→Y|X)": b_withX,
            "BIC(Y~X)": bic_Y_X,
            "BIC(M~X)+BIC(Y~M)": bic_B1_total,
            "BIC(Y~X+M)": bic_Y_XM,
            "ΔBIC_path": delta_bic_path,
            "ΔBIC_add": delta_bic_add,
            "mediator?": bool(is_mediator),
        })

    table = pd.DataFrame(rows).sort_values(
        by=["mediator?", "ΔBIC_path", "ΔBIC_add"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    return table


# =========================================================
# ===================== 总控函数 ==========================
# =========================================================
def boss_mediator_only(
    df: pd.DataFrame,
    X_name: str,
    Y_name: str,
    score_func: str = "local_score_BIC_from_cov",
    parameters: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    build_min_graph: bool = True,
):
    """
    1) BOSS 思路找到最佳排列（强制 X 在 Y 之前，且全局得分单调提升）；
    2) 仅在 X 与 Y 之间识别“是否存在中介作用”的变量；
    3) （可选）构建仅含 X->Y 与 X->M->Y 的最小图。
    """
    X_np = df.values
    node_names = list(df.columns)

    order, name_to_idx = best_order_with_constraint(
        X_np, node_names,
        score_func=score_func,
        parameters=parameters,
        verbose=verbose,
        X_name=X_name, Y_name=Y_name,
    )

    table = find_mediators_on_order(
        df=df, node_names=node_names, order=order,
        X_name=X_name, Y_name=Y_name,
    )

    result = {
        "order": [node_names[i] for i in order],
        "X": X_name,
        "Y": Y_name,
        "mediators_table": table
    }

    if build_min_graph:
        nodes = [GraphNode(nm) for nm in node_names]
        name_to_node = {nm: nodes[i] for i, nm in enumerate(node_names)}
        G = GeneralGraph(nodes)

        # 固定约束：X -> Y
        G.add_directed_edge(name_to_node[X_name], name_to_node[Y_name])

        # 对被判为“有中介作用”的变量，连 X->M 与 M->Y
        if not table.empty:
            for _, r in table.iterrows():
                if bool(r["mediator?"]):
                    m = r["candidate"]
                    G.add_directed_edge(name_to_node[X_name], name_to_node[m])
                    G.add_directed_edge(name_to_node[m], name_to_node[Y_name])

        result["graph"] = G

    return result


# =========================================================
# ======================== 示例 ===========================
# =========================================================
if __name__ == "__main__":
    # 读取数据（按你的实际路径修改）
    result_file = os.path.join("results", "mediator_metrics.csv")

    # 初始化 CSV 文件并写入表头
    with open(result_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "data_file", "num_confounders", "num_mediators",
            "precision", "recall", "accuracy", "f1",
            "num_true_mediators", "num_predicted_mediators"
        ])

    precisions, recalls, accuracies, f1s = [], [], [], []

    for i in range(1, 11):
        for j in range(1, 11):
            df = pd.read_csv('./data/data'+str(i)+'_'+str(j)+'.csv')
            # 示例列名（按需要调整）
            confounder = [f"confounder{k}" for k in range(1, i + 1)]
            mediator = [f"mediator{k}" for k in range(1, j + 1)]
            df.columns = ["treatment", "outcome"] + mediator + confounder

            X_name = "treatment"       # 处理变量
            Y_name = "outcome"  # 结果变量

            out = boss_mediator_only(
                df, X_name=X_name, Y_name=Y_name,
                score_func="local_score_BIC_from_cov",
                parameters={"lambda_value": 2},
                verbose=True,
                build_min_graph=True,
            )

            print("\nBest order:", out["order"])
            print("\nMediator candidates (between X and Y only):")
            if out["mediators_table"].empty:
                print("(None between X and Y)")
            else:
                cols_show = [
                    "candidate", "mediator?",
                    "ΔBIC_path", "ΔBIC_add",
                    "a (X→M)", "b|X (M→Y|X)"
                ]
                print(out["mediators_table"][cols_show].to_string(index=False))

            # 可选：画“最小图”（需要 graphviz）
            # if "graph" in out:
            #     try:
            #         from causallearn.utils.GraphUtils import GraphUtils
            #         import matplotlib.image as mpimg
            #         import matplotlib.pyplot as plt
            #         import io
            #
            #         pyd = GraphUtils.to_pydot(out["graph"])
            #         tmp_png = pyd.create_png(f="png")
            #         fp = io.BytesIO(tmp_png)
            #         img = mpimg.imread(fp, format='png')
            #         plt.axis('off')
            #         plt.imshow(img)
            #         plt.show()
            #     except Exception as e:
            #         print("Plot skipped:", repr(e))

            # 预测的中介变量
            predicted_mediators = out["mediators_table"].query("`mediator?` == True")["candidate"].tolist()

            # 所有候选变量集合（包含中介与非中介）
            all_candidates = [f"mediator{k}" for k in range(1, j + 1)] + [f"confounder{k}" for k in range(1, i + 1)]

            # 转为二进制标签：1=中介, 0=非中介
            y_true = [1 if v in mediator else 0 for v in all_candidates]
            y_pred = [1 if v in predicted_mediators else 0 for v in all_candidates]

            # 计算评估指标
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            print(f"\nEvaluation for data{i}_{j}.csv:")
            print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  Accuracy: {accuracy:.4f}  F1: {f1:.4f}")

            precisions.append(precision)
            recalls.append(recall)
            accuracies.append(accuracy)
            f1s.append(f1)

            # 打印并保存结果
            print(f"data{i}_{j}.csv | Precision={precision:.3f}  Recall={recall:.3f}  "
                  f"Accuracy={accuracy:.3f}  F1={f1:.3f}")

            with open(result_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"data{i}_{j}.csv", i, j,
                    f"{precision:.4f}", f"{recall:.4f}", f"{accuracy:.4f}", f"{f1:.4f}",
                    len(mediator), len(predicted_mediators)
                ])

    summary = {
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "Accuracy": np.mean(accuracies),
        "F1": np.mean(f1s),
    }

    print("\n===== Overall Average Metrics =====")
    for k, v in summary.items():
        print(f"{k:10s}: {v:.4f}")

