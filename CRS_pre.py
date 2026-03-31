# import os
# # 必须放在 import numpy 或 sklearn 之前！
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# ==========================================
# 1. 实验参数与预处理自由配置区
# ==========================================
config = {
    "file_path": 'NGs.mat',    # 数据集路径
    "n_anchors": 500,            # 锚点数 p
    "beta": 0.1,                # 共正则化参数
    "gamma": 1.0,               # 熵正则化参数
    "max_iter": 50,              # 外层最大迭代次数
    "inner_max_iter": 10,        # 内层 (H_i, H*) 最大迭代次数
    "n_neighbors": 5,            # 锚点图近邻数 k
    "lr": 0.01,                  # 初始学习率
    "lr_decay": 0.95,            # 学习率衰减率
    "seed": 42,                  # 随机种子
    
    # --- 预处理自由切换 ---
    "row_norm": 'l2',            # 选项: 'l2', 'std', 'minmax', None (复现原结果设为 None)
    "use_pca": True,            # 选项: True, False (复现原结果设为 False)
    "pca_dim": 100,              
    
    # --- 锚点选取策略 ---
    "anchor_method": 'kmeans',   # 选项: 'kmeans', 'random', 'ensemble' (集成平均模式)
    "n_trials": 5                
}

class EvaluationMetrics:
    @staticmethod
    def get_metrics(y_true, y_pred):
        y_true = np.array(y_true).astype(int).flatten()
        y_pred = np.array(y_pred).astype(int).flatten()
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        idx_map = {row: col for row, col in zip(row_ind, col_ind)}
        y_pred_mapped = np.array([idx_map[label] if label in idx_map else label for label in y_pred])
        acc = metrics.accuracy_score(y_true, y_pred_mapped)
        purity = np.sum(np.max(w, axis=1)) / y_true.size
        f1 = metrics.f1_score(y_true, y_pred_mapped, average='macro', zero_division=0)
        return {"ACC": acc, "NMI": nmi, "ARI": ari, "Purity": purity, "F1": f1}

class CoReg_SMMSC_Final:
    def __init__(self, cfg):
        self.cfg = cfg
        self.p = cfg["n_anchors"]
        self.beta = cfg["beta"]
        self.gamma = cfg["gamma"]
        self.max_iter = cfg["max_iter"]
        self.inner_max_iter = cfg["inner_max_iter"] 
        self.n_neighbors = cfg["n_neighbors"]
        self.lr = cfg["lr"]
        self.lr_decay = cfg.get("lr_decay", 1.0)
        self.seed = cfg["seed"]
        self.alpha = None
        self.H_list = []  
        self.H_star = None 
        self.obj_history = []

    def _preprocess(self, X_list):
        # 如果不开启预处理，直接返回原列表，不进行类型转换以保持精度对齐
        if self.cfg.get("row_norm") is None and not self.cfg.get("use_pca"):
            return X_list
            
        new_X = []
        for Xv in X_list:
            if hasattr(Xv, 'toarray'): Xv = Xv.toarray()
            if self.cfg.get("use_pca"):
                pca = PCA(n_components=min(Xv.shape[1], self.cfg["pca_dim"]), random_state=self.seed)
                Xv = pca.fit_transform(Xv)
            method = self.cfg.get("row_norm")
            if method == 'l2': Xv = preprocessing.normalize(Xv, norm='l2', axis=1)
            elif method == 'std': Xv = preprocessing.scale(Xv)
            elif method == 'minmax': Xv = preprocessing.minmax_scale(Xv)
            new_X.append(Xv.astype(np.float64))
        return new_X

    def _single_W_construct(self, X_v, seed):
        if hasattr(X_v, 'toarray'): X_v = X_v.toarray() # 确保密集矩阵
        n = X_v.shape[0]
        method = self.cfg.get("anchor_method")
        
        if method == 'random':
            idx = np.random.RandomState(seed).choice(n, self.p, replace=False)
            Z_v = X_v[idx]
        else:
            # 【关键修改】将 n_init 从 1 改为 3，与 CoReg_SMMSC.py 保持完全一致
            kmeans = KMeans(n_clusters=self.p, random_state=seed, n_init=3).fit(X_v)
            Z_v = kmeans.cluster_centers_
        
        dist = np.sum(X_v**2, axis=1).reshape(-1, 1) + np.sum(Z_v**2, axis=1) - 2 * np.dot(X_v, Z_v.T)
        dist = np.maximum(dist, 0)
        
        W = np.zeros((n, self.p))
        for i in range(n):
            idx = np.argsort(dist[i, :])[:self.n_neighbors + 1]
            nearest_dist = dist[i, idx]
            d_kp1 = nearest_dist[-1]
            denom = self.n_neighbors * d_kp1 - np.sum(nearest_dist[:-1]) + 1e-10
            for j in range(self.n_neighbors):
                W[i, idx[j]] = (d_kp1 - nearest_dist[j]) / denom
        return W

    def _construct_W_tilde(self, X_v):
        n = X_v.shape[0]
        if self.cfg.get("anchor_method") == 'ensemble':
            n_trials = self.cfg.get("n_trials", 5)
            W_sum = np.zeros((n, self.p))
            for t in range(n_trials):
                W_sum += self._single_W_construct(X_v, seed=self.seed + t)
            W_final = W_sum / n_trials
        else:
            W_final = self._single_W_construct(X_v, seed=self.seed)
        
        delta = np.sum(W_final, axis=0)
        W_tilde = W_final / np.sqrt(delta + 1e-10)
        return W_tilde

    def solve(self, X_list, n_clusters):
        X_list = self._preprocess(X_list)
        self.c = n_clusters
        V = len(X_list)
        W_tildes = [self._construct_W_tilde(X_v) for X_v in X_list]
        
        self.alpha = np.ones(V) / V
        M_concat = np.hstack(W_tildes)
        U, _, _ = svds(M_concat, k=self.c)
        self.H_star = U
        self.H_list = [U.copy() for _ in range(V)]
        
        for it in range(self.max_iter):
            for inner_it in range(self.inner_max_iter):
                for v in range(V):
                    Ai = np.hstack([W_tildes[v], np.sqrt(self.beta) * self.H_star])
                    Ui, _, _ = svds(Ai, k=self.c)
                    self.H_list[v] = Ui
                
                Q = np.hstack([np.sqrt(self.alpha[v] * self.beta) * self.H_list[v] for v in range(V)])
                U_star, _, _ = svds(Q, k=self.c)
                self.H_star = U_star

            term_clustering = 0
            term_coreg = 0
            for v in range(V):
                term_clustering += self.alpha[v] * np.linalg.norm(self.H_list[v].T @ W_tildes[v])**2
                term_coreg += self.alpha[v] * self.beta * np.linalg.norm(self.H_list[v].T @ self.H_star)**2
            term_entropy = self.gamma * np.sum(self.alpha * np.log(self.alpha + 1e-12))
            self.obj_history.append(term_clustering + term_coreg + term_entropy)

            raw_grads = np.zeros(V)
            for v in range(V):
                val_mi = np.linalg.norm(self.H_list[v].T @ W_tildes[v])**2
                val_coreg = self.beta * np.linalg.norm(self.H_list[v].T @ self.H_star)**2
                raw_grads[v] = val_mi + val_coreg + self.gamma * (np.log(self.alpha[v] + 1e-12) + 1)

            u = np.argmax(self.alpha)
            g = np.zeros(V)
            for v in range(V):
                if v == u: continue
                reduced_grad_v = raw_grads[v] - raw_grads[u]
                g[v] = -reduced_grad_v if (self.alpha[v] > 0 or reduced_grad_v < 0) else 0
            g[u] = -np.sum(np.delete(g, u))

            self.alpha = self.alpha + self.lr * g
            self.alpha = np.maximum(self.alpha, 0)
            self.alpha /= (np.sum(self.alpha) + 1e-12)
            self.lr *= self.lr_decay
            if it > 1 and abs(self.obj_history[-1] - self.obj_history[-2]) < 1e-6: break

        final_kmeans = KMeans(n_clusters=self.c, random_state=self.seed, n_init=10).fit(self.H_star)
        return final_kmeans.labels_

def main():
    try:
        data = sio.loadmat(config["file_path"])
        X_raw = data['X']
        X_list = [X_raw[0, i].astype(float) for i in range(X_raw.shape[1])]
        Y = data['Y'].flatten().astype(int)
        if np.min(Y) > 0: Y = Y - np.min(Y)
        n_clusters = len(np.unique(Y))
        model = CoReg_SMMSC_Final(config)
        pred_labels = model.solve(X_list, n_clusters)
        res = EvaluationMetrics.get_metrics(Y, pred_labels)
        print("\n=== 实验结果 ===")
        for k, v in res.items(): print(f"{k:<8}: {v:.4f}")
        # === 收敛图像 ===
        plt.figure(figsize=(8, 4))
        plt.plot(model.obj_history, 'b-o', markersize=4)
        plt.title(f"Convergence of {config['anchor_method']} Anchor Method")
        plt.xlabel("Outer Iterations (Alpha Update)")
        plt.ylabel("Objective Value")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()