import os
# 必须放在 import numpy 或 sklearn 之前！
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment

# ==========================================
# 0. 全局路径配置区
# ==========================================
DATA_DIR = './' 
SAVE_DIR = './Heatmap_Results' # 结果将保存在此新文件夹下

# ==========================================
# 1. 5 个数据集的参数配置区
# ==========================================
DATASET_CONFIGS = {
    'mfeat.mat': {
        'coreg': {"n_anchors": 500, "beta": 0.01, "gamma": 10.0, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.01, "lr_decay": 0.95, "seed": 42, "row_norm": None, "use_pca": False, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5},
        'smmsc': {"n_anchors": 500, "beta": 0.0, "gamma": 10.0, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.01, "lr_decay": 0.95, "seed": 42, "row_norm": None, "use_pca": False, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5}
    },
    'UCI_3view.mat': {
        'coreg': {"n_anchors": 100, "beta": 0.01, "gamma": 0.01, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.01, "lr_decay": 0.95, "seed": 42, "row_norm": None, "use_pca": False, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5},
        'smmsc': {"n_anchors": 100, "beta": 0.0, "gamma": 0.01, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.01, "lr_decay": 0.95, "seed": 42, "row_norm": None, "use_pca": False, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5}
    },
    'bbcsport_2view.mat': {
        'coreg': {"n_anchors": 200, "beta": 0.01, "gamma": 0.001, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.01, "lr_decay": 0.95, "seed": 42, "row_norm": 'l2', "use_pca": True, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5},
        'smmsc': {"n_anchors": 200, "beta": 0.0, "gamma": 0.001, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.01, "lr_decay": 0.95, "seed": 42, "row_norm": 'l2', "use_pca": True, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5}
    },
    'NGs.mat': {
        'coreg': {"n_anchors": 500, "beta": 0.1, "gamma": 1.0, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.03, "lr_decay": 0.95, "seed": 42, "row_norm": 'l2', "use_pca": True, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5},
        'smmsc': {"n_anchors": 500, "beta": 0.0, "gamma": 1.0, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.03, "lr_decay": 0.95, "seed": 42, "row_norm": 'l2', "use_pca": True, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5}
    },
    'WebKB.mat': {
        'coreg': {"n_anchors": 100, "beta": 5.0, "gamma": 0.001, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.01, "lr_decay": 0.95, "seed": 42, "row_norm": 'l2', "use_pca": True, "pca_dim": 100, "anchor_method": 'kmeans', "n_trials": 5},
        'smmsc': {"n_anchors": 100, "beta": 0.0, "gamma": 0.001, "max_iter": 50, "inner_max_iter": 10, "n_neighbors": 5, "lr": 0.01, "lr_decay": 0.95, "seed": 42, "row_norm": 'l2', "use_pca": True, "pca_dim": 150, "anchor_method": 'kmeans', "n_trials": 5}
    }
}

# ==========================================
# 2. 评估函数
# ==========================================
class EvaluationMetrics:
    @staticmethod
    def get_metrics(y_true, y_pred):
        y_true = np.array(y_true).astype(int).flatten()
        y_pred = np.array(y_pred).astype(int).flatten()
        acc = metrics.accuracy_score(y_true, y_pred) # 简化的示例，真实应映射最佳匹配
        return {"ACC": acc}

# ==========================================
# 3. 核心模型 - 改进模型 (CoReg_SMMSC_Final)
# ==========================================
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
        if self.cfg.get("row_norm") is None and not self.cfg.get("use_pca"): return X_list
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
        if hasattr(X_v, 'toarray'): X_v = X_v.toarray()
        n = X_v.shape[0]
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
            for j in range(self.n_neighbors): W[i, idx[j]] = (d_kp1 - nearest_dist[j]) / denom
        return W

    def _construct_W_tilde(self, X_v):
        W_final = self._single_W_construct(X_v, seed=self.seed)
        delta = np.sum(W_final, axis=0)
        return W_final / np.sqrt(delta + 1e-10)

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

            term_clustering = term_coreg = 0
            raw_grads = np.zeros(V)
            for v in range(V):
                term_clustering += self.alpha[v] * np.linalg.norm(self.H_list[v].T @ W_tildes[v])**2
                term_coreg += self.alpha[v] * self.beta * np.linalg.norm(self.H_list[v].T @ self.H_star)**2
                raw_grads[v] = np.linalg.norm(self.H_list[v].T @ W_tildes[v])**2 + self.beta * np.linalg.norm(self.H_list[v].T @ self.H_star)**2 + self.gamma * (np.log(self.alpha[v] + 1e-12) + 1)
            self.obj_history.append(term_clustering + term_coreg + self.gamma * np.sum(self.alpha * np.log(self.alpha + 1e-12)))

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

        return self.H_star # 【注意：热力图只需返回连续的 H_star】


# ==========================================
# 4. 基线消融模型 (WSMSC_CRS_Ablation)
# ==========================================
class WSMSC_CRS_Ablation(CoReg_SMMSC_Final):
    def solve(self, X_list, n_clusters):
        X_list = self._preprocess(X_list)
        V = len(X_list)
        W_tildes = [self._construct_W_tilde(X_v) for X_v in X_list]
        self.alpha = np.ones(V) / V
        lr = self.lr
        
        for it in range(self.max_iter):
            M_concat = np.hstack([np.sqrt(self.alpha[v]) * W_tildes[v] for v in range(V)])
            try:
                U_star, _, _ = svds(M_concat, k=n_clusters)
                self.H_star = U_star
            except:
                U_star = np.random.randn(W_tildes[0].shape[0], n_clusters)
                self.H_star, _ = np.linalg.qr(U_star)

            term_clustering = 0
            raw_grads = np.zeros(V)
            for v in range(V):
                term_clustering += self.alpha[v] * np.linalg.norm(self.H_star.T @ W_tildes[v])**2
                raw_grads[v] = np.linalg.norm(self.H_star.T @ W_tildes[v])**2 + self.gamma * (np.log(self.alpha[v] + 1e-12) + 1)
            self.obj_history.append(term_clustering + self.gamma * np.sum(self.alpha * np.log(self.alpha + 1e-12)))

            u = np.argmax(self.alpha)
            g = np.zeros(V)
            for v in range(V):
                if v == u: continue
                reduced_grad_v = raw_grads[v] - raw_grads[u]
                g[v] = -reduced_grad_v if (self.alpha[v] > 0 or reduced_grad_v < 0) else 0
            g[u] = -np.sum(np.delete(g, u))

            self.alpha = self.alpha + lr * g
            self.alpha = np.maximum(self.alpha, 1e-10)
            self.alpha /= np.sum(self.alpha)
            lr *= self.lr_decay
            if it > 1 and abs(self.obj_history[-1] - self.obj_history[-2]) < 1e-6: break

        return self.H_star


# ==========================================
# 5. 主程序：绘制块对角热力图
# ==========================================
def run_heatmap_visualization():
    if not os.path.exists(SAVE_DIR): 
        os.makedirs(SAVE_DIR)

    for dataset_name, configs in DATASET_CONFIGS.items():
        print(f"\n==========================================")
        print(f"正在处理数据集: {dataset_name} 生成热力图...")
        file_path = os.path.join(DATA_DIR, dataset_name)
        
        try:
            data = sio.loadmat(file_path)
            X_raw = data['X']
            X_list = [X_raw[0, i].astype(float) for i in range(X_raw.shape[1])]
            Y = data['Y'].flatten().astype(int)
            if np.min(Y) > 0: Y = Y - np.min(Y)
            n_clusters = len(np.unique(Y))
        except Exception as e:
            print(f"[跳过] 找不到文件或加载失败 {file_path}，报错信息: {e}")
            continue

        # 1. 提取 H_star
        print(" -> 正在运行 SMMSC (基线模型)...")
        model_smmsc = WSMSC_CRS_Ablation(configs['smmsc'])
        H_smmsc = model_smmsc.solve(X_list, n_clusters)

        print(" -> 正在运行 CoReg-SMMSC (改进模型)...")
        model_coreg = CoReg_SMMSC_Final(configs['coreg'])
        H_coreg = model_coreg.solve(X_list, n_clusters)

        # 2. 对谱嵌入进行 L2 行归一化，计算余弦相似度矩阵
        H_smmsc_norm = preprocessing.normalize(H_smmsc, norm='l2', axis=1)
        H_coreg_norm = preprocessing.normalize(H_coreg, norm='l2', axis=1)

        S_smmsc = H_smmsc_norm @ H_smmsc_norm.T
        S_coreg = H_coreg_norm @ H_coreg_norm.T

        # 3. 按真实标签 Y 对样本进行重新排序，使得同类样本在矩阵中物理相连
        sort_idx = np.argsort(Y)
        S_smmsc_sorted = S_smmsc[sort_idx, :][:, sort_idx]
        S_coreg_sorted = S_coreg[sort_idx, :][:, sort_idx]

        # 计算各类别的断点，用于绘制辅助线
        _, counts = np.unique(Y[sort_idx], return_counts=True)
        boundaries = np.cumsum(counts)[:-1]

        # 4. 绘制对比热力图
        print(" -> 正在生成块对角化图像...")
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        
        # 选用经典的学术热力图配色，统一取值范围映射 [0, 1]
        cmap_style = 'viridis' 

        # 子图 1: 基线
        im1 = axes[0].imshow(np.clip(S_smmsc_sorted, 0, 1), cmap=cmap_style, aspect='auto', interpolation='none')
        axes[0].set_title(f"SMMSC (Baseline)", fontsize=15, pad=12)
        axes[0].axis('off')
        
        # 子图 2: 改进模型
        im2 = axes[1].imshow(np.clip(S_coreg_sorted, 0, 1), cmap=cmap_style, aspect='auto', interpolation='none')
        axes[1].set_title(f"CoReg-SMMSC (Proposed)", fontsize=15, pad=12)
        axes[1].axis('off')

        # 核心亮点：绘制真实标签边界（白色虚线网格）
        for ax in axes:
            for b in boundaries:
                ax.axhline(b - 0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
                ax.axvline(b - 0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.7)

        # 共享 Colorbar
        cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8, pad=0.03)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=12)
        
        dataset_title = os.path.splitext(dataset_name)[0]
        # plt.suptitle(f"Block-diagonal Heatmap of Consensus Representation on {dataset_title}", fontsize=18, y=1.05)
        
        save_path = os.path.join(SAVE_DIR, f'heatmap_{dataset_title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" [完成] 热力图已保存至: {save_path}")

if __name__ == "__main__":
    run_heatmap_visualization()