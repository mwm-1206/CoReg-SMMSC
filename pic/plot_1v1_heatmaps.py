import os
# 必须放在 import numpy 或 sklearn 之前！
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib as mpl
import sys

# =======================================================
# 核心精髓：直接从你的 heatmap.py 中导入模型和配置参数！
# 这样完全不需要修改你的 heatmap.py 原文件
# =======================================================
try:
    from heatmap import CoReg_SMMSC_Final, DATASET_CONFIGS
except ImportError:
    print("【错误】无法导入 heatmap.py。请确保本脚本和 heatmap.py 放在同一个文件夹下！")
    sys.exit(1)

# ==========================================
# 0. 全局路径与显示配置区
# ==========================================
BASE_RES_DIR = './allheat'       # 存放 MATLAB 基线结果的文件夹
SAVE_DIR = './Heatmap_1v1_Plots' # 生成的 1x2 对比图保存文件夹
DATA_DIR = './'                  # 你的原始数据集文件夹 (与 heatmap.py 默认保持一致)

BASELINES_PREFIX = ['OMSC', 'MERA_MSC', 'UDBGL']

# 显示的精美名称
METHOD_DISPLAY_NAMES = {
    'OMSC': 'OMSC',
    'MERA_MSC': 'MERA-MSC',
    'UDBGL': 'UDBGL'
}

# 绘图字体配置 (复刻学术绘图风格)
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['font.family'] = 'sans-serif'

def process_and_sort_S(H_matrix, sort_idx, N_target):
    """ 处理连续表示矩阵，计算余弦相似度并根据真实标签排序 """
    
    # 【色彩还原修改 1】：强行取实部，消除微小的虚数残余，并将 NaN 转为 0
    H_matrix = np.nan_to_num(H_matrix.real)

    # 解决 MERA 等算法可能的样本截断/补齐问题
    if H_matrix.shape[0] > N_target:
        H_matrix = H_matrix[:N_target, :]
    elif H_matrix.shape[0] < N_target:
        padding = np.random.rand(N_target - H_matrix.shape[0], H_matrix.shape[1]) * 1e-4
        H_matrix = np.vstack((H_matrix, padding))

    # L2 行归一化
    H_norm = preprocessing.normalize(H_matrix, norm='l2', axis=1)
    
    # 计算 Cosine 相似度矩阵 S = H H^T
    S = H_norm @ H_norm.T
    np.clip(S, 0, 1, out=S) # 约束在 [0,1]
    
    # 按标签的排序索引对 S 的行和列重排，形成块对角
    S_sorted = S[sort_idx, :][:, sort_idx]
    return S_sorted

def generate_1v1_plots():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"\n==========================================")
    print(f"开始生成 1v1 (1x2) 对比热力图...")
    print(f"==========================================")
    
    # 遍历你在 heatmap.py 中配置的所有数据集
    for dataset_filename, configs in DATASET_CONFIGS.items():
        dataset_title = os.path.splitext(dataset_filename)[0]
        print(f"\n>>> 正在处理数据集: {dataset_title}")
        
        # --- 1. 读取原数据和真实标签 Y ---
        file_path = os.path.join(DATA_DIR, dataset_filename)
        try:
            data = sio.loadmat(file_path)
            X_raw = data['X']
            X_list = [X_raw[0, i].astype(float) for i in range(X_raw.shape[1])]
            
            # 兼容各种不同的标签命名
            if 'Y' in data: Y = data['Y']
            elif 'y' in data: Y = data['y']
            elif 'gnd' in data: Y = data['gnd']
            elif 'labels' in data: Y = data['labels']
            else: raise ValueError("Label Y not found")
            
            Y = Y.flatten().astype(int)
            if np.min(Y) > 0: Y = Y - np.min(Y)
            n_clusters = len(np.unique(Y))
            num_samples = len(Y)
        except Exception as e:
            print(f"  [跳过] 文件加载失败 {file_path}: {e}")
            continue

        sort_idx = np.argsort(Y)
        _, counts = np.unique(Y[sort_idx], return_counts=True)
        boundaries = np.cumsum(counts)[:-1]

        # --- 2. 运行 CoReg-SMMSC ---
        print("  -> 正在运行 CoReg-SMMSC (Proposed)...")
        try:
            model_coreg = CoReg_SMMSC_Final(configs['coreg'])
            H_p = model_coreg.solve(X_list, n_clusters)
            S_coreg_sorted = process_and_sort_S(H_p, sort_idx, num_samples)
        except Exception as e:
            print(f"  [报错] CoReg-SMMSC 运行失败，跳过该数据集: {e}")
            continue

        # --- 3. 智能匹配数据集的预处理变体 ---
        variants_to_try = [dataset_title]
        dt_lower = dataset_title.lower()
        if 'bbcsport' in dt_lower:
            variants_to_try = ['bbcsport_2view', 'bbcsport_processed']
        elif 'ngs' in dt_lower:
            variants_to_try = ['NGs', 'NGs_processed']
        elif 'webkb' in dt_lower:
            variants_to_try = ['WebKB', 'WebKB_processed']

        # --- 4. 遍历并绘制 1v1 对比图 ---
        for prefix in BASELINES_PREFIX:
            baseline_path = None
            # 寻找该基线最优的数据集变体结果
            for var in variants_to_try:
                path = os.path.join(BASE_RES_DIR, f'{prefix}_{var}_H.mat')
                if os.path.exists(path):
                    baseline_path = path
                    break
            
            if not baseline_path:
                print(f"  -> [缺失] 找不到 {prefix} 在 {dataset_title} 上的结果，已自动跳过该基线。")
                continue
                
            # 加载并处理 MATLAB 基线矩阵
            try:
                H_mat = sio.loadmat(baseline_path)
                if 'H_star' in H_mat: H_b = H_mat['H_star']
                elif 'Z' in H_mat: H_b = H_mat['Z']
                elif 'U' in H_mat: H_b = H_mat['U']
                else: raise ValueError("找不到连续矩阵变量")
                
                # OMSC 容错：如果是 K x N 则转置为 N x K
                if H_b.shape[0] != num_samples and H_b.shape[1] == num_samples:
                    H_b = H_b.T
                    
                S_baseline_sorted = process_and_sort_S(H_b, sort_idx, num_samples)
            except Exception as e:
                print(f"  -> [报错] 读取 {baseline_path} 失败: {e}")
                continue

            # ================= 5. 绘图 (完全复刻原生格式) =================
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            cmap_style = 'viridis'

            baseline_display_name = METHOD_DISPLAY_NAMES.get(prefix, prefix.replace('_', '-'))

            # 子图 1: 基线
            # 【色彩还原修改 2】：使用 'nearest' 插值，避免矩阵过密时抗锯齿带来的像素混合模糊
            im1 = axes[0].imshow(S_baseline_sorted, cmap=cmap_style, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
            axes[0].set_title(f"{baseline_display_name} (Baseline)", fontsize=15, pad=12)
            axes[0].axis('off')
            
            # 子图 2: 改进模型 (Proposed)
            im2 = axes[1].imshow(S_coreg_sorted, cmap=cmap_style, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
            axes[1].set_title(f"CoReg-SMMSC (Proposed)", fontsize=15, pad=12) 
            axes[1].axis('off')

            # 绘制白色虚线网格
            for ax in axes:
                for b in boundaries:
                    ax.axhline(b - 0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.7)
                    ax.axvline(b - 0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.7)

            # 共享 Colorbar
            cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8, pad=0.03)
            cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=14)
            
            # 【色彩还原修改 3】：后缀从 .jpg 变更为 .png，避开 JPEG 有损压缩引起的偏色
            save_file_name = f'heatmap_{prefix}_vs_Proposed_{dataset_title}.png'
            save_path = os.path.join(SAVE_DIR, save_file_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> [出图成功] {save_file_name}")

if __name__ == "__main__":
    generate_1v1_plots()