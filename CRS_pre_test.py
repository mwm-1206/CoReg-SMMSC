import os
# 必须放在 import numpy 或 sklearn 之前！
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.io as sio
from CRS_pre import CoReg_SMMSC_Final, EvaluationMetrics  
import time

def grid_search():
    # 配置文件路径，可根据需要修改
    file_path = 'mfeat.mat' 
    try:
        data = sio.loadmat(file_path)
        X_raw = data['X']
        X_list = [X_raw[0, i].astype(float) for i in range(X_raw.shape[1])]
        Y = data['Y'].flatten()
        if np.min(Y) > 0: Y = Y - np.min(Y)
        n_clusters = len(np.unique(Y))
        print(f"数据加载成功: {file_path}, 包含 {len(X_list)} 个视图, {len(Y)} 个样本, {n_clusters} 个类别")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 网格搜索范围
    anchors = [ 100, 200, 500]          
    betas = [0.01, 0.1, 1, 5, 10]     
    gammas = [0.001, 0.01, 0.1, 1, 10]    
    
    best_acc = 0
    best_params = {}

    header = f"{'p':>4} | {'Beta':>6} | {'Gamma':>6} | {'ACC':>6} | {'NMI':>6} | {'ARI':>6} | {'Pur':>6} | {'F1':>6}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for p in anchors:
        for b in betas:
            for g in gammas:
                current_cfg = {
                    "file_path": file_path, 
                    "n_anchors": p,
                    "beta": b,           
                    "gamma": g, 
                    "max_iter": 50,      
                    "inner_max_iter": 10,
                    "n_neighbors": 5,
                    "lr": 0.01,
                    "lr_decay": 0.95,
                    "seed": 42,
                    # --- CRS_pre 新增预处理配置 ---
                    "row_norm": None,            
                    "use_pca": False,            
                    "pca_dim": 100,              
                    "anchor_method": 'kmeans',   
                    "n_trials": 5                
                }
                
                try:
                    model = CoReg_SMMSC_Final(current_cfg)
                    
                    start_time = time.time()
                    pred_labels = model.solve(X_list, n_clusters)
                    elapsed = time.time() - start_time
                    
                    res = EvaluationMetrics.get_metrics(Y, pred_labels)
                    
                    print(f"{p:4d} | {b:6.3f} | {g:6.3f} | {res['ACC']:.3f} | {res['NMI']:.3f} | {res['ARI']:.3f} | "
                          f"{res['Purity']:.3f} | {res['F1']:.3f}")
                    
                    if res['ACC'] > best_acc:
                        best_acc = res['ACC']
                        best_params = {
                            "p": p, 
                            "beta": b, 
                            "gamma": g, 
                            "metrics": res,
                            "obj_history": model.obj_history 
                        }
                        
                except Exception as e:
                    print(f"参数 p={p}, b={b}, g={g} 运行报错: {e}")

    print("-" * len(header))
    print("网格搜索结束！")
    if best_params:
        print(f"\n最佳参数组合: p={best_params['p']}, beta={best_params['beta']}, gamma={best_params['gamma']}")
        print("对应完整指标:")
        metrics_order = ['ACC', 'NMI', 'ARI', 'Purity', 'F1']
        for k in metrics_order:
            if k in best_params['metrics']:
                print(f"  - {k:8}: {best_params['metrics'][k]:.4f}")
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(best_params['obj_history'], 'r-o', markersize=4)
        plt.title(f"Convergence of Best Result (p={best_params['p']}, b={best_params['beta']}, g={best_params['gamma']})")
        plt.xlabel("Outer Iterations")
        plt.ylabel("Objective Value")
        plt.grid(True)
        plt.show()
    else:
        print("未找到有效结果。")
    print("=" * len(header))

if __name__ == "__main__":
    grid_search()