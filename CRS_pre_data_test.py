import os
# 必须放在 import numpy 或 sklearn 之前！
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy.io as sio
import time
import pandas as pd
import matplotlib.pyplot as plt
from CRS_pre import CoReg_SMMSC_Final, EvaluationMetrics

def run_multi_dataset_test():
    # 1. 参数配置
    dataset_files = ['NGs.mat'] # 可在此添加更多数据集文件名
    
    # 定义网格搜索空间
    anchors = [50, 100, 200, 500]          
    betas = [0.01, 0.1, 1.0, 5.0, 10.0]     
    gammas = [0.001, 0.01, 0.1, 1.0, 10.0]

    # 输出主文件夹
    output_main_dir = 'test5'
    if not os.path.exists(output_main_dir):
        os.makedirs(output_main_dir)

    # 2. 循环遍历数据集
    for file_path in dataset_files:
        dataset_name = os.path.splitext(file_path)[0]
        print(f"\n{'#'*30}")
        print(f"开始处理数据集: {dataset_name}")
        print(f"{'#'*30}")

        dataset_output_dir = os.path.join(output_main_dir, dataset_name)
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)

        # 加载数据
        try:
            data = sio.loadmat(file_path)
            X_raw = data['X']
            X_list = [X_raw[0, i].astype(float) for i in range(X_raw.shape[1])]
            Y = data['Y'].flatten()
            if np.min(Y) > 0: Y = Y - np.min(Y)
            n_clusters = len(np.unique(Y))
        except Exception as e:
            print(f"数据集 {file_path} 加载失败: {e}")
            continue

        results_list = []
        best_acc = 0
        best_info = {}

        # 3. 网格搜索
        for p in anchors:
            for b in betas:
                for g in gammas:
                    current_cfg = {
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
                        "row_norm": 'l2',            
                        "use_pca": True,            
                        "pca_dim": 100,              
                        "anchor_method": 'kmeans',   
                        "n_trials": 5
                    }
                    
                    try:
                        model = CoReg_SMMSC_Final(current_cfg)
                        pred_labels = model.solve(X_list, n_clusters)
                        res = EvaluationMetrics.get_metrics(Y, pred_labels)
                        
                        # 记录结果
                        row = {
                            'p': p, 'Beta': b, 'Gamma': g,
                            'ACC': res['ACC'], 'NMI': res['NMI'], 
                            'ARI': res['ARI'], 'Purity': res['Purity'], 'F1': res['F1']
                        }
                        results_list.append(row)
                        print(f"[P={p}, B={b}, G={g}] ACC: {res['ACC']:.4f}")

                        # 更新最优
                        if res['ACC'] > best_acc:
                            best_acc = res['ACC']
                            best_info = {
                                'p': p, 'beta': b, 'gamma': g,
                                'metrics': res,
                                'obj_history': model.obj_history
                            }
                    except Exception as e:
                        print(f"参数 p={p}, b={b}, g={g} 报错: {e}")

        # 4. 保存当前数据集的结果到 Excel
        df = pd.DataFrame(results_list)
        excel_name = f"{dataset_name}_GridSearch_Results.xlsx"
        df.to_excel(os.path.join(dataset_output_dir, excel_name), index=False)
        
        # 5. 保存最优参数的收敛图像
        if best_info:
            plt.figure(figsize=(10, 5))
            plt.plot(best_info['obj_history'], 'r-o', markersize=4)
            plt.title(f"Convergence: {dataset_name} (Best p={best_info['p']}, b={best_info['beta']}, g={best_info['gamma']})")
            plt.xlabel("Outer Iterations")
            plt.ylabel("Objective Value")
            plt.grid(True)
            
            plot_name = f"{dataset_name}_Best_Convergence.png"
            plt.savefig(os.path.join(dataset_output_dir, plot_name))
            plt.close() 

            # 打印数据集总结
            summary_txt = os.path.join(dataset_output_dir, f"{dataset_name}_summary.txt")
            with open(summary_txt, 'w') as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Best Params: p={best_info['p']}, beta={best_info['beta']}, gamma={best_info['gamma']}\n")
                f.write(f"Best Metrics: {best_info['metrics']}\n")

    print(f"\n所有实验运行完毕！结果已保存在 '{output_main_dir}' 文件夹中。")

if __name__ == "__main__":
    run_multi_dataset_test()