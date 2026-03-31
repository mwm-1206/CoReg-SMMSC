# CoReg-SMMSC

基于极小极大化框架的共正则化多视图谱聚类 (Co-Regularized Min-Max Multi-View Spectral Clustering) 的代码实现。

## ⚠️ 运行前必读：路径配置
所有脚本内的数据读取路径均默认为**当前目录**（如 `DATA_DIR = './'`）。
**在运行代码前，请自行处理数据路径**：可以手动修改代码里的路径指向 `data/` 文件夹，或者直接把要跑的 `.mat` 数据集复制到脚本所在的同级目录。

## 📁 目录结构
* `data/`：文中所用多视图数据集（`.mat` 格式）
* `main/`：核心算法与参数搜索代码
  * `CRS_pre.py`：主算法代码与单次运行测试
  * `CRS_pre_test.py`：单数据集参数网格搜索
  * `CRS_pre_data_test.py`：多数据集批量测试与结果导出
* `pic/`：实验结果可视化代码
  * `allheat/`：其他对比算法的结果文件（用于生成对比图）
  * `convergence.py`：绘制收敛曲线
  * `heatmap.py` / `plot_1v1_heatmaps.py`：绘制余弦相似度热力图
  * `yepao.py`：绘制多指标对比气泡图
  * `pic.m`：MATLAB 脚本，用于绘制 3D 柱状图

## 🛠️ 环境依赖
* Python: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`
* MATLAB: 仅用于运行 `pic.m`

## 🚀 快速开始
1. **数据就位**：确保需要用到的 `.mat` 文件与执行脚本的路径对应。
2. **跑通算法**：进入 `main/` 目录，执行 `python CRS_pre.py` 验证主模型。
3. **生成图表**：进入 `pic/` 目录，执行对应的 Python 脚本或运行 MATLAB 脚本即可生成论文配图。