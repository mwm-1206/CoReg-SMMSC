import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备
datasets = ['mfeat', 'UCI_3view', 'BBCSport', 'NGs', 'WebKB']
methods = ['UDBGL', 'MERA-MSC', 'OMSC', 'EGTFC', 'SMMSC', 'CoReg-SMMSC']
metrics = ['ACC', 'NMI', 'ARI', 'F-score']

data = {
    'ACC': [
        [0.7620, 0.9260, 0.8165, 0.8525, 0.9225, 0.9740], 
        [0.6230, 0.9715, 0.8265, 0.8515, 0.8090, 0.9620], 
        [0.5772, 0.7985, 0.7684, 0.9522, 0.9724, 0.9779], 
        [0.4040, 0.7256, 0.7380, 0.9600, 0.7640, 0.9440], 
        [0.9172, np.nan, 0.9439, 0.8525, 0.9762, 0.9829]  
    ],
    'NMI': [
        [0.7261, 0.9714, 0.7871, 0.8812, 0.8679, 0.9423],
        [0.5880, 0.9857, 0.7893, 0.8507, 0.8166, 0.9185],
        [0.2269, 0.6975, 0.6108, 0.8688, 0.9143, 0.9323],
        [0.1360, 0.6849, 0.5923, 0.8790, 0.7724, 0.8469],
        [0.5184, np.nan, 0.6785, 0.2048, 0.7947, 0.8423]
    ],
    'ARI': [
        [0.6587, 0.9250, 0.7189, 0.8247, 0.8359, 0.9432],
        [0.4683, 0.9693, 0.7373, 0.7851, 0.7397, 0.9176],
        [0.2716, 0.6757, 0.5795, 0.8915, 0.9209, 0.9373],
        [0.1112, 0.5728, 0.5661, 0.9025, 0.6761, 0.8645],
        [0.6757, np.nan, 0.7729, 0.3781, 0.8967, 0.9248]
    ],
    'F-score': [
        [0.6939, 0.9329, 0.7487, 0.8434, 0.9234, 0.9740],
        [0.5220, 0.9725, 0.7644, 0.8069, 0.7847, 0.9622],
        [0.4756, 0.7647, 0.6776, 0.9178, 0.9773, 0.9809],
        [0.3070, 0.6651, 0.6568, 0.9218, 0.7066, 0.9444],
        [0.8790, np.nan, 0.9160, 0.8276, 0.9652, 0.9747]
    ]
}

# 全局字体设置
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix"
})

# 2. 创建 2x2 画布
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

# 选用四个和谐且易于区分的高级单色渐变
cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    values = np.array(data[metric])
    
    v_min, v_max = np.nanmin(values), np.nanmax(values)
    
    x, y = np.meshgrid(np.arange(len(methods)), np.arange(len(datasets)))
    x = x.flatten()
    y = y.flatten()
    c = values.flatten()
    
    # 过滤 NaN (针对 MERA-MSC 在 WebKB 上无结果的情况)
    mask = ~np.isnan(c)
    x_val = x[mask]
    y_val = y[mask]
    c_val = c[mask]
    
    # 计算归一化大小 (0到1)
    norm_c = (c_val - v_min) / (v_max - v_min + 1e-8)
    
    # 调整圆圈大小，保证最小值气泡也足够写下两个数字
    sizes = 800 + norm_c * 1400 
    
    # 绘制气泡，去掉边缘黑线 (edgecolors='none')
    sc = ax.scatter(x_val, y_val, s=sizes, c=c_val, cmap=cmaps[idx], 
                    vmin=v_min, vmax=v_max, alpha=0.85, edgecolors='none', zorder=2)
    
    # 【核心修改】在圆圈中间写入保留两位的数值
    for i in range(len(x_val)):
        # 智能文字颜色：底色深用白色，底色浅用黑色
        text_color = 'white' if norm_c[i] > 0.55 else 'black'
        
        # 写入文字，格式化为保留两位小数 (如 0.97)
        ax.text(x_val[i], y_val[i], f"{c_val[i]:.4f}", 
                ha='center', va='center', color=text_color, 
                fontsize=11, fontweight='bold', zorder=3)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=12)
    
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=13)
    
    # 反转Y轴，让 mfeat 在最上面，符合常规表格阅读习惯
    ax.invert_yaxis()
    
    ax.set_title(metric, fontsize=18, pad=15, fontweight='bold')
    
    # 突出 Proposed 模型
    ax.get_xticklabels()[-1].set_color('#c00000')
    ax.get_xticklabels()[-1].set_fontweight('bold')
    
    # 背景网格与极简边框
    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=1, alpha=0.3, zorder=0)
    ax.tick_params(which='minor', length=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bbbbbb')
    ax.spines['bottom'].set_color('#bbbbbb')

# 调整子图间距，避免文字重叠
plt.tight_layout()
plt.subplots_adjust(hspace=0.25, wspace=0.15)
plt.savefig('elegant_bubble_chart_2x2.png', dpi=300, bbox_inches='tight')
plt.show()