import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from .State import State
from matplotlib.ticker import ScalarFormatter # 导入 ScalarFormatter


line_styles = [
    (0, (1000, 1)),      # 实线 (非常长的实线段，几乎是连续的)
    (0, (1, 1)),         # 极细的点线 (类似于 :)
    (0, (5, 5, 5)),      # 更细的点线
    (0, (3, 5, 1, 5)),   # 点划线 (类似于 -.)
    (0, (6, 2, 2, 2)),   # 自定义：长虚线后跟小点
    (0, (5, 1)),         # 密集点划线 (短实线后跟短间隔)
    (0, (1, 5)),         # 稀疏点线 (短点后跟长间隔)
    (0, (10, 0)),        # 粗实线 (理论上就是纯实线，因为间隔为0)
    (0, (3, 1, 1, 1)),   # 虚线点线 (短虚线，点，短虚线，点...)
    (0, (2, 2, 5, 2))    # 自定义：短虚线，短间隔，长虚线，短间隔
]

colors = [
    'blue',
    'red',
    'green',
    'purple',
    'orange',
    'cyan',
    'magenta',
    'lime',     # 亮绿色
    'gold',     # 金色
    'teal'      # 青色/蓝绿色
]

def timeCurve(target_state: np.ndarray, times: list, markNodes: list):
    fig, ax = plt.subplots(figsize=(10,8*0.9))
    for i, node_index in enumerate(markNodes):
        ax.plot(times, target_state[:, i],
             label=f'Node {node_index}',
             color=colors[i],
             linestyle=line_styles[i],
             linewidth = 3) # You can adjust markersize as needed
    # ax.set_title(f"Graph N = {N}", fontsize = 22)
    ax.set_xlabel("Time", fontsize = 20)
    ax.set_ylabel("Probability", fontsize = 20)
    ax.tick_params(axis='y', labelsize=15) 
    ax.tick_params(axis='x', labelsize=15)
    ax.legend(fontsize = 20)
    return fig,ax

def probDistribution(stator: State):
    probs = stator.getProbabilities()**2
    N = len(probs)
    fig, ax = plt.subplots(figsize=(10,8*0.9))
    sns.barplot(x=np.arange(len(probs)), y=probs, color='#5275D6',ax = ax)
    ax.set_title(f"Graph N = {N}", fontsize = 22)
    ax.set_xlabel("Nodes", fontsize = 20)
    ax.set_ylabel("Probability",fontsize = 20)

    desired_x_ticks = np.arange(0, len(probs), len(probs)//10) # Indices for the ticks you want to show
    desired_x_labels = [str(i) for i in desired_x_ticks] # Corresponding labels

    ax.set_xticks(desired_x_ticks, desired_x_labels, fontsize = 15) # Set custom ticks and labels
    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    # 启用科学计数法
    formatter.set_scientific(True)
    # 设置科学计数法的指数范围，scilimits=(0,0) 表示所有数字都使用科学计数法
    formatter.set_powerlimits((0, 0)) 
    
    # 将 Y 轴的主要刻度格式化器设置为我们创建的 ScalarFormatter
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='y', labelsize=15) 
    return fig,ax