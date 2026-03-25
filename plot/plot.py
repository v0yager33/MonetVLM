import matplotlib.pyplot as plt
import seaborn as sns
import os

# =================配置区域=================

# 1. 定义优雅的莫兰迪色系 (从高分到低分)
# 颜色逻辑：最高分用温暖的陶土红突出，其余依次过渡到冷静的灰蓝色和浅灰色
elegant_colors = [
    "#C05C50",  # 陶土红 (最高分，醒目且稳重)
    "#5D7A8C",  # 雾霾蓝
    "#8DA399",  # 鼠尾草绿
    "#B0B8B5",  # 暖灰
    "#D1D6D5"   # 浅灰 (最低分，视觉后退)
]

# 2. 数据准备 (顺序已反转：从高到低)
models = [
    "MonetVLM (GRPO)",
    "MonetVLM (SFT)",
    "SparkVLM (Full SFT)",
    "SparkVLM (ViT Freezed SFT)",
    "SparkVLM (Adapter)"
]
scores = [34.5, 16.7, 1.0, 1.2, 0.8]

# =================绘图区域=================

# 设置全局字体和风格
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] # 优先使用无衬线字体
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", context="talk")

# 创建画布
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制柱状图
# width=0.6 让柱子稍微细一点，显得更精致
bars = ax.bar(models, scores, color=elegant_colors, edgecolor='none', width=0.65)

# 设置标题和标签
# 标题去掉粗体，改用常规加细体，更显优雅
ax.set_title("Performance on Wikiart-MonetVLM-200", 
             fontsize=16, fontweight='normal', color='#333333', pad=25, loc='left')

ax.set_ylabel("Score", fontsize=12, color='#555555', labelpad=10)
ax.set_xlabel("", fontsize=12, color='#555555') # 隐藏横坐标大标题，让图表更简洁

# 设置纵坐标范围 (顶部留出 15% 空间给数字)
max_score = max(scores)
ax.set_ylim(0, max_score * 1.15)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2., 
        height + (max_score * 0.01), # 动态调整距离
        f'{height}', 
        ha='center', 
        va='bottom', 
        fontsize=11, 
        weight='bold',
        color='#444444' # 深灰色字体，比纯黑柔和
    )

# 美化网格和边框
# 只保留横向网格，颜色极淡
ax.grid(axis='y', linestyle='-', linewidth=0.8, color='#EEEEEE', zorder=0)
ax.grid(axis='x', visible=False)

# 移除顶部和右侧边框 (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#DDDDDD')
ax.spines['bottom'].set_color('#DDDDDD')

# 调整刻度标签样式
ax.tick_params(axis='both', which='major', labelsize=11, color='#666666', length=4)
# 横坐标标签不旋转，因为去掉了换行符，且顺序是从高到低，阅读顺畅
plt.xticks(ha='center')

# 自动布局调整
plt.tight_layout()

# 保存文件
output_filename = "performance.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 优雅风格图表已生成: {os.path.abspath(output_filename)}")

# 显示
try:
    plt.show()
except:
    pass