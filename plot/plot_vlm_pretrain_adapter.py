#!/usr/bin/env python3
"""
绘制 VLM Adapter 预训练实验的 TensorBoard 日志图表
"""
import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 配置中文字体支持（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_vlm_pretrain_adapter():
    """绘制 VLM Adapter 预训练实验的训练曲线"""
    
    # TensorBoard 日志路径
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'save', 'vlm_pretrain_adapter', 'runs')
    
    if not os.path.exists(log_dir):
        print(f"警告: 日志目录不存在: {log_dir}")
        return
    
    # 加载 TensorBoard 事件文件
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # 获取所有可用的 scalar tags
    tags = event_acc.Tags()['scalars']
    print(f"可用的 tags: {tags}")
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制 loss 曲线
    if 'train/loss' in tags:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = []
        values = []
        for event in event_acc.Scalars('train/loss'):
            steps.append(event.step)
            values.append(event.value)
        
        ax.plot(steps, values, 'b-', linewidth=2, label='Train Loss')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('VLM Adapter Pretrain - Training Loss', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, 'vlm_pretrain_adapter_loss.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()
    
    # 绘制 learning rate 曲线
    lr_tags = [tag for tag in tags if 'lr' in tag.lower() or 'learning_rate' in tag.lower()]
    if lr_tags:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for lr_tag in lr_tags:
            steps = []
            values = []
            for event in event_acc.Scalars(lr_tag):
                steps.append(event.step)
                values.append(event.value)
            
            ax.plot(steps, values, linewidth=2, label=lr_tag)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('VLM Adapter Pretrain - Learning Rate', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, 'vlm_pretrain_adapter_lr.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()
    
    print("VLM Adapter 预训练实验绘图完成!")

if __name__ == '__main__':
    plot_vlm_pretrain_adapter()
