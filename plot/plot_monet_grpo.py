#!/usr/bin/env python3
"""
绘制 MonetVLM GRPO 实验的 TensorBoard 日志图表
注意：GRPO 训练可能还在进行中，此脚本会处理不完整的日志
"""
import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_monet_grpo():
    """绘制 MonetVLM GRPO 实验的训练曲线"""
    
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'save', 'monet_grpo', 'runs')
    
    if not os.path.exists(log_dir):
        print(f"警告: 日志目录不存在: {log_dir}")
        print("提示: GRPO 训练可能还未开始或日志目录尚未创建")
        return
    
    # 检查是否有事件文件
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"警告: 日志目录中没有找到事件文件: {log_dir}")
        print("提示: GRPO 训练可能还在进行中，请稍后再运行此脚本")
        return
    
    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
    except Exception as e:
        print(f"警告: 加载 TensorBoard 日志时出错: {e}")
        print("提示: GRPO 训练可能还在写入日志，请稍后再试")
        return
    
    tags = event_acc.Tags()['scalars']
    print(f"可用的 tags: {tags}")
    
    if not tags:
        print("警告: 没有找到任何 scalar tags，日志可能还在写入中")
        return
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制 loss 相关曲线
    loss_tags = [tag for tag in tags if 'loss' in tag.lower() or 'reward' in tag.lower()]
    if loss_tags:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for loss_tag in loss_tags:
            steps = []
            values = []
            for event in event_acc.Scalars(loss_tag):
                steps.append(event.step)
                values.append(event.value)
            
            ax.plot(steps, values, linewidth=2, label=loss_tag)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('MonetVLM GRPO - Loss & Reward', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, 'monet_grpo_loss.png')
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
        ax.set_title('MonetVLM GRPO - Learning Rate', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, 'monet_grpo_lr.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()
    
    # 绘制 KL divergence（如果有）
    kl_tags = [tag for tag in tags if 'kl' in tag.lower()]
    if kl_tags:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for kl_tag in kl_tags:
            steps = []
            values = []
            for event in event_acc.Scalars(kl_tag):
                steps.append(event.step)
                values.append(event.value)
            
            ax.plot(steps, values, linewidth=2, label=kl_tag)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title('MonetVLM GRPO - KL Divergence', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, 'monet_grpo_kl.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()
    
    print("MonetVLM GRPO 实验绘图完成!")

if __name__ == '__main__':
    plot_monet_grpo()
