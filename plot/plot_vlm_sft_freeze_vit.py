#!/usr/bin/env python3
"""
绘制 VLM SFT 冻结 ViT 实验的 TensorBoard 日志图表
"""
import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_vlm_sft_freeze_vit():
    """绘制 VLM SFT 冻结 ViT 实验的训练曲线"""
    
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'save', 'vlm_sft_freeze_vit', 'runs')
    
    if not os.path.exists(log_dir):
        print(f"警告: 日志目录不存在: {log_dir}")
        return
    
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    tags = event_acc.Tags()['scalars']
    print(f"可用的 tags: {tags}")
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制训练和验证 loss
    loss_tags = ['train/loss', 'eval/loss']
    available_loss_tags = [tag for tag in loss_tags if tag in tags]
    
    if available_loss_tags:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'train/loss': 'b', 'eval/loss': 'r'}
        for loss_tag in available_loss_tags:
            steps = []
            values = []
            for event in event_acc.Scalars(loss_tag):
                steps.append(event.step)
                values.append(event.value)
            
            label = 'Train Loss' if 'train' in loss_tag else 'Eval Loss'
            ax.plot(steps, values, color=colors.get(loss_tag, 'g'), linewidth=2, label=label)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('VLM SFT (Freeze ViT) - Training & Evaluation Loss', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, 'vlm_sft_freeze_vit_loss.png')
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
        ax.set_title('VLM SFT (Freeze ViT) - Learning Rate', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, 'vlm_sft_freeze_vit_lr.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_path}")
        plt.close()
    
    print("VLM SFT 冻结 ViT 实验绘图完成!")

if __name__ == '__main__':
    plot_vlm_sft_freeze_vit()
