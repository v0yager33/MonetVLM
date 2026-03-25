"""
SparkVLM 自定义 SFT Trainer - 纯 PyTorch 实现
支持：BF16 混合精度、梯度累积、Cosine 学习率调度、Checkpoint 保存、分组学习率
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class SFTTrainer:
    """纯 PyTorch 实现的 SFT 训练器，专为 SparkVLM 多模态模型设计。"""

    def __init__(
        self,
        model,
        train_dataset,
        data_collator,
        output_dir="save/vlm_sft",
        learning_rate=2e-5,
        vit_lr=None,
        adapter_lr=None,
        llm_lr=None,
        num_epochs=3,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        dataloader_num_workers=2,
        eval_datasets=None,
        eval_steps=1000,
        eval_on_start=True,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.vit_lr = vit_lr
        self.adapter_lr = adapter_lr
        self.llm_lr = llm_lr
        self.num_epochs = num_epochs
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.bf16 = bf16
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        self.dataloader_num_workers = dataloader_num_workers
        self.eval_datasets = eval_datasets or {}
        self.eval_steps = eval_steps
        self.eval_on_start = eval_on_start

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)

        # TensorBoard 日志
        tb_log_dir = os.path.join(output_dir, "runs")
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"  TensorBoard logs: {tb_log_dir}")

    def _classify_param(self, param_name):
        """根据参数名判断所属模块：vit / adapter / llm。"""
        if param_name.startswith("vision_model."):
            return "vit"
        elif param_name.startswith("linear1.") or param_name.startswith("linear2."):
            return "adapter"
        else:
            return "llm"

    def _build_optimizer(self):
        """构建 AdamW 优化器，支持 ViT / Adapter / LLM 分组学习率。"""
        no_decay_keywords = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        # 确定各模块的学习率，未指定则回退到统一的 learning_rate
        module_lr = {
            "vit": self.vit_lr if self.vit_lr is not None else self.learning_rate,
            "adapter": self.adapter_lr if self.adapter_lr is not None else self.learning_rate,
            "llm": self.llm_lr if self.llm_lr is not None else self.learning_rate,
        }

        # 按 (模块, 是否 no_decay) 分为 6 组
        groups = {}
        for module_name in ["vit", "adapter", "llm"]:
            for use_decay in [True, False]:
                key = (module_name, use_decay)
                groups[key] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            module = self._classify_param(name)
            has_decay = not any(nd in name for nd in no_decay_keywords)
            groups[(module, has_decay)].append(param)

        param_groups = []
        for (module_name, use_decay), params in groups.items():
            if not params:
                continue
            param_groups.append({
                "params": params,
                "lr": module_lr[module_name],
                "weight_decay": self.weight_decay if use_decay else 0.0,
            })

        # 打印分组学习率信息
        use_grouped_lr = (self.vit_lr is not None or self.adapter_lr is not None or self.llm_lr is not None)
        if use_grouped_lr:
            print(f"  Grouped LR: ViT={module_lr['vit']:.1e}, Adapter={module_lr['adapter']:.1e}, LLM={module_lr['llm']:.1e}")
        else:
            print(f"  Uniform LR: {self.learning_rate:.1e}")

        return AdamW(param_groups, lr=self.learning_rate)

    def _build_scheduler(self, optimizer, total_steps):
        """Cosine Annealing with Linear Warmup 学习率调度器。"""
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _prepare_batch(self, batch):
        """将 batch 中的 tensor 移动到目标设备。"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _forward_step(self, batch):
        """执行一次前向传播，返回 loss。"""
        forward_kwargs = {
            "input_ids": batch["input_ids"],
            "labels": batch["labels"],
            "attention_mask": batch.get("attention_mask"),
        }
        if "pixel_values" in batch:
            forward_kwargs["pixel_values"] = batch["pixel_values"]
            forward_kwargs["image_sizes"] = batch["image_sizes"]
            forward_kwargs["image_grid_thw"] = batch["image_grid_thw"]

        outputs = self.model(**forward_kwargs)
        return outputs.loss

    def save_checkpoint(self, epoch=None, step=None):
        """保存模型 checkpoint。"""
        if epoch is not None:
            save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        elif step is not None:
            save_path = os.path.join(self.output_dir, f"checkpoint-step-{step}")
        else:
            save_path = self.output_dir

        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        if hasattr(self.model, "tokenizer"):
            self.model.tokenizer.save_pretrained(save_path)
        print(f"  [Checkpoint] Saved to {save_path}")

    @torch.no_grad()
    def evaluate(self, global_step=None):
        """在每个验证集上分别计算平均 loss 并记录到 TensorBoard。"""
        if not self.eval_datasets:
            return {}

        self.model.eval()
        step_info = f"step {global_step}" if global_step is not None else "final"
        results = {}

        for eval_name, eval_dataset in self.eval_datasets.items():
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.per_device_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=self.dataloader_num_workers,
                pin_memory=True,
            )

            total_loss = 0.0
            total_batches = 0

            for batch in eval_dataloader:
                batch = self._prepare_batch(batch)
                if self.bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = self._forward_step(batch)
                else:
                    loss = self._forward_step(batch)
                total_loss += loss.item()
                total_batches += 1

            avg_eval_loss = total_loss / max(total_batches, 1)
            results[eval_name] = avg_eval_loss

            print(f"  [Eval @ {step_info}] {eval_name}: val_loss={avg_eval_loss:.4f} ({len(eval_dataset)} samples)")

            if global_step is not None:
                self.tb_writer.add_scalar(f"eval/{eval_name}_loss", avg_eval_loss, global_step)

        self.model.train()
        return results

    def train(self):
        """主训练循环。"""
        self.model.to(self.device)
        self.model.train()

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            pin_memory=True,
        )

        total_steps = (len(dataloader) // self.gradient_accumulation_steps) * self.num_epochs
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, total_steps)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"  Total optimization steps: {total_steps}")
        print(f"  Effective batch size: {self.per_device_batch_size * self.gradient_accumulation_steps}")
        if self.eval_datasets:
            eval_info = ", ".join(f"{name}({len(ds)})" for name, ds in self.eval_datasets.items())
            print(f"  Eval every {self.eval_steps} steps on: {eval_info}")

        # 训练开始前做一次初始 eval 作为 baseline
        if self.eval_datasets and self.eval_on_start:
            self.evaluate(global_step=0)

        global_step = 0
        accumulated_loss = 0.0
        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            progress_bar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                dynamic_ncols=True,
            )

            for step_in_epoch, batch in progress_bar:
                batch = self._prepare_batch(batch)

                if self.bf16:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = self._forward_step(batch)
                else:
                    loss = self._forward_step(batch)

                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()
                accumulated_loss += loss.item()
                epoch_loss += loss.item()
                epoch_steps += 1

                if (step_in_epoch + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.logging_steps == 0:
                        avg_loss = accumulated_loss / (self.logging_steps * self.gradient_accumulation_steps)
                        all_lrs = scheduler.get_last_lr()
                        current_lr = all_lrs[0]
                        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}", step=f"{global_step}/{total_steps}")
                        self.tb_writer.add_scalar("train/loss", avg_loss, global_step)
                        for group_idx, param_group in enumerate(optimizer.param_groups):
                            group_name = param_group.get("group_name", f"group_{group_idx}")
                            self.tb_writer.add_scalar(f"train/lr_{group_name}", all_lrs[group_idx], global_step)
                        accumulated_loss = 0.0

                    if self.eval_datasets and global_step % self.eval_steps == 0:
                        self.evaluate(global_step=global_step)

                    if self.save_strategy == "steps" and global_step % 500 == 0:
                        self.save_checkpoint(step=global_step)

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"  [Epoch {epoch + 1}/{self.num_epochs}] avg_loss={avg_epoch_loss:.4f}")

            if self.save_strategy == "epoch":
                self.save_checkpoint(epoch=epoch + 1)

        # 训练结束时做一次最终 eval
        if self.eval_datasets:
            self.evaluate(global_step=global_step)

        self.save_checkpoint()
        self.tb_writer.close()
        total_time = time.time() - start_time
        print(f"  Training complete! Total time: {total_time:.1f}s")
