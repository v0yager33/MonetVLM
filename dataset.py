import json
import torch
import os
import math
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any
import torch.nn.functional as F

def smart_resize(image, patch_size=14, compress_grid=2, max_patches=1024):
    """
    Qwen-VL 原生动态分辨率：
    等比例缩放图像，并确保宽高严格为 base_unit (14 * 2 = 28) 的倍数。
    """
    base_unit = patch_size * compress_grid # 28
    w, h = image.size
    area = w * h
    max_area = max_patches * (base_unit ** 2)
    
    if area > max_area:
        scale = math.sqrt(max_area / area)
        w, h = int(w * scale), int(h * scale)
        
    new_w = max(base_unit, round(w / base_unit) * base_unit)
    new_h = max(base_unit, round(h / base_unit) * base_unit)
    return image.resize((new_w, new_h), Image.BICUBIC)

class VLMDataset(Dataset):
    def __init__(self, jsonl_path, processor, tokenizer, config=None, max_length=2048):
        self.data_list = []
        if isinstance(jsonl_path, str):
            jsonl_paths = [jsonl_path]
        else:
            jsonl_paths = jsonl_path
        for path in jsonl_paths:
            source_name = os.path.splitext(os.path.basename(path))[0]
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data_list.append(json.loads(line.strip()))
            print(f"  Loaded {path} ({source_name}), cumulative samples: {len(self.data_list)}")                    
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 支持多种格式的图像输入：单张路径、多张路径列表、带 path 字典的列表
        image_paths = []
        if "images" in item and isinstance(item["images"], list):
            image_paths = [img.get("path", "") if isinstance(img, dict) else img for img in item["images"]]
        elif "image_path" in item:
            if isinstance(item["image_path"], list):
                image_paths = item["image_path"]
            else:
                image_paths = [item["image_path"]]
        
        pixel_values_list = []
        image_sizes_list = []
        image_grid_thw_list = []
        image_pad_nums = []
        
        # 1. 加载并动态缩放每一张图像
        for img_path in image_paths:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
            else:
                image = Image.new('RGB', (384, 384), color='white')
                
            image = smart_resize(image, patch_size=14, compress_grid=2)
            w, h = image.size
            
            grid_t, grid_h, grid_w = 1, h // 28, w // 28
            image_pad_nums.append(grid_t * grid_h * grid_w)
            image_grid_thw_list.append([grid_t, grid_h, grid_w])
            
            pv = self.processor.image_processor(images=image, do_resize=False, return_tensors="pt")['pixel_values'].squeeze(0)
            pixel_values_list.append(pv)
            image_sizes_list.append([h, w])
        
        # 2. SFT 多轮对话支持与 Prompt Masking
        messages = [{"role": "system", "content": "You are a helpful multimodal assistant."}]
        
        if "conversations" in item:
            for conv in item["conversations"]:
                role = "user" if conv["from"] == "human" else "assistant"
                messages.append({"role": role, "content": conv["value"]})
        else:
            # 兼容普通指令微调
            img_placeholders = "".join(["\n<image>"] * len(image_paths))
            messages.append({"role": "user", "content": f"图片内容是什么{img_placeholders}"})
            messages.append({"role": "assistant", "content": item.get("text", "图片内容为空")})
            
        # 根据动态分辨率，按顺序替换文本中对应图片产生的 <|image_pad|> 数量
        global_img_idx = 0
        for msg in messages:
            while "<image>" in msg["content"] and global_img_idx < len(image_pad_nums):
                pad_str = "<|image_pad|>" * image_pad_nums[global_img_idx]
                msg["content"] = msg["content"].replace("<image>", pad_str, 1) # 逐个替换
                global_img_idx += 1
            # 移除没有图像匹配的多余标签
            if "<image>" in msg["content"]:
                msg["content"] = msg["content"].replace("<image>", "")
        
        # 3. 构造 ChatML 格式并只对 Assistant 的回答计算 Loss
        input_ids = []
        labels = []
        
        sys_text = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
        sys_ids = self.tokenizer(sys_text, add_special_tokens=False)['input_ids']
        input_ids.extend(sys_ids)
        labels.extend([-100] * len(sys_ids))
        
        for i in range(1, len(messages), 2):
            user_msg = messages[i]
            assistant_msg = messages[i+1] if i+1 < len(messages) else None
            
            user_text = f"<|im_start|>user\n{user_msg['content']}<|im_end|>\n"
            user_ids = self.tokenizer(user_text, add_special_tokens=False)['input_ids']
            input_ids.extend(user_ids)
            labels.extend([-100] * len(user_ids))
            
            if assistant_msg:
                prompt_text = "<|im_start|>assistant\n"
                prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
                input_ids.extend(prompt_ids)
                labels.extend([-100] * len(prompt_ids))
                
                ans_text = f"{assistant_msg['content']}<|im_end|>\n"
                ans_ids = self.tokenizer(ans_text, add_special_tokens=False)['input_ids']
                input_ids.extend(ans_ids)
                labels.extend(ans_ids)
                
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        res = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': [1] * len(input_ids),
        }
        
        if len(pixel_values_list) > 0:
            res['pixel_values'] = pixel_values_list
            res['image_sizes'] = image_sizes_list
            res['image_grid_thw'] = image_grid_thw_list
            
        return res

class VLMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        
        input_ids, labels, attention_mask = [], [], []
        flat_pixel_values = []
        flat_image_sizes = []
        flat_image_grid_thw = []
        
        for feature in features:
            pad_len = max_len - len(feature['input_ids'])
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(feature['labels'] + [-100] * pad_len)
            attention_mask.append(feature['attention_mask'] + [0] * pad_len)
            
            # 将所有样本的图像平铺到一个维度中
            if 'pixel_values' in feature:
                flat_pixel_values.extend(feature['pixel_values'])
                flat_image_sizes.extend(feature['image_sizes'])
                flat_image_grid_thw.extend(feature['image_grid_thw'])
            
        batch_dict = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
        
        if len(flat_pixel_values) > 0:
            # 找出这个 Batch 中所有图片的全局最大高宽，以此进行 Padding
            max_h = max(size[0] for size in flat_image_sizes)
            max_w = max(size[1] for size in flat_image_sizes)
            
            padded_pixel_values = []
            for pv in flat_pixel_values:
                pad_h, pad_w = max_h - pv.shape[1], max_w - pv.shape[2]
                padded_pv = F.pad(pv, (0, pad_w, 0, pad_h), value=0.0)
                padded_pixel_values.append(padded_pv)
                
            # 最终的 pixel_values 维度: [Total_Images_In_Batch, C, Max_H, Max_W]
            batch_dict['pixel_values'] = torch.stack(padded_pixel_values, dim=0)
            batch_dict['image_sizes'] = torch.tensor(flat_image_sizes, dtype=torch.long)
            batch_dict['image_grid_thw'] = torch.tensor(flat_image_grid_thw, dtype=torch.long)
            
        return batch_dict
