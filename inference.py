import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
from vlm_model import SparkVLM, SparkVLMConfig
from dataset import smart_resize

def load_model(model_dir, device="cuda"):
    print(f"Loading model from {model_dir}...")
    config = SparkVLMConfig.from_pretrained(model_dir)
    # 增加 trust_remote_code 防止某些自定义层加载失败
    model = SparkVLM.from_pretrained(model_dir, config=config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)

    model.to(device)
    model.eval()
    return model, processor, tokenizer

def sample_next_token(logits, temperature=0.0, top_p=0.9):
    """支持贪婪和采样的通用函数"""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    
    # 执行采样
    logits = logits / temperature
    # 简单的 top_p 过滤（可选）
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    logits[sorted_indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate(model, processor, tokenizer, prompt, image_path=None,
             max_new_tokens=512, temperature=0.6, top_p=0.95, device="cuda"):
    
    # 1. 图像处理 (逻辑保持不变)
    image_pad_num = 0
    pixel_values, image_sizes, image_grid_thw = None, None, None
    if image_path:
        image = Image.open(image_path).convert("RGB")
        image = smart_resize(image, patch_size=14, compress_grid=2)
        width, height = image.size
        grid_t, grid_h, grid_w = 1, height // 28, width // 28
        image_pad_num = grid_t * grid_h * grid_w
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=device)
        image_sizes = torch.tensor([[height, width]], device=device)
        pixel_values = processor.image_processor(images=image, do_resize=False, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

    # 2. 文本处理
    # 注意：WikiArt 评测时 query 通常已经包含了选项，这里直接拼模板
    messages = [
        {"role": "system", "content": "You are a helpful multimodal assistant."},
        {"role": "user", "content": f"{prompt}\n<image>"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text.replace("<image>", "<|image_pad|>" * image_pad_num)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    # 3. 初始化 Attention Mask (核心修正)
    attention_mask = torch.ones_like(input_ids)
    
    # 4. 推理过程
    generated_ids = []
    eos_token_id = tokenizer.eos_token_id

    # Prefill
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    next_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    # Decode
    for _ in range(max_new_tokens):
        next_token_id = sample_next_token(next_logits, temperature, top_p)
        
        if next_token_id.item() == eos_token_id:
            break
        
        generated_ids.append(next_token_id)
        
        # 更新 mask：在后面补一个 1
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)

        outputs = model(
            input_ids=next_token_id,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    if generated_ids:
        return tokenizer.decode(torch.cat(generated_ids, dim=1)[0], skip_special_tokens=True)
    return ""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MonetVLM Inference")
    parser.add_argument("--model_dir", type=str, default="save/vlm_sft_full",
                        help="Path to the trained model directory")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="Describe this image.",
                        help="Prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model, processor, tokenizer = load_model(args.model_dir)
    response = generate(model, processor, tokenizer, args.prompt,
                        image_path=args.image, max_new_tokens=args.max_new_tokens)

    print("\n" + "=" * 50)
    print("User:", args.prompt)
    print("Assistant:", response)
    print("=" * 50 + "\n")
