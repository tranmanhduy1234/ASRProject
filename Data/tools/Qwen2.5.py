import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import math
from tqdm import tqdm

# --- CẤU HÌNH ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output_merged.txt"
BATCH_SIZE = 8       # Batch size cho MỖI GPU (Tổng throughput = 16)
NUM_GPUS = 2         # Bạn có 2 GPU T4

def load_model_on_gpu(gpu_id):
    """
    Hàm load model riêng cho từng GPU
    """
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Đang tải model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model cụ thể vào GPU được chỉ định
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device, # Ép model nằm trọn trên GPU này
        torch_dtype="auto",
        # load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.float16
    )
    return model, tokenizer

def run_worker(gpu_id, lines, output_temp_file):
    """
    Worker xử lý một phần dữ liệu trên GPU được chỉ định
    """
    try:
        model, tokenizer = load_model_on_gpu(gpu_id)
        
        system_prompt = "You are a professional translator. Translate the following Vietnamese text to English. Do not add any explanation."
        
        translated_lines = []
        
        # Batch processing
        total_batches = math.ceil(len(lines) / BATCH_SIZE)
        
        # Sử dụng tqdm với vị trí khác nhau để không đè lên nhau trên terminal
        progress_bar = tqdm(total=len(lines), desc=f"GPU {gpu_id}", position=gpu_id, leave=True)

        for i in range(0, len(lines), BATCH_SIZE):
            batch_texts = lines[i : i + BATCH_SIZE]
            
            # Chuẩn bị Prompt
            batch_prompts = []
            for text in batch_texts:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(prompt)

            # Tokenize & Move to GPU
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            ).to(f"cuda:{gpu_id}")

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )

            # Decode
            input_len = inputs.input_ids.shape[1]
            output_ids = generated_ids[:, input_len:]
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Lưu kết quả tạm vào list
            cleaned = [s.strip().replace("\n", " ") for s in decoded]
            translated_lines.extend(cleaned)
            
            progress_bar.update(len(batch_texts))
        
        progress_bar.close()

        # Ghi file tạm của GPU này
        print(f"[GPU {gpu_id}] Hoàn thành! Đang ghi file tạm: {output_temp_file}")
        with open(output_temp_file, "w", encoding="utf-8") as f:
            for line in translated_lines:
                f.write(line + "\n")
                
    except Exception as e:
        print(f"[GPU {gpu_id}] Lỗi nghiêm trọng: {e}")

def main():
    # 1. Cấu hình Multiprocessing cho CUDA
    # Bắt buộc dùng 'spawn' khi làm việc với CUDA multiprocess
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 2. Đọc dữ liệu
    if not os.path.exists(INPUT_FILE):
        print("Không tìm thấy file input.")
        # Tạo file giả lập để test
        with open(INPUT_FILE, "w", encoding="utf-8") as f:
            f.write("Xin chào\n" * 20) 
        print("Đã tạo file mẫu input.txt")
        
    print(f"📂 Đang đọc {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(all_lines)
    print(f"📊 Tổng số dòng: {total_lines}")

    # 3. Chia dữ liệu cho các GPU (Splitting)
    # GPU 0 lấy nửa đầu, GPU 1 lấy nửa sau
    chunk_size = math.ceil(total_lines / NUM_GPUS)
    chunks = [all_lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
    
    # Đảm bảo list chunks đủ độ dài bằng số GPU (trường hợp file ít dòng)
    while len(chunks) < NUM_GPUS:
        chunks.append([])

    # 4. Khởi tạo Process
    processes = []
    temp_files = []

    print(f"🚀 Bắt đầu khởi chạy {NUM_GPUS} workers trên 2 GPU T4...")
    
    for rank in range(NUM_GPUS):
        temp_file = f"temp_output_gpu_{rank}.txt"
        temp_files.append(temp_file)
        
        # Tạo process
        p = mp.Process(
            target=run_worker, 
            args=(rank, chunks[rank], temp_file)
        )
        p.start()
        processes.append(p)

    # 5. Đợi tất cả hoàn thành (Join)
    for p in processes:
        p.join()

    print("\n✅ Tất cả GPU đã xử lý xong. Đang gộp file...")

    # 6. Gộp kết quả (Merging)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                # Xóa file tạm
                os.remove(temp_file)
    
    print(f"🎉 Hoàn tất! Kết quả lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()