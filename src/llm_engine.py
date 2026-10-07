import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

class LLMEngine:
    def __init__(self):
        print(f"[*] Đang tải LLM Gốc: {config.BASE_MODEL_ID} (Chuẩn FP16)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
        
        # device_map="auto" sẽ phân phối tự động các layers cho 2x GPU T4 trên Kaggle
        self.model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            torch_dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("[+] Khởi tạo LLM thành công trên Multi-GPU!")

    def post_process_output(self, raw_text: str):
        parsed_answer = ""
        parsed_unit = ""
        
        # Lọc đáp án
        ans_match = re.search(r"ANSWER:\s*(.*)", raw_text)
        if ans_match:
            raw_ans = ans_match.group(1).strip()
            parsed_answer = re.sub(r'(\s*(?:,|\b[vV]à\b)\s*)', '; ', raw_ans)
            
        # Lọc đơn vị và ép chuẩn ASCII
        unit_match = re.search(r"UNIT:\s*(.*)", raw_text)
        if unit_match:
            raw_unit = unit_match.group(1).strip()
            unit_mapping = {
                r"\Omega": "ohm", r"\mu F": "uF", r"\mu C": "uC", r"^\circ C": "C",
            }
            parsed_unit = raw_unit
            for latex, ascii_u in unit_mapping.items():
                parsed_unit = parsed_unit.replace(latex, ascii_u)
                
        return {
            "full_response": raw_text,
            "parsed_answer": parsed_answer,
            "parsed_unit": parsed_unit
        }

    def generate(self, messages: list) -> dict:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0, input_length:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        full_output = config.ASSISTANT_PREFILL + output_text
        return self.post_process_output(full_output)
