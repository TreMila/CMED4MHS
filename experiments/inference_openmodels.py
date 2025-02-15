import argparse
import json
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)


SYSTEM_PROMPT = "你是一位专业的心理咨询师，具备丰富的心理学知识和咨询技巧，请你根据以下患者的话语，生成合适的回复，展现共情能力。\n"


def call_qwen(model, tokenizer, text, model_path, top_p):
    def chat(prompt):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=top_p
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    pred_answer = chat(text)
    
    return {
        "text": text,
        "pred_answer": pred_answer
    }

    
def call_internlm(model, tokenizer, text, model_path, top_p):
    def chat(prompt):
        response,_ = model.chat(
            tokenizer, 
            SYSTEM_PROMPT+prompt, 
            history=[], 
            do_sample=True, 
            top_p=top_p, 
            max_new_tokens=300)
        return response
    
    pred_answer = chat(text)
    
    return {
        "text": text,
        "pred_answer": pred_answer
    }


def call_yi(model, tokenizer, text, model_path, top_p):
    def chat(prompt, max_new_tokens=20):
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT+prompt}
        ]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = model.generate(
            input_ids.to('cuda'), 
            eos_token_id=tokenizer.eos_token_id, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    pred_answer = chat(text, max_new_tokens=300)
    
    return {
        "text": text,
        "pred_answer": pred_answer
    }


def call_deepseek(model, tokenizer, text, model_path, top_p):
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    def chat(prompt):
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT+prompt}
        ]
        input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_tensor.to(model.device), max_new_tokens=300, do_sample=True, top_p=top_p)
        response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return response
    
    pred_answer = chat(text)
    
    return {
        "text": text,
        "pred_answer": pred_answer
    }


MODEL_CLASSES = {
    "qwen":(AutoModelForCausalLM, AutoTokenizer, call_qwen),
    "internlm": (AutoModelForCausalLM, AutoTokenizer, call_internlm),
    "yi": (AutoModelForCausalLM, AutoTokenizer, call_yi),
    "deepseek": (AutoModelForCausalLM, AutoTokenizer, call_deepseek),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='auto', type=str)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--data_file', default=None, type=str)
    parser.add_argument('--output_file', default=None, type=str)
    args = parser.parse_args()
    
    
    # 加载模型
    model_class, tokenizer_class, call_func = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.base_model, trust_remote_code=True)
    model = model_class.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16,
        device_map="auto", 
        trust_remote_code=True
        )
    model.eval()
    
    # 加载测试集
    with open(args.data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    examples = [item['patient_text'] for item in data]
    
    print("Start inference.")

    with open(args.output_file, 'a', encoding='utf-8') as f:
        for example in tqdm(examples, desc="Predicting..."):
            result = call_func(model, tokenizer, "患者："+example, args.base_model)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    


if __name__ == '__main__':
    main()