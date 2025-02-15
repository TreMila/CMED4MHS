import jieba
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import json
import pandas as pd 
from tqdm import tqdm

import numpy as np


def bleu(reference, candidate):
    reference_tokenized = [list(jieba.cut(reference))]
    candidate_tokenized = list(jieba.cut(candidate))
    weights = [
        (1.0, ),            # BLEU-1
        (0.5, 0.5),         # BLEU-2
        (1/3, 1/3, 1/3),    # BLEU-3
        (0.25, 0.25, 0.25, 0.25)  # BLEU-4
    ]
    bleu_score = [
        sentence_bleu(reference_tokenized, candidate_tokenized, smoothing_function=SmoothingFunction().method1, weights=w)
        for w in weights
    ]
    result = {f"bleu{i+1}": score for i, score in enumerate(bleu_score)}
    return result


def rouge(reference, candidate):
    rouge = Rouge()
    reference_tokenized = ' '.join(jieba.cut(reference))
    candidate_tokenized = ' '.join(jieba.cut(candidate))
    score = rouge.get_scores(candidate_tokenized, reference_tokenized, avg=True)
    result = {key: value['f'] for key, value in score.items()}
    return result



def machine_metric(generate_refs, generate_preds):
    results = [] 
    try:
        for idx,(ref, pred) in enumerate(zip(generate_refs, generate_preds)):
            if len(pred) == 0:
                continue
            bleu_score = bleu(ref, pred)
            rouge_score = rouge(ref, pred)
            results.append({**bleu_score, **rouge_score})
    except Exception as e:
        print(idx, e)
        if "maximum recursion depth exceeded in comparison" in str(e):
            results.append({"bleu1":0.0, "bleu2":0.0, "bleu3":0.0, "bleu4":0.0, "rouge-1":0.0, "rouge-2":0.0, "rouge-l":0.0})

    avg_res = {}
    for key in results[0].keys():
        avg_res[key] = sum([item[key] for item in results]) / len(results)
    
    return avg_res 


def main():
    test_file = 'path/to/test_examples.jsonl'
    pred_dir = ['path/to/pred_file1.josnl', 'path/to/pred_file2.jsonl', '...']

    with open(test_file, 'r') as f:
        data = [json.loads(line) for line in f]
    generate_refs = [item['doctor_text'] for item in data]

    metrics = []
    for pred_file in tqdm(pred_dir, total=len(pred_dir), desc='Calculating metrics...'):
        with open(pred_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
            
        generate_preds = [item['pred_answer'] for item in results]
        metric = machine_metric(generate_refs, generate_preds)
        metrics.append(metric)

    # 获取模型名称
    index_names = [file.split('/')[-1].split('generate_')[-1].split('.jsonl')[0] for file in pred_dir]
    # 保存结果
    df = pd.DataFrame(metrics, index=index_names)
    df.to_excel('path/to/save_result.xlsx')
