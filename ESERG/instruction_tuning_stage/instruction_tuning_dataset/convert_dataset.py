import json
import random

# 划分训练集、验证集、测试集
def split_dataset(dataset, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    random.seed(42)
    random.shuffle(dataset)
    train_size = int(len(dataset) * train_ratio)
    valid_size = int(len(dataset) * valid_ratio)
    test_size = len(dataset) - train_size - valid_size
    return dataset[:train_size], dataset[train_size:train_size+valid_size], dataset[-test_size:]


# 保存数据集
def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
            

# 转为训练框架所需的数据集格式
def convert_format(dataset):
    new_dataset = []
    
    for item in dataset:
        conversations = []
        conversations.append({
                "from": "human",
                "value": item["instruction"] + '\n' + item["input"]
        })
        conversations.append({
            "from": "gpt",
            "value": item["output"]
        })
        

        new_dataset.append({'conversations':conversations})
    return new_dataset

# 构造指令微调数据集数据集
# 子任务一：情绪识别
def construct_emo_dataset(data):
    random.seed(42)
    with open('./emo_template.json','r') as f:
        emo_template = json.load(f)

    results = []
    for item in data:
        emo_template_idx = random.choice(range(5))
        emo_dic = emo_template[emo_template_idx].copy()
        emo_dic['input'] = "患者：" + item['text']
        emo_dic['output'] = item['emotion']
        results.append(emo_dic)
    
    emo_dataset = convert_format(results)
    return emo_dataset

# 子任务二：策略选择
def construct_stra_dataset(data):
    random.seed(42)
    with open('./stra_template.json','r') as f:
        stra_template = json.load(f)

    results = []
    for item in data:
        stra_template_idx = random.choice(range(5))
        stra_dic = stra_template[stra_template_idx].copy()
        stra_dic['input'] = "患者：" + item['text']
        stra_dic['output'] = item['strategy']
        results.append(stra_dic)
    
    stra_dataset = convert_format(results)
    return stra_dataset

# 子任务三：回复生成
def construct_resgen_dataset(data):
    with open('./rsp_template.json','r') as f:
        template = json.load(f)

    results = []
    for item in data:
        template_idx = random.choice(range(5))
        dic = template[template_idx].copy()
        dic['input'] = "患者：" + item['text'] + "\n" + "患者情绪：" + item['emotion'] + "\n" + "心理咨询师回应策略：" + item['strategy']
        dic['output'] = item['response']
        results.append(dic)
        
    resgen_dataset = convert_format(results)
    return resgen_dataset



if __name__ == '__main__':
    # 读取原始数据，划分数据集
    with open('path/to/CMED.jsonl','r') as f:
        data = [json.loads(line) for line in f]
    
    # 构造微调数据集
    emo_dataset = construct_emo_dataset(data)
    stra_dataset = construct_stra_dataset(data)
    resgen_dataset = construct_resgen_dataset(data)
    
    # 划分数据集
    emo_train_data, emo_valid_data, emo_test_data = split_dataset(emo_dataset)
    stra_train_data, stra_valid_data, stra_test_data = split_dataset(stra_dataset)
    resgen_train_data, resgen_valid_data, resgen_test_data = split_dataset(resgen_dataset)
    
    # 保存数据集
    save_jsonl(emo_train_data, 'path/to/save_path.jsonl')
    save_jsonl(emo_valid_data, 'path/to/save_path.jsonl')
    save_jsonl(emo_test_data, 'path/to/save_path.jsonl')
    
    save_jsonl(stra_train_data, 'path/to/save_path.jsonl')
    save_jsonl(stra_valid_data, 'path/to/save_path.jsonl')
    save_jsonl(stra_test_data, 'path/to/save_path.jsonl')
    
    save_jsonl(resgen_train_data, 'path/to/save_path.jsonl')
    save_jsonl(resgen_valid_data, 'path/to/save_path.jsonl')
    save_jsonl(resgen_test_data, 'path/to/save_path.jsonl')
    