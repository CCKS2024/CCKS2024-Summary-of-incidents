import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader

# 文件路径
file_path = 'D:\\temp\\pythonProject\\testa.json'
output_path = 'D:\\temp\\pythonProject\\result.txt'

# 读取 JSON 文件
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 将多行 JSON 数据解析为单个列表
data_list = [json.loads(line) for line in lines]

# 提取 events 并构建新的数据集
records = []
for data in data_list:
    for event in data['events']:
        record = {
            'content': event['content'],
            'id': event['id']
        }
        records.append(record)

# 创建 HuggingFace 数据集
dataset = Dataset.from_list(records)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("./train1/checkpoint-3000")
model = AutoModelForSeq2SeqLM.from_pretrained("./train1/checkpoint-3000").to("cuda")

# 生成摘要函数
def generate_batch_summaries(batch):
    inputs = tokenizer(batch['content'], return_tensors="pt", max_length=512, truncation=True, padding=True).to("cuda")  # 增加 max_length 值
    summary_ids = model.generate(inputs['input_ids'], max_length=300, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)  # 增加 max_length 值
    summaries = [tokenizer.decode(s, skip_special_tokens=True).replace('\ufffd', '') for s in summary_ids]
    return summaries

# 批量处理
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size)

prev_prefix = None
results = {"summarizations": []}
all_results = []

for batch in dataloader:
    summaries = generate_batch_summaries(batch)
    for i, summary in enumerate(summaries):
        current_id = batch['id'][i]
        current_prefix = current_id.split('_')[0]
        # 如果前一个 id 的前缀与当前 id 的前缀不同，将前一个结果保存并重置 results
        if prev_prefix is not None and prev_prefix != current_prefix:
            all_results.append(results)
            results = {"summarizations": []}  # 重置 results
        # 添加当前结果到 results
        result = {
            "id": current_id,
            "event-summarization": summary
        }
        results["summarizations"].append(result)
        # 更新前一个 id 的前缀
        prev_prefix = current_prefix

# 添加最后一个前缀的结果
if results["summarizations"]:
    all_results.append(results)

# 写入文件
with open(output_path, 'w', encoding='utf-8') as f:
    for res in all_results:
        json_results = json.dumps(res, ensure_ascii=False)
        f.write(json_results)
        f.write('\n')  # 添加换行符，确保每个 JSON 对象在一行

print(f"结果已保存到 {output_path}")
