import json
from datasets import Dataset, DatasetDict

# 从文件中加载 JSON 数据
file_path = 'S:\\1NLP\\Bert\\train.json'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 将多行 JSON 数据解析为单个列表
data_list = [json.loads(line) for line in lines]
# 提取 events 和 summarizations 并构建新的数据集
records = []
for data in data_list:
    for event in data['events']:
        record = {
            'content': event['content'],
            'title': next(summ['event-summarization'] for summ in data['summarizations'] if summ['id'] == event['id'])
        }
        records.append(record)

# 创建 HuggingFace 数据集
dataset = Dataset.from_list(records)

# 将数据集拆分成训练集和测试集
ds = dataset.train_test_split(test_size=0.1, seed=42)

# 打印数据集样本
print(ds)
