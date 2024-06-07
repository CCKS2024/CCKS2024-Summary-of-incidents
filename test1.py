import json

import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

# # 加载数据集
# ds = load_from_disk("./nlpcc_2017/")
#
# ds = load_from_disk(r"S:\\1NLP\\Bert\\T5\\nlpcc_2017")
# # 测试集留100条，设定随机种子
# ds = ds.train_test_split(test_size=100, seed=42)
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

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# 定义处理函数
def process_func(examples):
    inputs = tokenizer(examples['content'], max_length=384, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples['title'], max_length=64, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

# 处理数据集
tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds["train"].column_names)

# 检查一个处理后的样本
print(tokenizer.decode(tokenized_ds["train"][0]["input_ids"]))
print(tokenized_ds["train"][0]["labels"])

# 设置训练参数
args = Seq2SeqTrainingArguments(
    output_dir="./summary_bart",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    logging_steps=8,
    num_train_epochs=1
)

# 创建 Trainer
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
)

# 训练模型
trainer.train()

# 生成摘要函数
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=384, truncation=True)
    summary_ids = model.generate(inputs['input_ids'].to("cuda"), max_length=64, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 测试生成摘要
input_text = ds["test"][-1]["content"]
summary = generate_summary(input_text)
print("摘要:", summary)

# 预测测试集
def predict_test():
    predict = []
    with torch.no_grad():
        for d in ds["test"]:
            summary = generate_summary(d["content"])
            predict.append(summary)
            print("curID:", len(predict))
    return predict

result = predict_test()
print(result)
