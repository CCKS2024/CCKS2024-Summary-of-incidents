from transformers import pipeline
import json

# 使用Hugging Face的transformers库加载预训练的摘要生成模型
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 读取上传的文件内容
file_path = "testa.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = file.read()

# 生成摘要
summary = summarizer(data, max_length=150, min_length=30, do_sample=False)

# 输出生成的摘要
summary_text = summary[0]['summary_text']
