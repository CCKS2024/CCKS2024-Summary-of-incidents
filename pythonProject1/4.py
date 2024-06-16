#读取文件
f='邯郸一补习班男教师因殴打学生被拘，涉事机构涉嫌无证被查封_教育家_澎湃新闻-ThePape。'
f1='10631'
#清洗数据
import re
import jieba
text = re.sub(r'[[0-9]*]',' ',f)#去除类似[1]，[2]
text = re.sub(r'\s+',' ',text)#用单个空格替换了所有额外的空格
sentences = re.split('(。|！|\!|\.|？|\?)',text)#分句


#加载停用词

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
stopwords = stopwordslist("停用词.txt")

#词频
word2count = {} #line 1
for word in jieba.cut(text): #对整个文本分词
    if word not in stopwords:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
for key in word2count.keys():
    word2count[key] = word2count[key] / max(word2count.values())


#计算句子得分
sent2score = {}
for sentence in sentences:
    for word in jieba.cut(sentence):
        if word in word2count.keys():
            if len(sentence)<300:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]

#字典排序
def dic_order_value_and_get_key(dicts, count):
    # by hellojesson
    # 字典根据value排序，并且获取value排名前几的key
    final_result = []
    # 先对字典排序
    sorted_dic = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    tmp_set = set()  # 定义集合 会去重元素 --此处存在一个问题，成绩相同的会忽略，有待改进
    for item in sorted_dic:
        tmp_set.add(item[1])
    for list_item in sorted(tmp_set, reverse=True)[:count]:
        for dic_item in sorted_dic:
            if dic_item[1] == list_item:
                final_result.append(dic_item[0])
    return final_result

#摘要输出
final_resul=dic_order_value_and_get_key(sent2score,1)
f2='{'+'"summarizations":'+' [{"id": "'+f1+'_1", '+'"event-summarization": "'+"".join(final_resul)+'"}]}'
print(f2)
with open('output.txt', 'a', encoding='utf-8') as file:
    file.write(f2)

