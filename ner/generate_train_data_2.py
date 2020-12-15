import pandas as pd
import random
import re
import json
from collections import Counter

#获取实体字典

def get_regonize_list():
    labels = []
    for i in range(1000):

        ann_file = pd.read_csv("/home/wq/ner/train/{}.ann".format(i), header=None)
        for s in ann_file[0].values:
            s = s.split()
            labels.append((s[4], s[1]))
            # 这里重新获取每个实体的首末索引
    a = list(set(labels))

    b = [i[0] for i in a]
    c = dict(Counter(b))
    #这里仅仅获取具有唯一标识的字典
    e = [k for k, v in c.items() if v == 1]

    return e

def cut_text(text,label,maxlen):
    sentence = []
    s = ""
    start = 0
    end = 0
    for k in text.split("。"):
        if len(s+k)<=maxlen:
            k += "。"
            s+=k
        else:
            start = end
            end = start+len(s)
            sentence.append([s,label[start:end]])
            k +="。"
            s = k
    assert len(s[:-1])==len(label[end:])
    sentence.append([s[:-1],label[end:]])
    return sentence
def cut_text_valid(text,labels,maxlen):
    sentence = []
    s = ""
    start = 0
    end = 0
    for k in text.split("。"):
        if len(s+k)<=maxlen:
            k += "。"
            s+=k
        else:
            start = end
            end = start+len(s)
            sentence.append(s)
            k +="。"
            s = k
    sentence.append(s[:-1])
    sentence.append(labels)
    #sentence里面的数据可能为空预测的时候要判断下
    return sentence
def judge(s1,s_lis):
    for s2 in s_lis:
        if s1==s2:
            return False
        if s1 in s2 or s2 in s1:
            return True
    return False
def train_valid_data(data_numbers,rate,maxlen,reg_list):
    #n = random.sample(range(0,data_numbers),data_numbers)
    data = []
    for i in range(data_numbers):
        txt_file = open("/home/wq/ner/train/{}.txt".format(i), encoding="utf-8")
        for text in txt_file:
            #text = re.sub(r'\s+', '', text).strip()
            # texts.append(text)
            #去掉尾部空格
            text = text.rstrip()
            text_label = ["O" for i in text]
            labels = []
          
            ann_file = pd.read_csv("/home/wq/ner/train/{}.ann".format(i), header=None)
            for s in ann_file[0].values:
                s = s.split()
                labels.append((s[4], s[1]))
                for j in range(int(s[2]), int(s[3])):
                    if text_label[j]!="O":
                        print("############")
                    if j==int(s[2]):
                        text_label[j] = "B-" + s[1]
                    else:
                        text_label[j] = "I-" + s[1]
                #将具有唯一标识的实体进行补充标注
                if s[4] in reg_list:
                    indexs = [m.start() for m in re.finditer(s[4], text)]
                    for start_index in indexs:
                        for i in range(start_index,start_index+len(s[4])):
                            if text_label[i]!="O":
                                break
                            else:
                                if i==start_index:
                                    text_label[j] = "B-" + s[1]
                                else:
                                    text_label[j] = "I-" + s[1]

            data.append([text,text_label,labels])
    #设置个种子
    random.seed(2020)
    random.shuffle(data)
    count0 = 0
    count1 = 0
    with open("/home/wq/ner/train/ner.train","w",encoding = "utf-8") as f:
        for text,text_label,labels in data[:int(data_numbers*rate)]:

            tmp = []
            if len(text) <= maxlen:
                tmp.append([text, text_label])
            else:
                tmp.extend(cut_text(text, text_label, maxlen))
            for t,l in tmp:
                count0 += 1
                if not t:
                    continue
                for i in range(len(t)):
                    s = t[i] + "\t" + l[i]
                    f.write(s)
                    f.write("\n")
                f.write("\n")
        f.close()

    valid_data = []
    with open("/home/wq/ner/train/ner.valid","w",encoding = "utf-8") as f:
        for text,text_label,labels in data[int(rate*len(data)):]:
            tmp = []
            if len(text) <= maxlen:
                tmp.append([text, text_label])
                valid_data.append([text, labels])
            else:
                tmp.extend(cut_text(text, text_label, maxlen))
                valid_data.append(cut_text_valid(text, labels, maxlen))
            for t, l in tmp:
                count1+=1
                if not t:
                    continue
                for i in range(len(t)):
                    s = t[i] + "\t" + l[i]
                    f.write(s)
                    f.write("\n")
                f.write("\n")
        f.close()
    print(len(valid_data))
    print(valid_data[0])
    result = {}
    result["valid_data"] = valid_data
    with open('/home/wq/ner/valid_data.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False))
        f.close()
reg_list = get_regonize_list()
train_valid_data(1000,0.8,510,reg_list)

