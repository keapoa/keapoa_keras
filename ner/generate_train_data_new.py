import pandas as pd
import random
import re
import json
from collections import Counter
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
def recogenize_index_scope(start,end,labels):
    res = []
    for i in labels:
        if i[2]>=start and i[3]<end:
            res.append((i[0],i[1],i[2]-start,i[3]-start))
    return res
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
            sentence.append([s,recogenize_index_scope(start,end,labels)])
            k +="。"
            s = k
    sentence.append([s[:-1],recogenize_index_scope(end,len(text),labels)])
    #sentence里面的数据可能为空预测的时候要判断下
    return sentence
def judge(s1,s_lis):
    for s2 in s_lis:
        if s1==s2:
            return False
        if s1 in s2 or s2 in s1:
            return True
    return False
def train_valid_data(data_numbers,k,maxlen):
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
                #保留索引
                labels.append((s[4], s[1],int(s[2]),int(s[3])))
                for j in range(int(s[2]), int(s[3])):
                    if text_label[j]!="O":
                        print("############")
                    if j==int(s[2]):
                        text_label[j] = "B-" + s[1]
                    else:
                        text_label[j] = "I-" + s[1]
            data.append([text,text_label,labels])
    #设置个种子
    seed_value = 1213
    random.seed(seed_value)
    random.shuffle(data)
    count0 = 0
    count1 = 0
    #五折交叉验证
    print(len(data))
    length = int(len(data)//k)
    print(length)
    for j in range(k):
        m = j*length
        n = (j+1)*length

        with open("/home/wq/ner/train/ner_{}_{}.train".format(seed_value,j),"w",encoding = "utf-8") as f:

            for text,text_label,labels in (data[:m]+data[n:]):

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
        with open("/home/wq/ner/train/ner_{}_{}.valid".format(seed_value,j),"w",encoding = "utf-8") as f:

            for text,text_label,labels in data[m:n]:
                tmp = []
                #print("valid")
                if len(text) <= maxlen:
                    tmp.append([text, text_label])
                    valid_data.append([text, labels])
                else:
                    tmp.extend(cut_text(text, text_label, maxlen))
                    valid_data.extend(cut_text_valid(text, labels, maxlen))
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
        with open('/home/wq/ner/valid_data_{}_{}.json'.format(seed_value,j), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False))
            f.close()
train_valid_data(1000,10,510)

