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
def train_valid_data(data_numbers,rate,maxlen):
    #n = random.sample(range(0,data_numbers),data_numbers)
    data = []
    for i in range(data_numbers):
        txt_file = open("/home/wq/ner/train/{}.txt".format(i), encoding="utf-8")
        for text in txt_file:
            text = re.sub(r'\s+', '', text).strip()
            # texts.append(text)
            text_label = ["O" for i in text]
            labels = []
            n = ["O" for i in range(len(text))]
            ann_file = pd.read_csv("/home/wq/ner/train/{}.ann".format(i), header=None)
            recognize = {}
            for s in ann_file[0].values:
                s = s.split()
                labels.append((s[4], s[1]))
                # 读取过的标签

                recognize[s[4]] = recognize.get(s[4], 0) + 1
                # 这里重新获取每个实体的首末索引
                indexs = [m.start() for m in re.finditer(s[4], text)]
                # start_index = text.index(s[4])
                # end_index = start_index + len(s[4])
                # for start_index in indexs[recognize[s[4]]-1:]:
                # 判断下当前实体是否是实体重叠了。还有几个实体包含的。。。
                if judge(s[4], recognize.keys()):
                    continue
                if recognize[s[4]] == 1:
                    for start_index in indexs:
                        for j in range(start_index, start_index + len(s[4])):
                            if text_label[j] != "O":
                                # 忽略实体包含的
                                continue
                                #print("########################")
                            if j == start_index:
                                text_label[j] = "B-" + s[1]
                            else:
                                text_label[j] = "I-" + s[1]
                else:
                    start_index = indexs[recognize[s[4]] - 1]
                    for j in range(start_index, start_index + len(s[4])):
                        if text_label[j] != "O":
                            a = "B-" + s[1]
                            if text_label[j] == a:
                                break
                            else:
                                if j == start_index:
                                    text_label[j] = "B-" + s[1]
                                else:
                                    text_label[j] = "I-" + s[1]
        #打开文件，将数据写入文件中

            data.append([text,text_label,labels])
    """
    datas = []
    for text,label in data:
        if len(text) <= maxlen:
            datas.append([text,label])
        else:
            datas.extend(cut_text(text,label,maxlen))
    
    print(len(datas))
    """
    #设置个种子
    random.seed(1)
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
                    s = t[i] + " " + l[i]
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
                    s = t[i] + " " + l[i]
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
train_valid_data(1000,0.8,510)

