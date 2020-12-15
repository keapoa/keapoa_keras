import pandas as pd
import random
import re
import json
from collections import Counter
query_sign_map = {
"tags":['DRUG', 'DRUG_INGREDIENT', 'DISEASE',"SYMPTOM","SYNDROME","DISEASE_GROUP","FOOD","FOOD_GROUP","PERSON_GROUP","DRUG_GROUP","DRUG_DOSAGE","DRUG_TASTE","DRUG_EFFICACY"],
    "natural_query":{
        "DRUG":"找出水蜜丸,五灵脂,孕妇金花片,桂花药膏,百安洗液，抗生素,富康胶囊,感冒药等中药名称",
        "DRUG_INGREDIENT":"找出当归,人参,反藜芦,黄藤素等中药药物成分",
        "DISEASE":"找出高血压,慢性病,药疹,阴道炎,肝癌,更年期综合征,泌尿呼吸道感染等疾病名称",
        "SYMPTOM":"找出头晕,心悸,小腹胀痛,药疹,阴道炎,气血不足,月经后错等疾病症状",
        "SYNDROME":"找出血瘀,气滞,气血不足,血热有瘀,气血两虚,更年期综合征等证候",
        "DISEASE_GROUP":"找出肾病,慢性病,支气管炎,阴道炎,更年期综合症,胃癌等人体组织部位疾病",
        "FOOD":"找出苹果,萝卜,木耳,五灵脂等食物",
        "FOOD_GROUP":"找出油腻食物,鱼类,辛辣,忌辛辣,生冷等中医药禁用食物",
        "PERSON_GROUP":"找出孕妇,老年患者,糖尿病患者,发热病人,青春期少女等中医药适用及禁用人群",
        "DRUG_GROUP":"找出止咳药,感冒药,非处方药物等某类药品",
        "DRUG_DOSAGE":"找出浓缩丸,糖衣片,软胶囊,水蜜丸,硬胶囊剂等药物剂型",
        "DRUG_TASTE":"找出味甘,酸涩,辛辣,咸,气凉等药物性味",
        "DRUG_EFFICACY":"找出滋阴补肾,五灵脂,去瘀生新,活血化瘀,促进免疫等中药功效"}
}
def recogenize_index_scope(start,end,labels):
    res = []
    for i in labels:
        if i[0]>=start and i[1]<end:
            res.append((i[0]-start,i[1]-start,i[2]))
    return res
def cut_text(text,labels,maxlen):
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
def cut_text_test(text,maxlen):
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
    return sentence
def train_valid_test_data(data_numbers,rate,maxlen):
    #n = random.sample(range(0,data_numbers),data_numbers)
    data = []
    for i in range(data_numbers):
        txt_file = open("/home/wq/ner/train/{}.txt".format(i), encoding="utf-8")
        for text in txt_file:

            #去掉尾部空格
            text = text.rstrip()
            labels = []
            ann_file = pd.read_csv("/home/wq/ner/train/{}.ann".format(i), header=None)
            for s in ann_file[0].values:
                s = s.split()
                #保留索引
                labels.append((int(s[2]),int(s[3]),s[1]))

            data.append([text,labels])
    #设置个种子
    seed_value = 2020
    random.seed(seed_value)
    random.shuffle(data)
    train_data = []
    for text,labels in data[:int(data_numbers*rate)]:

        if len(text) <= maxlen:
            for k in query_sign_map["natural_query"]:
                query = query_sign_map["natural_query"][k]
                start_label = [0]*(len(query)+len(text)+3)
                end_label = [0]*(len(query)+len(text)+3)
                #
                T = []
                left_length = len(query)+2
                for (start_index,end_index,label) in labels:
                    if label == k:
                        start_label[left_length+start_index] = 1
                        end_label[left_length+end_index-1] = 1
                        T.append(((start_index,end_index,label)))
                train_data.append((query,text,start_label,end_label,T,k))

        else:
            for t,labs in cut_text(text, labels, maxlen):
                for k in query_sign_map["natural_query"]:
                    query = query_sign_map["natural_query"][k]
                    start_label = [0] * (len(query) + len(t) + 3)
                    end_label = [0] * (len(query) + len(t) + 3)
                    #
                    T = []
                    left_length = len(query) + 2
                    for (start_index, end_index, l) in labs:
                        if l == k:
                            start_label[left_length + start_index] = 1
                            end_label[left_length + end_index - 1] = 1
                            T.append(((start_index, end_index, l)))
                    train_data.append((query, t, start_label, end_label, T, k))

    print(train_data[10])
    valid_data = []

    for text,labels in data[int(data_numbers*rate):]:

        if len(text) <= maxlen:
            for k in query_sign_map["natural_query"]:
                query = query_sign_map["natural_query"][k]
                start_label = [0]*(len(query)+len(text)+3)
                end_label = [0]*(len(query)+len(text)+3)
                #
                T = []
                left_length = len(query)+2
                for (start_index,end_index,label) in labels:
                    if label == k:
                        start_label[left_length+start_index] = 1
                        end_label[left_length+end_index-1] = 1
                        T.append(((start_index,end_index,label)))
                valid_data.append((query,text,start_label,end_label,T,k))
        else:
            for t,labs in cut_text(text, labels, maxlen):
                for k in query_sign_map["natural_query"]:
                    query = query_sign_map["natural_query"][k]
                    start_label = [0] * (len(query) + len(t) + 3)
                    end_label = [0] * (len(query) + len(t) + 3)
                    #
                    T = []
                    left_length = len(query) + 2
                    for (start_index, end_index, l) in labs:
                        if l == k:
                            start_label[left_length + start_index] = 1
                            end_label[left_length + end_index - 1] = 1
                            T.append(((start_index, end_index, l)))
                    valid_data.append((query, t, start_label, end_label, T, k))

    print(valid_data[10])
    #测试集数据
    test_data = []
    for i in range(1000,1500):
        txt_file = open("/home/wq/ner/test/{}.txt".format(i), encoding="utf-8")
        for text in txt_file:
            #去掉尾部空格
            text = text.rstrip()
            tmps = []

            if len(text) <= maxlen:
                tmp = []
                for k in query_sign_map["natural_query"]:
                    query = query_sign_map["natural_query"][k]

                    tmp.append((query,text,k))
                tmps.append(tmp)
            else:

                for s in cut_text_test(text, maxlen):
                    tmp = []
                    for k in query_sign_map["natural_query"]:
                        query = query_sign_map["natural_query"][k]
                        tmp.append((query,s,k))
                    tmps.append(tmp)
        test_data.append(tmps)

    print(test_data[0])
    result = {}
    result["valid"] = valid_data
    result["train"] = train_data
    result["test"] = test_data
    with open('/home/wq/ner/mrc/data/mrc_base_data_seed{}.json'.format(seed_value), 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False))
        f.close()
train_valid_test_data(1000,0.8,300)

