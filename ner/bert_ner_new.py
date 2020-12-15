import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import *
from keras.models import Model
from recompute import recompute_grad
from collections import Counter
import json
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.models import Model,load_model
import re
import os
import random
seed_value = 1213
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["RECOMPUTE"] = "1"
#修饰层
class MyDense(Dense):
    @recompute_grad
    def call(self, inputs):
        return super(MyDense, self).call(inputs)
class MyConditionalRandomField(ConditionalRandomField):
    @recompute_grad
    def call(self, inputs):
        return super(MyConditionalRandomField, self).call(inputs)
maxlen = 512
epochs = 7
batch_size = 5
bert_layers = 24
unit = 768
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000 # 必要时扩大CRF层的学习率

# bert配置
config_path = '/home/wq/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/wq/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/wq/chinese_L-12_H-768_A-12/vocab.txt'

#Robert配置
config_path = '/home/wq/ner/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/wq/ner/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/wq/ner/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

#Roberta-large配置

config_path = '/home/wq/ner/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '/home/wq/ner/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '/home/wq/ner/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'


#加载实体字典
labels_dic = json.load(open("/home/wq/ner/labels.json",encoding="utf-8"))["labels"]
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split('\t')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# 标注数据
#train_data = load_data('/home/wq/ner/train/ner.train')
#valid_data = load_data('/home/wq/ner/train/ner.valid')
#test_data = load_data('/home/wq/ner/test/ner.test')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射
labels = ['DRUG', 'DRUG_INGREDIENT', 'DISEASE',"SYMPTOM","SYNDROME","DISEASE_GROUP","FOOD","FOOD_GROUP","PERSON_GROUP","DRUG_GROUP","DRUG_DOSAGE","DRUG_TASTE","DRUG_EFFICACY"]
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            #把长度都弄成maxlen
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []




class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l) for w, l in entities],[(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l,mapping[w[0]][0],mapping[w[-1]][-1]+1) for w, l in entities]


def build_model():
    with tf.device("/gpu:1"):
        model = build_transformer_model(
            config_path,
            checkpoint_path, )
    # output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers-1)

    with tf.device("/gpu:0"):
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        output = model.get_layer(output_layer).output
        # output = Bidirectional(LSTM(unit, return_sequences=True))(output)
        output = MyDense(num_labels)(output)
        CRF = MyConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        output = CRF(output)

        model = Model(model.input, output)
        # model = multi_gpu_model(model,2)
        model.summary()
        model.compile(
            loss=CRF.sparse_loss,
            optimizer=Adam(learing_rate),
            metrics=[CRF.sparse_accuracy])

    return model,CRF


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text)[0])
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0
        self.best_val_f1_v = 0
    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        #print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        f1_v,precision_v,recall_v = evaluate_valid(dev_data)
        # 保存最优
        if normal_train and not cross_train:
            if f1 >= self.best_val_f1:
                self.best_val_f1 = f1
                model.save_weights('./best_model.weight')
            print(
                'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
                (f1, precision, recall, self.best_val_f1)
            )
            if f1_v >= self.best_val_f1_v:
                self.best_val_f1_v = f1_v
                model.save_weights('./best_model_new.weights')
            print(
                'valid:  f1_v: %.5f, precision_v: %.5f, recall_v: %.5f, best f1_v: %.5f\n' %
                (f1_v, precision_v, recall_v,self.best_val_f1_v))
        if cross_train and not normal_train:
            if f1 >= self.best_val_f1:
                 self.best_val_f1 = f1
                 model.save_weights('./best_model_{}_{}.weights'.format(seed_value,id))
            print(
                 'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
                 (f1, precision, recall, self.best_val_f1)
                 )
            if f1_v >= self.best_val_f1_v:
                 self.best_val_f1_v = f1_v
                 model.save_weights('./best_model_new_{}_{}.weights'.format(seed_value,id))
            print(
                 'valid:  f1_v: %.5f, precision_v: %.5f, recall_v: %.5f, best f1_v: %.5f\n' %
                 (f1_v, precision_v, recall_v,self.best_val_f1_v))


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
    #sentence里面的数据可能为空预测的时候要判断下
    return sentence

def test_data(data_numbers,maxlen):
    data = []
    source_data = []
    for i in range(1000,1000+data_numbers):
        txt_file = open("/home/wq/ner/test/{}.txt".format(i),encoding = "utf-8")
        for text in txt_file:
            source_data.append(text)
            #text = re.sub(r'\s+', '', text).strip()
            text = text.rstrip()
            if len(text) <= maxlen:
                data.append([text])
            else:
                data.append(cut_text_test(text,maxlen))
    return data,source_data

def evaluate_valid(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for t in tqdm(data):
        R = set(NER.recognize(t[0])[1])
        T = set([tuple(i) for i in t[1]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

#构建自己的验证集
#dev_data = json.load(open("/home/wq/ner/valid_data.json",encoding="utf-8"))["valid_data"]
#print(dev_data[0])
if __name__ == '__main__':
    import sys
    arg = sys.argv
    if arg[1]=="train":
        normal_train = True
        cross_train = False
        model, CRF = build_model()
        NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
        train_data = load_data('/home/wq/ner/train/ner.train')
        valid_data = load_data('/home/wq/ner/train/ner.valid')
        dev_data = json.load(open("/home/wq/ner/valid_data.json",encoding="utf-8"))["valid_data"]
        evaluator = Evaluator()
        train_generator = data_generator(train_data, batch_size)

        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )

    elif arg[1]=="test":
        def predict(data, source_data):
            """预测测试集
            """
            from collections import Counter

            for i in range(len(data)):
                recognize = []
                cur_text = 0
                tmp = []
                for text in data[i]:
                    _,r = NER.recognize(text)
                    for recognize,l,start,end in r:
                        tmp.append((l,cur_text+start,cur_text+end,recognize))
                    cur_text += len(text) 
                       
                with open("/home/wq/ner/test/{}.ann".format(1000 + i), "w", encoding="utf-8") as f:
                    for i in range(len(tmp)):
                        f.write("T{}\t{} {} {}\t{}".format(i + 1, tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3]))
                        f.write("\n")
                    f.close()
                    # 生成测试数据
        data, source_data = test_data(500, 510)
        model, CRF = build_model()
        model.load_weights('./best_model_2020_0.weights')
        NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
        predict(data, source_data)
    elif arg[1]=="cross_train":
        normal_train = False
        cross_train = True
        for id in range(10):
            K.clear_session()
            print("第{}次训练开始".format(id))
            model,CRF = build_model()
            NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
            train_data = load_data('/home/wq/ner/train/ner_{}_{}.train'.format(seed_value,id))
            valid_data = load_data('/home/wq/ner/train/ner_{}_{}.valid'.format(seed_value,id))
            print(len(train_data),len(valid_data))
            dev_data = json.load(open("/home/wq/ner/valid_data_{}_{}.json".format(seed_value,id),encoding="utf-8"))["valid_data"]
            evaluator = Evaluator()
            train_generator = data_generator(train_data, batch_size) 
            model.fit_generator(
             train_generator.forfit(),
             steps_per_epoch=len(train_generator),
             epochs=epochs,
             callbacks=[evaluator]
         )
    elif arg[1]=="cross_test":
        
        from collections import Counter
        data, source_data = test_data(500, 510)
        model,CRF = build_model()
        predict0 = []
        predict1 = []
        predict2 = []
        predict3 = []
        predict4 = []
        for j in range(5):
            print("开始第{}个模型预测".format(j))
            model.load_weights('./best_model_{}_{}.weights'.format(seed_value,j))
            NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
            for i in range(len(data)):
                cur_text = 0
                tmp = []
                for text in data[i]:
                    
                     _,r = NER.recognize(text)
                    
                     for recognize,l,start,end in r:
                         tmp.append((l,cur_text+start,cur_text+end,recognize))
                     cur_text += len(text)
                                
                if j==0:
                    predict0.append(tmp)
                if j==1:
                    predict1.append(tmp)
                if j==2:
                    predict2.append(tmp)
                if j==3:
                    predict3.append(tmp)
                if j==4:
                    predict4.append(tmp)
        pre0,pre1,pre2,pre3,pre4 = [],[],[],[],[]
        for j in range(5,10):
            print("开始第{}个模型预测".format(j))
            model.load_weights('./best_model_{}_{}.weights'.format(1213,j))
            NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
            for i in range(len(data)):
                 cur_text = 0
                 tmp = []
                 for text in data[i]:
 
                      _,r = NER.recognize(text)
 
                      for recognize,l,start,end in r:
                          tmp.append((l,cur_text+start,cur_text+end,recognize))
                      cur_text += len(text)
                 if j==5:
                     pre0.append(tmp)
                 if j==6:
                     pre1.append(tmp)
                 if j==7:
                     pre2.append(tmp)
                 if j==8:
                     pre3.append(tmp)
                 if j==9:
                     pre4.append(tmp)

        for index in range(len(data)):
            tmps = predict0[index]+predict1[index]+predict2[index]+predict3[index]+predict4[index]+pre0[index]+pre1[index]+pre2[index]+pre3[index]+pre4[index] 
            tmps = dict(Counter(tmps))
            tmps = [k for k,v in tmps.items() if v>6]
            with open("/home/wq/ner/test/{}.ann".format(1000 + index), "w", encoding="utf-8") as f:
                  for i in range(len(tmps)):
                      f.write("T{}\t{} {} {}\t{}".format(i + 1, tmps[i][0], tmps[i][1], tmps[i][2], tmps[i][3]))
                      f.write("\n")
                  f.close()
            #K.clear_session()
       

