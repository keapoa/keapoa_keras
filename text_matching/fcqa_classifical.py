#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320
#..
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import *
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score
import tensorflow as tf
from keras.callbacks import Callback
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 10
learning_rate = 1e-5
bert_layers = 12
units = 200
"""
# bert配置
config_path = '/home/wq/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/wq/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/wq/chinese_L-12_H-768_A-12/vocab.txt'
"""

#Robert配置
config_path = '/home/wq/ner/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/wq/ner/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/wq/ner/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
"""
#Roberta-large配置

config_path = '/home/wq/ner/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '/home/wq/ner/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '/home/wq/ner/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'
"""

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def load_data(data):
    D = []
    for i in range(data.shape[0]):
        text1, text2, label = str(data.iloc[i,1]),str(data.iloc[i,3]),data.iloc[i,4]
        D.append((text1, text2, int(label)))
    return D
def load_test(data):
    D = []
    for i in range(data.shape[0]):
        text1, text2= str(data.iloc[i, 1]), str(data.iloc[i, 3])
        token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
        D.append([np.array([token_ids]),np.array([segment_ids])])
    return D

# 加载数据集
train = pd.read_csv("/home/wq/fcqa/train_data_kold10.csv")
valid = pd.read_csv("/home/wq/fcqa/valid_data_kold10.csv")
print("原始训练集数据量：{}".format(train.shape[0]))
print("原始验证集的数据量：{}".format(valid.shape[0]))
#这里构建新的训练集数据集合呢
new_train = pd.read_csv("/home/wq/fcqa/new_train.csv")
print("新增后的训练集量为：{}".format(new_train.shape[0]))
#如果重新切分的话，分割下面的数据

all_data = pd.concat([new_train,valid])
all_data = all_data.sample(frac = 1.0,random_state=2020)
all_data = all_data.reset_index(drop=True)
n = int(all_data.shape[0]*0.8)
train = all_data.iloc[:n,:]
valid = all_data.iloc[n:,:]
print("新的总的数据量为：{}".format(all_data.shape[0]))
print("新的训练集的数据量为：{}".format(train.shape[0]))
print("新的验证集的数据量为：{}".format(valid.shape[0]))

test = pd.read_csv("/home/wq/fcqa/test_data.csv")
#加载伪标签数据
train_data = load_data(train)
#不重新分割数据
#train_data = load_data(new_train)
valid_data = load_data(valid)
test_data = load_test(test)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
"""
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)
"""
#with tf.device("/gpu:1"):
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
)
con = []
for i in range(1,4):
    output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - i)
    output = bert.get_layer(output_layer).output
    output = Lambda(lambda x:x[:,0,:])(output)
    weight = Dense(1)(output)
    out = Lambda(lambda x:x[0]*x[1])([output,weight])
    con.append(out)
output = Lambda(lambda x :x[0]+x[1]+x[2])(con)
#output = Dropout(rate=0.0)(bert.model.output)

#不加这段代码会报错，cudnn不支持mask
#直接加lstm貌似没效果
"""
output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - i)
output = bert.get_layer(output_layer).output
output = Lambda(lambda x: x, output_shape=lambda s: s)(output)
output = Bidirectional(CuDNNGRU(units, return_sequences=True))(output)
max_pool = GlobalMaxPooling1D()(output)

output_b = Lambda(lambda x: x[:,0,:])(bert.model.output)
output_b = Lambda(lambda x: x, output_shape=lambda s: s)(output_b)
output = concatenate([output_b, max_pool])

con.append(max_pool)
"""
#output = concatenate(con)

output = Dense(units=2, activation='softmax')(output)
model = keras.models.Model(bert.input, output)
#model = multi_gpu_model(model,2)
model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
#test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    f1, p, r = 0., 0.,0.
    y_preds = []
    y_trues = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1).tolist()
        y_true = y_true[:, 0].tolist()
        y_preds+=y_pred
        y_trues+=y_true
    p = precision_score(y_trues,y_preds)
    r = recall_score(y_trues,y_preds)
    f1 =  f1_score(y_trues,y_preds)
    return p,r,f1


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        p,r,f1 = evaluate(valid_generator)
        if f1 > self.best_f1:
            self.best_f1 = f1
            model.save_weights('best_model_fcqa_layer_unlabel.weights')
        else:
            #调整学习速率
            lr = K.get_value(model.optimizer.lr)
            print("第{}epoch的lr为{}".format(epoch,lr))
            K.set_value(model.optimizer.lr, lr * 0.5)
            lr = K.get_value(model.optimizer.lr)
            print("第{}epoch的{}".format(epoch,lr))
        print(
            u'val_p: %.5f, val_r: %.5f, val_f1:%.5f, best_f1: %.5f\n' %
            (p, r, f1,self.best_f1)
        )

def test_evaluate(data):
    result = []
    for x_true in data:
        y_pred = model.predict(x_true).argmax(axis=1).tolist()
        result+=y_pred
    return result
def test_unlabel(data):
    index = []
    labels = []
    for i,x_true in enumerate(data):

        y_pred = model.predict(x_true)
        y_label = y_pred.argmax(axis=1)
        if i == 0:
            print(y_pred)
            print(y_label)
        if y_pred[0][y_label[0]] > 0.99:
            index.append(i)
            labels.extend(y_label)
    return index,labels
if __name__ == '__main__':

    import sys
    arg = sys.argv
    if arg[1]=="train":
        evaluator = Evaluator()

        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=8,
            callbacks=[evaluator])
    elif arg[1]=="test":
        model.load_weights('best_model_fcqa_layer_unlabel.weights')
        result = test_evaluate(test_data)
        assert len(result)==len(test_data)
        print("预测的样本总条数为：{}".format(len(result)))

        test["label"] = result

        #生成提交文件
        test[["question_id","reply_id","label"]].to_csv("./submission_unlabel.tsv",header = None,index = None,sep = "\t")
    else:
        model.load_weights('best_model_fcqa_layer.weights')
        index,labels = test_unlabel(test_data)
        assert len(index)==len(labels)
        print("伪标签的是条数为{}".format(len(index)))
        test_data_unlabel = test.iloc[index,:]
        test_data_unlabel["label"] = labels
        test_data_unlabel.to_csv("./test_data_unlabel.csv",index = False)
        new_train = pd.concat([train,test_data_unlabel])
        print("新的训练集数量{}".format(new_train.shape[0]))
        new_train.to_csv("./new_train.csv",index = False)
