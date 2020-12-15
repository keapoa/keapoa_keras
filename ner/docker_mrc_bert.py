import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import *
import json
import tensorflow as tf
#from tqdm import tqdm
import sys
import random
import os
seed_value = 2020
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
#os.environ["RECOMPUTE"] = "1"
set_gelu('tanh')  # 切换gelu版本

maxlen = 300
batch_size = 12
learning_rate = 1e-5
bert_layers = 12
#字典

query_sign_map = {
"tags":['DRUG', 'DRUG_INGREDIENT', 'DISEASE',"SYMPTOM","SYNDROME","DISEASE_GROUP","FOOD","FOOD_GROUP","PERSON_GROUP","DRUG_GROUP","DRUG_DOSAGE","DRUG_TASTE","DRUG_EFFICACY"],
    "natural_query":{
        "DRUG":"找出六味地黄丸,逍遥散等中药名称",
        "DRUG_INGREDIENT":"找出当归,人参等中药药物成分",
        "DISEASE":"找出高血压,心绞痛,糖尿病等疾病名称",
        "SYMPTOM":"找出头晕,心悸,小腹胀痛等疾病症状",
        "SYNDROME":"找出血瘀,气滞,气血不足,气血两虚等证候",
        "DISEASE_GROUP":"找出肾病,肝病,肺病等人体组织部位疾病",
        "FOOD":"找出苹果,萝卜,木耳等食物",
        "FOOD_GROUP":"找出油腻食物,辛辣食物,凉性食物等中医药禁用食物",
        "PERSON_GROUP":"找出孕妇,经期妇女,儿童,青春期少女等中医药适用及禁用人群",
        "DRUG_GROUP":"找出止咳药,退烧药等某类药品",
        "DRUG_DOSAGE":"找出浓缩丸,水蜜丸,糖衣片等药物剂型",
        "DRUG_TASTE":"找出味甘,酸涩,气凉等药物性味",
        "DRUG_EFFICACY":"找出滋阴补肾,去瘀生新,活血化瘀等中药功效"}
}

# bert配置
"""
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
"""

#docker镜像路径

config_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

#读取docker测试集数据
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

test_data = []

for i in range(1500,1997):
    txt_file = open("/tcdata/juesai/{}.txt".format(i), encoding="utf-8")
    #txt_file = open("/home/wq/ner/test/{}.txt".format(i), encoding="utf-8")
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

# 加载数据集
#data = json.load(open("./mrc_base_data_seed{}.json".format(seed_value),encoding="utf-8"))
#train_data = data["train"]

#valid_data = data["valid"]
#test_data = data["test"]
#print(test_data[0])
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

#构建问答ner,(query,text,start_labels,end_labels
#其次要把query以及[cls,seq] mask掉。利用query长度构建mask[0]*len(query)+[0]+[1]*len(text)+[0],构建mask
#

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_mask, batch_start_labels,  batch_end_labels= [],[],[],[],[]
        for is_end, (query,text,start_label,end_label,T,label) in self.sample(random):
            """
            token_ids, segment_ids = tokenizer.encode(
                query, text, maxlen=maxlen
            )
            """
            token_ids = []
            segment_ids = []
            token_ids.append(tokenizer.token_to_id("[CLS]"))
            segment_ids.append(0)
            for i in query:
                token_ids.append(tokenizer.token_to_id(i))
                segment_ids.append(0)
            token_ids.append(tokenizer.token_to_id("[SEP]"))
            segment_ids.append(0)
            for i in text:
                token_ids.append(tokenizer.token_to_id(i))
                segment_ids.append(tokenizer.token_to_id(1))
            token_ids.append(tokenizer.token_to_id("[SEP]"))
            segment_ids.append(1)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_mask.append([0]+[0]*len(query)+[0]+[1]*len(text)+[0])

            assert len([0]+[0]*len(query)+[0]+[1]*len(text)+[0])==len(token_ids)

            batch_start_labels.append(start_label)
            batch_end_labels.append(end_label)
            if len(batch_token_ids) == self.batch_size or is_end:

                batch_token_ids = sequence_padding(batch_token_ids)
                batch_mask = sequence_padding(batch_mask)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_start_labels = sequence_padding(batch_start_labels)
                batch_end_labels = sequence_padding(batch_end_labels)

                yield [batch_token_ids, batch_segment_ids, batch_mask,batch_start_labels, batch_end_labels],None
                batch_token_ids, batch_segment_ids, batch_mask, batch_start_labels,batch_end_labels = [], [], [], [], []

def focal_loss(logits,labels,mask,gamma = 2):
    pos_probs = logits[:,:,1]
    prob_label_pos = tf.where(K.equal(labels, 1), pos_probs, tf.ones_like(pos_probs))
    prob_label_neg = tf.where(K.equal(labels, 0), pos_probs, tf.zeros_like(pos_probs))
    loss = tf.pow(1. - prob_label_pos, gamma) * tf.log(prob_label_pos + 1e-7) + \
           tf.pow(prob_label_neg, gamma) * tf.log(1. - prob_label_neg + 1e-7)
    """
    loss = -loss * K.cast(mask, tf.float32)
    loss = tf.reduce_sum(loss, axis=-1, keepdims=True)
    loss = tf.reduce_mean(loss)
    """
    loss = K.sum(-loss*mask)/K.sum(mask)
    return loss
# 加载预训练模型
m = Input(shape = (None,))
start_label = Input(shape = (None,))
end_label = Input(shape = (None,))

bert = build_transformer_model(
            config_path,
            checkpoint_path, )
output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = bert.get_layer(output_layer).output
#output = Dropout(rate=0.1)(bert.model.output)


start_output = Dense(2,activation="softmax", name="start_loss")(output)
end_output = Dense(2,activation="softmax",name="end_loss")(output)

#start_output = Lambda(lambda x:K.permute_dimensions(x,(1,0,2)))(start_output)
#end_output = Lambda(lambda x:K.permute_dimensions(x,(1,0,2)))(end_output)
mask = Lambda(lambda x:K.cast(x,"float32"))(m)
model = keras.models.Model(bert.input+[m], [start_output,end_output])
#model.summary()
total_model = keras.models.Model(bert.input+[m,start_label,end_label], [start_output,end_output])

#print(start_label.shape,start_output.shape)
"""
start_loss = K.sparse_categorical_crossentropy(start_label,start_output)
start_loss = K.sum(start_loss*mask[:,:,0]) / K.sum(mask)

end_loss = K.sparse_categorical_crossentropy(end_label,end_output)
end_loss = K.sum(end_loss*mask[:,:,0]) / K.sum(mask)
"""
start_loss = focal_loss(start_output,start_label,mask)
end_loss = focal_loss(end_output,end_label,mask)


total_loss = start_loss + end_loss

total_model.add_loss(total_loss)
#total_loss = {"start_loss":"sparse_categorical_crossentropy","end_loss":"sparse_categorical_crossentropy"}
total_model.compile(
    optimizer=Adam(learning_rate),  # 用足够小的学习率
)
#total_model.summary()

# 转换数据集
#train_generator = data_generator(train_data, batch_size)
#valid_generator = data_generator(valid_data, 1)
#test_generator = data_generator(test_data, batch_size)

#每个样本保留实体标签索引对（标签，开始索引，结束索引）即可
def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for query,text,start_label,end_label,T,label in data:
        #print(query)
        #print(text)

        token_ids = []
        segment_ids = []
        token_ids.append(tokenizer.token_to_id("[CLS]"))
        segment_ids.append(0)
        for i in query:
            token_ids.append(tokenizer.token_to_id(i))
            segment_ids.append(0)
        token_ids.append(tokenizer.token_to_id("[SEP]"))
        segment_ids.append(0)
        for i in text:
            token_ids.append(tokenizer.token_to_id(i))
            segment_ids.append(tokenizer.token_to_id(1))
        token_ids.append(tokenizer.token_to_id("[SEP]"))
        segment_ids.append(1)
        #print(len(token_ids))
        mask = [0]+[0]*len(query)+[0]+[1]*len(text)+[0]
        start_logits,end_logits = model.predict([np.array([token_ids]),np.array([segment_ids]),np.array(mask)])
        #print(start_logits.shape)
        start_logits = np.argmax(start_logits[0],axis = 1)
        #print(start_logits)
        end_logits = np.argmax(end_logits[0],axis = 1)

        assert len(start_logits)==len(end_logits)
        #print(end_label)
        left_length = len(query)+2
        source_text = "@"+query+"@"+ text+"@"
        #print(len(source_text),len(start_logits))
        #print(start_logits)
        #print(end_logits)
        R = []
        end_index = 0
        for i in range(left_length,len(start_logits)-1):
            if start_logits[i]==1 and i >= end_index:
                for j in range(i,len(start_logits)-1):
                    if end_logits[j]==1:
                        end_index = j+1
                        R.append((i-left_length,j+1-left_length,label))
                        break
        
        R = set(R)
        #print(R)
        #print("*************************")
        #print(T)
        T = set([(s,e,l)for s,e,l in T])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

def test(all_datas):
    #print("测试数据的长度为{}".format(len(all_datas)))
    # 遍历每一条样本
    for num,datas in enumerate(all_datas):

        cur_text_length = 0
        tmps = []
        # 遍历每一条样本中的每一个分割样本
        for data in datas:

            #遍历每一个分割样本中的每一个类别样本
            for d in data:
                query,text,label = d[0],d[1],d[2]

                token_ids = []
                segment_ids = []
                token_ids.append(tokenizer.token_to_id("[CLS]"))
                segment_ids.append(0)
                for i in query:
                    token_ids.append(tokenizer.token_to_id(i))
                    segment_ids.append(0)
                token_ids.append(tokenizer.token_to_id("[SEP]"))
                segment_ids.append(0)
                for i in text:
                    token_ids.append(tokenizer.token_to_id(i))
                    segment_ids.append(tokenizer.token_to_id(1))
                token_ids.append(tokenizer.token_to_id("[SEP]"))
                segment_ids.append(1)

                mask = [0]+[0]*len(query)+[0]+[1]*len(text)+[0]
                start_logits,end_logits = model.predict([np.array([token_ids]),np.array([segment_ids]),np.array([mask])])
                start_logits = np.argmax(start_logits[0],axis = 1)
                end_logits = np.argmax(end_logits[0],axis = 1)
                source_text = "@"+query+"@"+text+"@"
                assert start_logits.shape==end_logits.shape
                left_length = len(query) + 2
                end_index = 0
                for i in range(left_length,len(start_logits)-1):
                    if i >= len(query)+2:
                        if start_logits[i]==1 and i >=end_index:
                            for j in range(i,len(start_logits)-1):
                                if end_logits[j]==1:
                                    end_index = j+1
                                    tmps.append((label,i-left_length+cur_text_length,j+1-left_length+cur_text_length,source_text[i:j+1]))
                                    break
            cur_text_length += len(text)
        with open("./result/{}.ann".format(1500 + num), "w", encoding="utf-8") as f:
            for i in range(len(tmps)):
                f.write("T{}\t{} {} {}\t{}".format(i + 1, tmps[i][0], tmps[i][1], tmps[i][2], tmps[i][3]))
                f.write("\n")
            f.close()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1,p,r = evaluate(valid_data)
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            total_model.save_weights('best_model.weights')
        print(
            u'val_p: %.5f,val_r: %.5f, val_f1: %.5f\n' %
            (p, r, self.best_val_f1)
        )


if __name__ == '__main__':
    import sys
    arg = sys.argv

    if arg[1]=="train":
        evaluator = Evaluator()

        total_model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=12,
            callbacks=[evaluator]
        )

    elif arg[1]=="test":
        #print("**************开始预测*************")
        total_model.load_weights('best_model.weights')
        test(test_data)
        
        #print("数据预测完成，开始压缩")
        # 压缩文件
        import zipfile
        import os
        import sys
        """
        # 定义一个函数，递归读取absDir文件夹中所有文件，并塞进zipFile文件中。参数absDir表示文件夹的绝对路径。
        def writeAllFileToZip(absDir, zipFile):
            for f in os.listdir(absDir):
                absFile = os.path.join(absDir, f)  # 子文件的绝对路径
                if os.path.isdir(absFile):  # 判断是文件夹，继续深度读取。
                    relFile = absFile[len(os.getcwd()) + 1:]  # 改成相对路径，否则解压zip是/User/xxx开头的文件。
                    zipFile.write(relFile)  # 在zip文件中创建文件夹
                    writeAllFileToZip(absFile, zipFile)  # 递归操作
                else:  # 判断是普通文件，直接写到zip文件中。
                    relFile = absFile[len(os.getcwd()) + 1:]
                    #print(relFile)# 改成相对路径
                    zipFile.write(relFile)
            return


        zipFilePath = os.path.join(sys.path[0], "result.zip")
        # 先定义zip文件绝对路径。sys.path[0]获取的是脚本所在绝对目录。
        # 因为zip文件存放在脚本同级目录，所以直接拼接得到zip文件的绝对路径。
        zipFile = zipfile.ZipFile(zipFilePath, "w", zipfile.ZIP_DEFLATED)
        # 创建空的zip文件(ZipFile类型)。参数w表示写模式。zipfile.ZIP_DEFLATE表示需要压缩，文件会变小。ZIP_STORED是单纯的复制，文件大小没变。
        absDir = os.path.join(sys.path[0], "./result")
        # 要压缩的文件夹绝对路径。
        writeAllFileToZip(absDir, zipFile)  # 开始压缩。如果当前工作目录跟脚本所在目录一样，直接运行这个函数。
        # 执行这条压缩命令前，要保证当前工作目录是脚本所在目录(absDir的父级目录)。否则会报找不到文件的错误。
        print("压缩成功")
        """
        z = zipfile.ZipFile('result.zip', 'w', zipfile.ZIP_DEFLATED)
        startdir = "./result"
        for dirpath, dirnames, filenames in os.walk(startdir):
            for filename in filenames:
                z.write(os.path.join(dirpath, filename))
        z.close()
    else:
        print("测试单条样本")
        text = " 每粒装0.4g  口服。经期或经前5天一次3～5粒，一日3次，经后可继续服用，一次3～5粒，一日2～3次。  通调气血，止痛调经。用于经期腹痛及因寒所致的月经失调 广东和平药业有限公司  用于经期腹痛及因寒冷所致的月经失调 尚不明确。  尚不明确。  非处方药物（甲类）,国家医保目录（乙类） "
        query = "找出头晕,心悸,小腹胀痛等疾病症状"
        label = "SYMPTOM"
        total_model.load_weights('best_model.weights')

        token_ids = []
        segment_ids = []
        token_ids.append(tokenizer.token_to_id("[CLS]"))
        segment_ids.append(0)
        for i in query:
            token_ids.append(tokenizer.token_to_id(i))
            segment_ids.append(0)
        token_ids.append(tokenizer.token_to_id("[SEP]"))
        segment_ids.append(0)
        for i in text:
            token_ids.append(tokenizer.token_to_id(i))
            segment_ids.append(tokenizer.token_to_id(1))
        token_ids.append(tokenizer.token_to_id("[SEP]"))
        segment_ids.append(1)

        mask = [0] + [0] * len(query) + [0] + [1] * len(text) + [0]
        start_logits, end_logits = model.predict([np.array([token_ids]), np.array([segment_ids]), np.array([mask])])
        start_logits = np.argmax(start_logits[0], axis=1)
        end_logits = np.argmax(end_logits[0], axis=1)

        print(query)
        print(text)
        print(start_logits)
        print(end_logits)
        cur_text_length = 0
        source_text = "@" + query + "@" + text + "@"
        assert start_logits.shape == end_logits.shape
        left_length = len(query) + 2
        end_index = 0
        tmps = []
        for i in range(left_length, len(start_logits) - 1):
            if i >= len(query) + 2:
                if start_logits[i] == 1 and i >= end_index:
                    for j in range(i, len(start_logits) - 1):
                        if end_logits[j] == 1:
                            end_index = j + 1
                            tmps.append((
                                        label, i - left_length + cur_text_length, j + 1 - left_length + cur_text_length,
                                        source_text[i:j + 1]))
                            break
        print(tmps)



