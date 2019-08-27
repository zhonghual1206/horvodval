# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
#bert微调运行程序

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import horovod.tensorflow as hvd
import metrics
import numpy as np
import random
import time

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
#定义要求变量
#数据目录，用于存放输入数据
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

#bert配置文件，一个json文件，与bert预训练模型的结构参数一致
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

#训练的任务名称
flags.DEFINE_string("task_name", None, "The name of the task to train.")

#BERT模型所训练的词汇表文件
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

#输出目录，checkpoints文件记录在这
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
#初始的checkpoint文件，通常来源于预训练的bert模型
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

#是否需要大小写敏感，设置为True为不敏感
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

#最大序列长度
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

#是否训练
flags.DEFINE_bool("do_train", False, "Whether to run training.")
#是否评估
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
#是否预测
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

#用于训练的批次大小
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
#用于评估的批次大小
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
#测试预测批次大小
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
#adam初始学习率
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
#训练数据轮询次数
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
#热身训练比例
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")
#记录checkpoints的频率
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
#多少步长进行评估
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
#是否使用tpu
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
#tpu名称
tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


#一个简单序列分类的训练/测试示例
class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid        #唯一标识
    self.text_a = text_a    #字符串，第一个未转向量化文本，若只有单个文本，只能指定为这个
    self.text_b = text_b    #可选，字符串，第二个未向量化文本。
    self.label = label      #标签，用于训练和评估，不用与预测。


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """
  '''
    伪示例，因此num输入示例是批处理大小的倍数。
当在TPU上运行eval/predict时，我们需要填充示例的数量
是批处理大小的倍数，因为TPU需要一个固定的批处理
大小。另一种选择是删除最后一批，这是不好的，因为这意味着
不会生成整个输出数据。
我们使用这个类而不是“None”，因为将“None”视为填充
可能会造成一些未知的错误。
  '''

#数据的单一特征集。
class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


#用于序列分类数据集的数据转换器的基类。
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  #获取训练数据样本
  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  #获取一个开发（评估）数据样本
  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  #获取一个测试（预测）数据样本
  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  #获取标签列表
  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  #读取文件
  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


#################################
#我们参考下面例子自写的类，请参考下面类为准

class DealProcessor (DataProcessor):
  def __init__(self):
    def _get_deal_param_split_data(data_dir):
      import pandas as pd

      dict_class = {'竣工失败，卡单无法收费，取消不了订单': '0', '撤单': '1', '补卡异常、无法补卡': '2', '资源大配置卡单': '3', '释放端口': '4', '超时免考核': '5', '提速无法竣工': '6', '加装副卡、宽带和IPTV拦截问题': '7', '更改促销人': '8', '处理黑名单': '9', '无法办理业务、竣工异常': '10', '调用集团接口失败': '11', '竣工异常，取消订单异常': '12', '拆机不竣工': '13', '无法撤单': '14', '办理套餐未竣工问题': '15', '业务规则拦截': '16', '补换卡异常': '17', '卡单不竣工': '18', '资源端口': '19', '销串问题': '20', '取消订单问题': '21', '资源侧号码状态': '22', '需停机问题': '23', '换卡，取消订单': '24', '补录优惠问题': '25', '改生效时间': '26', '拆机、收费拦截截图': '27', '无法过单问题': '28', '提取OLT业务号码': '29', '基站清查问题': '30', '光路数据恢复': '31', '修改工号密码': '32', '修改维护状态': '33', '工号外呼': '34', '施工平台卡单': '35', '取消订单卡单拦截': '36', '关联加装关系': '37', '卡单': '38', '号码实际使用人问题': '39', '无法取消工单，无法竣工，无法撤单': '40', '无法过收费': '41', '承诺在网无法拆机': '42', '设备模块状态': '43', '优惠数据异常': '44', '身份证失效': '45', '客户资料问题': '46', '办理移机拦截': '47', '业务办理拦截': '48', '资源预判，资源环节施工': '49', '证件头像显示模糊问题': '50', '拆机拦截': '51', 'UIM卡类问题': '52', '新装号码卡单、未竣工': '53', '刷新设备端口': '54', '资料接口卡单': '55', '代理商缴费余额不足': '56', '无法改号过收费': '57', 'LOID更新': '58', '竣工拦截问题': '59', 'LOID数据同步': '60', '资源拦截': '61', '无法办理过户': '62', '主卡无法竣工': '63', '拆机未竣工': '64', '受理拦截': '65', '告警': '66', '优惠拦截、取消订单拦截、办理拦截': '67', '办理改号拦截': '68', '第三方收费撤单问题': '69', '大资源配置环节卡单': '70', '修改起租时间': '71', '释放号码问题': '72', '打印发票问题': '73', '智慧助手撤单': '74', '新开号码取消订单、竣工咨询': '75', '竣工异常': '76', '修改接入方式': '77', '账号密码问题': '78', '加装宽带电视出现拦截': '79', '在线支付失败': '80', '竣工失败': '81'}
      label_texts = []
      df=pd.read_excel((os.path.join(data_dir, "dealresult.xls")), sheet_name='Sheet1')
      list1 = df['工单分类'].values
      list2 = df['msg'].values
      print(len(list1), len(list2))
      for label, text in zip(list1, list2):
          try:
              #label_texts.append([dict_class[str(label)], str(text).replace("\r","")])
              label_texts.append([dict_class[str(label)], str(text)])
          except:
              print(label, text)
      print("======>label_texts",label_texts)
      random.shuffle(label_texts)
      return label_texts
    self.data_dir="./data_dir/"
    self.lines = _get_deal_param_split_data(self.data_dir)
    print("========>lines",self.lines)
  def get_train_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      # lines = self._get_deal_param_split_data(data_dir, "train.xls")
      lines=self.lines
      for (i, line) in enumerate(lines[0:int(0.8*len(lines))]):
      # for (i, line) in enumerate(lines):
      #   print(line)
      #   if len(line)<2:
      #       continue
        guid = "train-%d" % (i)
        print("seh train",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid,text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print("======>zhle",e)
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      # lines = self._get_deal_param_split_data(data_dir, "dev.xls")
      #print(lines)
      lines=self.lines
      for (i, line) in enumerate(lines[int(0.8*len(lines)):int(0.9*len(lines))]):
      # for (i, line) in enumerate(lines):
      #   if len(line)<2:
      #       continue
        guid = "dev-%d" % (i)
        print("seh dev",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        examples.append(
            InputExample(guid=guid,text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print(e)
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      # lines = self._get_deal_param_split_data(data_dir, "test.xls")
      #print(lines)
      lines = self.lines
      #for (i, line) in enumerate(lines[int(0.9*len(lines)):-1]):
      for (i, line) in enumerate(lines):
        print(line)
        if len(line)<2:
            continue
        guid = "test-%d" % (i)
        print("seh testzhl",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid,text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print("=======>e",e)
    return examples

  def get_labels(self):
    """See base class."""
    return ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81"]
  '''
  def _get_deal_param_split_data(self, data_dir, file_name="dealresult.xls"):
    import pandas as pd
    dict_class = {'竣工失败，卡单无法收费，取消不了订单': '0', '撤单': '1', '补卡异常、无法补卡': '2', '资源大配置卡单': '3', '释放端口': '4', '超时免考核': '5', '提速无法竣工': '6', '加装副卡、宽带和IPTV拦截问题': '7', '更改促销人': '8', '处理黑名单': '9', '无法办理业务、竣工异常': '10', '调用集团接口失败': '11', '竣工异常，取消订单异常': '12', '拆机不竣工': '13', '无法撤单': '14', '办理套餐未竣工问题': '15', '业务规则拦截': '16', '补换卡异常': '17', '卡单不竣工': '18', '资源端口': '19', '销串问题': '20', '取消订单问题': '21', '资源侧号码状态': '22', '需停机问题': '23', '换卡，取消订单': '24', '补录优惠问题': '25', '改生效时间': '26', '拆机、收费拦截截图': '27', '无法过单问题': '28', '提取OLT业务号码': '29', '基站清查问题': '30', '光路数据恢复': '31', '修改工号密码': '32', '修改维护状态': '33', '工号外呼': '34', '施工平台卡单': '35', '取消订单卡单拦截': '36', '关联加装关系': '37', '卡单': '38', '号码实际使用人问题': '39', '无法取消工单，无法竣工，无法撤单': '40', '无法过收费': '41', '承诺在网无法拆机': '42', '设备模块状态': '43', '优惠数据异常': '44', '身份证失效': '45', '客户资料问题': '46', '办理移机拦截': '47', '业务办理拦截': '48', '资源预判，资源环节施工': '49', '证件头像显示模糊问题': '50', '拆机拦截': '51', 'UIM卡类问题': '52', '新装号码卡单、未竣工': '53', '刷新设备端口': '54', '资料接口卡单': '55', '代理商缴费余额不足': '56', '无法改号过收费': '57', 'LOID更新': '58', '竣工拦截问题': '59', 'LOID数据同步': '60', '资源拦截': '61', '无法办理过户': '62', '主卡无法竣工': '63', '拆机未竣工': '64', '受理拦截': '65', '告警': '66', '优惠拦截、取消订单拦截、办理拦截': '67', '办理改号拦截': '68', '第三方收费撤单问题': '69', '大资源配置环节卡单': '70', '修改起租时间': '71', '释放号码问题': '72', '打印发票问题': '73', '智慧助手撤单': '74', '新开号码取消订单、竣工咨询': '75', '竣工异常': '76', '修改接入方式': '77', '账号密码问题': '78', '加装宽带电视出现拦截': '79', '在线支付失败': '80', '竣工失败': '81'}

    label_texts = []
    df=pd.read_excel((os.path.join(data_dir, file_name)), sheet_name='Sheet1')
    list1 = df['label'].values
    list2 = df['deal_msg'].values
    print(len(list1), len(list2))
    for label, text in zip(list1, list2):
        try:  
            #label_texts.append([dict_class[str(label)], str(text).replace("\r","")])
            label_texts.append([dict_class[str(label)], str(text)])
        except:
            print(label, text)


    print(label_texts)
    random.shuffle(label_texts)
    return label_texts
    '''
class MoneyProcessor (DataProcessor):

  def get_train_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      lines = self._get_money_param_split_data(data_dir)
      for (i, line) in enumerate(lines[0:int(0.8*len(lines))]):
        print(line)
        if len(line)<2:
            continue
        guid = "train-%d" % (i)
        print("seh train",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print(e)
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      lines = self._get_money_param_split_data(data_dir)
      #print(lines)
      for (i, line) in enumerate(lines[int(0.8*len(lines)):int(0.9*len(lines))]):
        if len(line)<2:
            continue
        guid = "dev-%d" % (i)
        print("seh dev",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print(e)
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      #lines = self._read_tsv(os.path.join('/home/seh/bert-master/moneytest', "test.txt"))
      lines = self._get_money_param_split_data(data_dir)
      #print(lines)
      for (i, line) in enumerate(lines[int(0.9*len(lines)):-1]):
      #for (i, line) in enumerate(lines):
        print(line)
        if len(line)<2:
            continue
        guid = "test-%d" % (i)
        print("seh test",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print(e)
    return examples

  def get_labels(self):
    """See base class."""
    return ["0","1","2"]

  def _get_money_param_split_data(self, data_dir):
    import pandas as pd

    label_texts = []
    df=pd.read_excel((os.path.join(data_dir, "财务金额段落筛选-打标3.xlsx")), sheet_name='拼接2')
    list1 = df['金额段落'].values
    list2 = df['标签'].values
    print(len(list1), len(list2))
    for label, text in zip(list2, list1):
        try:  
            label_texts.append([str(label), str(text)])
        except:
            print(label, text)


    print(label_texts)
    random.shuffle(label_texts)
    return label_texts
 

class ChatProcessor (DataProcessor):

  def get_train_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      lines = self._get_chat_split_data(data_dir)
      for (i, line) in enumerate(lines[0:int(0.8*len(lines))]):
        print(line)
        if len(line)<2:
            continue
        if i == 0:
          continue
        guid = "train-%d" % (i)
        print("seh",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print(e)
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      lines = self._get_chat_split_data(data_dir)
      #print(lines)
      for (i, line) in enumerate(lines[int(0.8*len(lines)):int(0.9*len(lines))]):
        if len(line)<2:
            continue
        if i == 0:
          continue
        guid = "dev-%d" % (i)
        print("seh",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print(e)
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      lines = self._get_chat_split_data(data_dir)
      #print(lines)
      for (i, line) in enumerate(lines[int(0.9*len(lines)):-1]):
        print(line)
        if len(line)<2:
            continue
        if i == 0:
          continue
        guid = "test-%d" % (i)
        print("seh",line)
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = None
        label = tokenization.convert_to_unicode(line[0])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print(e)
    return examples

  def get_labels(self):
    """See base class."""
    return ["0","1","2"]

  def _get_chat_split_data(self, data_dir):
    label_texts = []
    label0,label1,label2=[],[],[]
    qa = self._read_tsv(os.path.join(data_dir, "qa.txt"))
    for each in qa:
      try:
        label0.append(each[0])
      except:
        print("error:", each)
    question = self._read_tsv(os.path.join(data_dir, "questions.txt"))
    for each in question:
      try:
        label0.append(each[0])
      except:
        print("error:", each)
    for each in list(set(label0)):
      try:
        label_texts.append(['0', each])
      except:
        print("error:", each)

    professional = self._read_tsv(os.path.join(data_dir, "professional_Q.txt"))
    for each in professional:
      try:
        label1.append(each[0])
      except:
        print("error:", each)
    lines = self._read_tsv(os.path.join(data_dir, "智慧客服问题分类.txt"))
    for each in lines:
      try:
        label1.append(each[1])
      except:
        print("error:", each)
    for each in list(set(label1)):
      label_texts.append(['1', each])

    other = self._read_tsv(os.path.join(data_dir, "others.txt"))
    for each in other:
      try:
        label2.append(each[0])
      except:
        print("error:", each)
    for each in list(set(label2)):
      try:
        label_texts.append(['2', each])
      except:
        print("error:", each)

    print(label_texts)
    random.shuffle(label_texts)
    return label_texts




class SimProcessor (DataProcessor):
  def get_train_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      lines = self._read_tsv(os.path.join(data_dir, "run_classfier数据集.txt"))
      lines2 = self._read_tsv(os.path.join(data_dir, "run_classfier数据集2.txt"))
      lines.extend(lines2)
      random.shuffle(lines)
      #print(lines)
      for (i, line) in enumerate(lines):
        print(line)
        if len(line)<2:
            continue
        if i == 0:
          continue
        guid = "train-%d" % (i)
        #if line[0] in [ '计划建设',  '采购辅助']:
        print("seh",line)
        text_a = tokenization.convert_to_unicode(line[0])
        text_b = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[2])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except Exception as e:
      print(e)
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      lines = self._read_tsv(os.path.join(data_dir, "run_classfier数据集.txt"))
      lines2 = self._read_tsv(os.path.join(data_dir, "run_classfier数据集2.txt"))
      lines = lines.extend(lines2)
      random.shuffle(lines)
      print(lines)
      for (i, line) in enumerate(lines):
        #print(line)
        if len(line)<2:
            continue
        if i == 0:
          continue
        guid = "dev-%d" % (i)
        #if line[0] in [ '计划建设',  '采购辅助']:
        print("seh",line)
        text_a = tokenization.convert_to_unicode(line[0])
        text_b = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[2])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except:
      pass
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    examples = []
    try:
      tf.logging.info('##### %s'%(data_dir))
      lines = self._read_tsv(os.path.join(data_dir, "run_classfier数据集.txt"))
      lines2 = self._read_tsv(os.path.join(data_dir, "run_classfier数据集2.txt"))
      lines = lines.extend(lines2)
      random.shuffle(lines)
      print(lines)
      for (i, line) in enumerate(lines):
        #print(line)
        if len(line)<2:
            continue
        if i == 0:
          continue
        guid = "test-%d" % (i)
        #if line[0] in [ '计划建设',  '采购辅助']:
        print("seh",line)
        text_a = tokenization.convert_to_unicode(line[0])
        text_b = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[2])
        # label = tokenization.convert_to_unicode(str(line[0]))
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    except:
      pass
    return examples
  def get_labels(self):
    """See base class."""
    #label_dict = {'人力辅助': '0', 'FICO': '1', '外部门户': '2', '计划建设': '3', '采购辅助': '4', '合同辅助': '5', '物流': '6', 'MM': '7', '法律辅助': '8', '财务辅助': '9'}
    #return label_dict.keys()
    #return ["3", "4"]
    #return ["0", "1", "2","5", "6", "7", "8", "9"]
    return ["0","1"]

class FfcsProcessor(DataProcessor):
  """Processor for the ffcs contract data set."""
 

  def get_train_examples(self, data_dir):
    """See base class."""
    #lines = self._read_tsv('C:\Users\xpcyc\Desktop\rule_result.txt')'
    label_dict = {'人力辅助': '0', 'FICO': '1', '外部门户': '2', '计划建设': '3', '采购辅助': '4', '合同辅助': '5', '物流': '6', 'MM': '7', '法律辅助': '8', '财务辅助': '9'}
    tf.logging.info('##### %s'%(data_dir))
    lines = self._read_tsv(os.path.join(data_dir, "智慧客服问题分类.txt"))
    random.shuffle(lines)
    examples = []
    #print(lines)
    for (i, line) in enumerate(lines):
      #print(line)
      if len(line)<2:
          continue
      if i == 0:
        continue
      guid = "train-%d" % (i)
      #if line[0] in [ '计划建设',  '采购辅助']:
      print("seh",line)
      text_a = tokenization.convert_to_unicode(line[1])
      #text_b = tokenization.convert_to_unicode(line[1])
      text_b=None
      label = tokenization.convert_to_unicode(str(label_dict[line[0]]))
      # label = tokenization.convert_to_unicode(str(line[0]))
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    #lines = self._read_tsv(os.path.join(data_dir, "ffcs_eval.txt"))
    label_dict = {'人力辅助': '0', 'FICO': '1', '外部门户': '2', '计划建设': '3', '采购辅助': '4', '合同辅助': '5', '物流': '6', 'MM': '7', '法律辅助': '8', '财务辅助': '9'}
    lines = self._read_tsv(os.path.join(data_dir, "智慧客服问题分类.txt"))
    random.shuffle(lines)
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      #if line[0] in [ '计划建设',  '采购辅助']:
      print("seh",line)
      text_a = tokenization.convert_to_unicode(line[1])
      #text_b = tokenization.convert_to_unicode(line[1])
      text_b=None
      label = tokenization.convert_to_unicode(str(label_dict[line[0]]))
          # label = tokenization.convert_to_unicode(str(line[0]))
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      #if line[0] in ['FICO', '人力辅助', '外部门户', 'MM', '物流', '财务辅助', '合同辅助', '法律辅助']:
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    #lines = self._read_tsv(os.path.join(data_dir, "ffcs_pred.txt"))
    label_dict = {'人力辅助': '0', 'FICO': '1', '外部门户': '2', '计划建设': '3', '采购辅助': '4', '合同辅助': '5', '物流': '6', 'MM': '7', '法律辅助': '8', '财务辅助': '9'}
    lines = self._read_tsv(os.path.join(data_dir, "智慧客服问题分类.txt"))
    random.shuffle(lines)
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "test-%d" % (i)
      #pif line[0] in ['FICO', '人力辅助', '外部门户', 'MM', '物流', '财务辅助', '合同辅助', '法律辅助']:
      #if line[0] in [ '计划建设',  '采购辅助']:
      print("seh",line)
      text_a = tokenization.convert_to_unicode(line[1])
      #text_b = tokenization.convert_to_unicode(line[1])
      text_b=None
      label = tokenization.convert_to_unicode(str(label_dict[line[0]]))
          # label = tokenization.convert_to_unicode(str(line[0]))
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    #label_dict = {'人力辅助': '0', 'FICO': '1', '外部门户': '2', '计划建设': '3', '采购辅助': '4', '合同辅助': '5', '物流': '6', 'MM': '7', '法律辅助': '8', '财务辅助': '9'}
    #return label_dict.keys()
    #return ["3", "4"]
    #return ["0", "1", "2","5", "6", "7", "8", "9"]
    return ["0","1","2","3","4","5","6","7","8","9"]
#####################

#XNLI数据集处理类
class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"    #这个例子只取中文数据进行处理

  #获取训练数据
  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)       #标号
      #转码确保为UTF-8
      text_a = tokenization.convert_to_unicode(line[0]) 
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  #获取开发集
  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  #获取标签
  def get_labels(self):
    """See base class."""
    #矛盾，蕴涵，中性
    return ["contradiction", "entailment", "neutral"]


#MultiNLI数据集处理
#该数据集与上个例子类似
class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

#MRPC数据集处理
class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""
  #获取训练集
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
  #获开发练集
  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
  #获取测试集
  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
  #获取标签
  def get_labels(self):
    """See base class."""
    return ["0", "1"]
  #创建训练和开发样本
  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

#Cola數據集處理
class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

#将例子转为特征
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  
  if isinstance(example, PaddingInputExample):
    #为空（PaddingInputExample），返回一个0为基准的空内容
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  #对标签编号
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  #将字符串标记化
  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    
    #截断长度为max_seq_length-3
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      #截断长度为max_seq_length-2
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  
  #A句每个词打标为0，B句每个词打标为1
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  #转数字标号
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  #将有数据为标志为1
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  #剩余位置标准打0
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  #标签转数字
  label_id = label_map[example.label]
  if ex_index < 800:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  #特征赋值
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature

#将输入样本集转化为一个TFRecord文件
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    #将原始数据进行特征转换
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
    
    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    #转化成tensorflow特征
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    #写入文件
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn

#将序列对就地截断到最大长度。
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  #这是一个简单的启发式算法，它总是截断较长的序列
  #一次截断一位。这比截断一个相等的百分比更有意义
  #因为如果一个序列非常短，那么每个符号
  #被截断的序列可能包含比较长的序列更多的信息。

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  
  #简单说就是将两个text之和截短到max_length
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


#创建一个分类模型
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  #构建图层
  #获取预训练模型的输出层
  output_layer = model.get_pooled_output()
  
  hidden_size = output_layer.shape[-1].value
  #设置输出权值
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
  #设置偏移量
  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    #输出output_layer矩阵乘输出权值
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    #加上偏移量
    logits = tf.nn.bias_add(logits, output_bias)
    #使用softmax转分类
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


#根据设定的参数+原预训练checkpoint，初始化一个原始模型
#返回一个TPU评估模型
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #创建一个分类模型
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    #获取初始的
    if init_checkpoint:
      #取当前变量和初始checkpoint变量中的并集
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        #使用从给定init_checkpoint加载的张量初始化当前变量
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization_hvd.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        conf_mat = metrics.get_metrics_ops(label_ids, predictions, num_labels)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_cm": conf_mat,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

#主程序
def main(_):
  hvd.init()
  FLAGS.output_dir = FLAGS.output_dir if hvd.rank() == 0 else os.path.join(FLAGS.output_dir, str(hvd.rank()))

  tf.logging.set_verbosity(tf.logging.INFO)

  #支持的数据处理单元，及其映射类名
  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "ffcs": FfcsProcessor,
      "sim": SimProcessor,
      "chat": ChatProcessor,
      "money": MoneyProcessor,
      "deal": DealProcessor,
  }

  #检查模型名称与do_lower_case配置是否符合
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  #检查是否有do_train/do_eval/do_predict至少一个
  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  #读取bert配置
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  #最大序列长度不可超过bert的最大嵌入位
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  #创建输出目录
  tf.gfile.MakeDirs(FLAGS.output_dir)

  #任务名称小写
  task_name = FLAGS.task_name.lower()

  #查看数据集处理单元是否存在
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  #指定数据集处理类
  processor = processors[task_name]()

  #获取标签列表
  label_list = processor.get_labels()

  #载入词典
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  
  #TPU设置
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = str(hvd.local_rank())

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host),
      log_step_count_steps=25,
      session_config=config)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  print("zhl trainbegining dealdata start ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
  if FLAGS.do_train:
    #获取训练样本
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    print("num of train_examples=",len(train_examples),",data_dir=",FLAGS.data_dir)
    #计算训练的次数
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    #计算预热训练的次数
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
	num_train_steps = num_train_steps // hvd.size()
	num_warmup_steps = num_warmup_steps // hvd.size()
	
  print('-------'*4)
  print(label_list)
  print(FLAGS.init_checkpoint)
  print('learning_rate: ',FLAGS.learning_rate)
  print('use_tpu: ',FLAGS.use_tpu)
  print('master: ',FLAGS.master)
  print('do_lower_case: ',FLAGS.do_lower_case)
  print('warmup_proportion: ',FLAGS.warmup_proportion)
  print('-------'*4)

  #构建一个评估模型
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  #TPU无效，则回退到普通的(CPU/GPU)评估模型
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  #训练
  if FLAGS.do_train:  
    #训练数据文件路径
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    #将数据转为训练用的特征数据
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    #创建一个要传递给TPUEstimator的输入包
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    #训练
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps, hooks=hooks)

  print("zhl trainend time ",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
  #评估
  if FLAGS.do_eval:
    #获取评估数据
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    #将数据转为训练用的特征数据
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    #将数据转为特征数据
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    #创建一个要传递给TPUEstimator的输入包
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)
    
    #评估
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      # 我们可以拿到混淆矩阵（现在时numpy的形式），调用metrics.py文件中的方法来得到precision，recall，f1值
      pre, rec, f1 = metrics.get_metrics(result["eval_cm"], len(label_list))
      tf.logging.info("eval_precision: {}".format(pre))
      tf.logging.info("eval_recall: {}".format(rec))
      tf.logging.info("eval_f1: {}".format(f1))
      tf.logging.info("eval_accuracy: {}".format(result["eval_accuracy"]))
      tf.logging.info("eval_loss: {}".format(result["eval_loss"]))

      np.save("conf_mat.npy", result["eval_cm"])
#      for key in sorted(result.keys()):
#        tf.logging.info("  %s = %s", key, str(result[key]))
#        writer.write("%s = %s\n" % (key, str(result[key])))

  #预测
  if FLAGS.do_predict:
    #获取预测数据
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())
    #数据转特征
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    #预测
    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  #设置必须的参数名称，如果没设置，程序输出异常
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  #运行主程序main()
  tf.app.run()
