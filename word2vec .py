"""Basic word2vec example."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import collections
import math
import os
import random
import zipfile
 
import numpy as np
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf
import pdb
# Step 1: 下载数据.
url = 'http://mattmahoney.net/dc/'
 
 
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename
 
#如果已经下载好了数据，忽略第一步
# filename = maybe_download('text8.zip', 31344016)   
 
filename='text8.zip'
# 读取数据生成词表
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
 
vocabulary = read_data(filename)
print ("######",vocabulary[0],vocabulary[1],"#########")
print('Data size', len(vocabulary))
 
# Step 2: 建立词库
vocabulary_size = 50000 
def build_dataset(words, n_words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))#只取频率最高的前50000个词，其他低频词为UNK
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)    
  data = list()                            
  unk_count = 0                           
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
 
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
#dictionary字典类型，存储单词和出现位置，即word-index,而reverse_dictionary是index-word
#count是列表类型，存储的是单词和出现次数，其中第一个是UNK
#data存储的是有效词的位置

del vocabulary  # 清空vocabulary所占内存
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])  #reverse_dictionary就是为了通过索引直接找到单词
 
data_index = 0
 
# Step 3: 构建skip-gram模型每次训练需要的batch和label.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0           #num_skips是对于一个中心词，随机抽取上下文单词的数量，也就是样本数量
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  
  buffer = collections.deque(maxlen=span)   #构建一个双端队列，储存span个数据
  if data_index + span > len(data):  #依次取span个单词，如果最后单词数不够，则从初始位置开始
    data_index = 0
  buffer.extend(data[data_index:data_index + span])  #一条数据，即中心词和上下文，后面利用这条数据构造num_skips个样本
  data_index += span
  for i in range(batch_size // num_skips):  #“中心词”的数量
    target = skip_window  # 为了中心词不与自身构成样本，skip_window是中心词的位置
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)  #在span长度中随机选取单词
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]   #batch里存储的是index值，代表中心词，即输入
      labels[i * num_skips + j, 0] = buffer[target]    #label里也存储index值，代表上下文，即输出，因为是上下文多个单词，所以labels是两维的
    if data_index == len(data):
      buffer[:] = data[:span]
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # 防止跳过一些单词，因为data_index有被初始化为0的操作
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels
 
# batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
# for i in range(8):
#   print(batch[i], reverse_dictionary[batch[i]],
#         '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
 
# Step 4: 建立skip-gram模型.
 
batch_size = 128
embedding_size = 128  # 词向量维度.
skip_window = 1       # 窗口大小.
num_skips = 2         # 一个中心词产生的样本数
 

#验证集参数
valid_size = 16     
valid_window = 100  
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
print ("#########",valid_examples,"##########")
num_sampled = 64    # 负样本数量.
 
graph = tf.Graph()
 
with graph.as_default():
 
  # 输入数据.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
 
  with tf.device('/cpu:0'):
    # Look up操作，相当于从从词表中选出输入词向量.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
 
    # 使用NCE损失，注：负采样NEG是NCE的简化版本
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
 
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))
 
  # 使用SGD优化.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
 
  # 计算验证集中词语的相似度用于展示
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
 
  # 初始化器
  init = tf.global_variables_initializer()
 
# Step 5: 训练模型.
num_steps = 100001
 
with tf.Session(graph=graph) as session:
  #初始化
  init.run()
  print('Initialized')
 
  average_loss = 0
  for step in xrange(num_steps):#训练轮数
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
 
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val
 
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # 每2000个batch计算一次平均损失.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0
 	#计算与验证集单词相近的词，进行展示
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # 取最接近的单词数量
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()
 
# Step 6: 可视化.
 
 
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
 
  plt.savefig(filename)
 
try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
 
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)
 
except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
