# 实现 FedScale 部署与自定义训练集以及模型

FedScale ([fedscale.ai](http://fedscale.ai/)) 提供高级应用程序接口，用于在各种硬件和软件后端大规模实施 FL 算法、部署和评估这些算法。本篇 guide 基于 FedScale 平台对于 DLRM 算法进行了简单的搭建与部署。


[FedScale 运行流程分析](#FedScale-运行流程分析)

[运行与部署](#运行与部署)

[FedScale 中的 FL 优化策略](#FedScale-中的-FL-优化策略)

# FedScale 运行流程分析

## 流程图

![flowchart.png](https://github.com/RohanYim/FedScale-DLRM-RunGuide/blob/main/flowchart.png)

## 关键文件及函数解析

### **aggregator.py：模拟服务端**

- init_model(): 初始化 model
- event_monitor(): 分发与接收clients事件
- update_weight_aggregation(): 聚合并更新模型参数

### **executor.py：模拟客户端**

- init_data(): 初始化数据并分发给模拟clients
- Train(): 训练模型
- Test(): 测试模型
- event_monitor(): 分发与接收server事件

### **config_parser.py：处理yaml参数输入**

### **torch_cilent.py / terserflow_client.py: 处理具体模型训练步骤（forward，backward）**

### **fllibs.py: 模型数据初始化**

- init_model(): 模型初始化
- init_data(): 数据初始化

# 运行与部署

## 本地配置：

MacOS:

MacBook Pro(M1 chip)

16G RAM

无 MPS 加速

## 配置阶段

### 安装和导入

```bash
# get code
git clone https://github.com/SymbioticLab/FedScale.git

# setup environment
cd FedScale

# Replace ~/.bashrc with ~/.bash_profile for MacOS
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc 
echo alias fedscale=\'bash $FEDSCALE_HOME/fedscale.sh\' >> ~/.bashrc 
conda init bash
. ~/.bashrc

# 配置 fedscale 虚拟环境
# MAC M1 芯片需要运行arm架构配置 environment-arm.yml
conda env create -f environment-arm.yml
conda activate fedscale    # conda deactivate
pip install -e .
```

### 环境配置问题：

**虚拟环境与本地环境的冲突：**

导致 conda 虚拟环境创建时，并没有按照指定版本 python 创建（本例中希望基于 python 3.8 版本创建，其实基于了 python 3.6 版本。）

```bash
(fedscale) haoransong@HaorandeMacBook-Pro docker % which python
alias python="/usr/local/bin/python3.6"
/Users/haoransong/opt/anaconda3/envs/fedscale/bin/python
```

若之前在`~/.bash_profile`设置了 python 路径的 alias，可能会导致虚拟环境中python版本指向错误，尝试：

```bash
# 更改 bash 文件
vim ~/.bash_profile

# 在 bash_profile 中删除 python alias 行
alias python="/usr/local/bin/python3.6"

# 应用新的 bash 文件
source .bash_profile
```

更新虚拟环境的 python 版本

```bash
# 首先激活虚拟环境
conda activate fedscale

# 在虚拟环境下
conda install python=3.8

# 重启虚拟环境
conda deactivate
conda activate fedscale
```

**MacOS M1 Chip Tensorflow 问题**

M1 chip 可能会出现 **["zsh: illegal hardware instruction python" when installing Tensorflow on macbook pro M1 [duplicate]](https://stackoverflow.com/questions/65383338/zsh-illegal-hardware-instruction-python-when-installing-tensorflow-on-macbook), 尝试:**

[Download TF for M1 whl](https://drive.google.com/drive/folders/1oSipZLnoeQB0Awz8U68KYeCPsULy_dQ7)

Run

```bash
pip install ~/Downloads/tensorflow-2.4.1-py3-none-any.whl
```

## 已有数据集及模型训练（例. Femnist 数据集）

### 数据集下载

```bash
# fedscale 提供了数据集下载的 bash 文件
cd ./benchmart/dataset
bash download.sh download [dataset_name]
```

可能会出现的问题：

```bash
download.sh: line 358: wget: command not found
```

`wget` 是一种常用的命令行工具，用于从网上下载文件。系统中找不到 `wget` 命令，尝试：

```bash
brew install wget
# and run the command again
bash download.sh download [dataset_name]
```

### 配置 `config` 文件：

```bash
cd /FedScale/benchmark/configs/femnist

# 修改 conf.yml
```

若本地运行，需要更改 ip 为本地回环地址：`127.0.0.1`

```yaml
# Configuration file of FAR training experiment

# ========== Cluster configuration ========== 
# ip address of the parameter server (need 1 GPU process)
ps_ip: 127.0.0.1
# ip address of each worker:# of available gpus process on each gpu in this node
# Note that if we collocate ps and worker on same GPU, then we need to decrease this number of available processes on that GPU by 1
# E.g., master node has 4 available processes, then 1 for the ps, and worker should be set to: worker:3
worker_ips:
    - 127.0.0.1:[4]
```

### 使用 Jupyter Notebook 运行 Demo

**Jupyter Notebook 中在虚拟环境中运行项目**

```bash
# Install a Conda Plug-In
conda install nb_conda

# Activate Virtual Environment
conda activate fedscale

# Install a Conda Plug-In in Virtual Environment
conda install ipykernel
# if you are in python3, try
pip3 install ipykernel

# Add environment to Jupyter with custom name
python -m ipykernel install --name fedscale
# if Permission denied, try
sudo python -m ipykernel install --name fedscale
```

**Aggregator(server side)：**

```python
import sys, os

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.aggregation.aggregator import Aggregator
Demo_Aggregator = Aggregator(parser.args)
### On CPU
parser.args.use_cuda = "False"
Demo_Aggregator.run()
```

**Executor(Client side)：**

```python
import torch
import logging
import math
from torch.autograd import Variable
import numpy as np

import sys, os

import fedscale.cloud.config_parser as parser
from fedscale.cloud.execution.torch_client import TorchClient
from fedscale.cloud.execution.executor import Executor
### On CPU
parser.args.use_cuda = "False"
Demo_Executor = Executor(parser.args)
Demo_Executor.run()
```

### tersorboard 查看结果：

```bash
tensorboard --logdir=<path_to_log_folder> --port=6007 --bind_all
```

![femnist_train.png](https://github.com/RohanYim/FedScale-DLRM-RunGuide/blob/main/femnist_train.png)

## 已有数据集及模型训练（淘宝点击数据集 & DLRM）

### 数据集下载

下载[淘宝点击数据集](https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom)

### 配置 config 文件：

```bash
mkdir /FedScale/benchmark/configs/taobao

# 新建 conf.yml
```

```bash
# Configuration file of FAR training experiment

# ========== Cluster configuration ==========
# ip address of the parameter server (need 1 GPU process)
ps_ip: 127.0.0.1

# ip address of each worker:# of available gpus process on each gpu in this node
# Note that if we collocate ps and worker on same GPU, then we need to decrease this number of available processes on that GPU by 1
# E.g., master node has 4 available processes, then 1 for the ps, and worker should be set to: worker:3
worker_ips:
    - 127.0.0.1:[2]

exp_path: $FEDSCALE_HOME/fedscale/cloud

# Entry function of executor and aggregator under $exp_path
executor_entry: execution/executor.py

aggregator_entry: aggregation/aggregator.py

auth:
    ssh_user: ""
    ssh_private_key: ~/.ssh/id_rsa

# cmd to run before we can indeed run FAR (in order)
setup_commands:
    - source $HOME/anaconda3/bin/activate fedscale

# ========== Additional job configuration ==========
# Default parameters are specified in config_parser.py, wherein more description of the parameter can be found

job_conf:
    - job_name: taobao # 修改job名称
    - log_path: $FEDSCALE_HOME/benchmark
    - wandb_token: 4221994eb764b3c6244c61a8c6ba5410xxxxxxxx # 新增 wandb api 观测
    - task: recommendation # 修改task名称
    - num_participants: 50
    - data_set: taobao # 修改 data_set
    - data_dir: $FEDSCALE_HOME/benchmark/dataset/data/taobao # 修改 data_set 路径
		# 删除 data_map_file
    - device_conf_file: $FEDSCALE_HOME/benchmark/dataset/data/device_info/client_device_capacity
    - device_avail_file: $FEDSCALE_HOME/benchmark/dataset/data/device_info/client_behave_trace
    - model: dlrm # 修改使用的 model
    - eval_interval: 5
    - rounds: 1000
    - filter_less: 21
    - num_loaders: 2
    - local_steps: 5
    - learning_rate: 0.01
    - batch_size: 256
    - test_bsz: 256
    - use_cuda: False
    - save_checkpoint: False
		# 新增训练参数
    - sparse_feature_number: 200000 200000 200000 200000 200000 200000 200000
    - sparse_feature_dim: 16
    - dense_feature_dim: 10
    - bot_layer_sizes: 122 64 16
    - top_layer_sizes: 512 256 1
    - num_field: 7
```

修改 config_parser.py 承接新的变量

```python
# for dlrm
parser.add_argument("--dense_feature_dim", type=int, default=16)
parser.add_argument("--bot_layer_sizes", type=int, nargs='+', default=[64, 128, 64])
parser.add_argument("--sparse_feature_number", type=int, nargs='+',default=[10000, 10000, 10000])
parser.add_argument("--sparse_feature_dim", type=int, default=16)
parser.add_argument("--top_layer_sizes", type=int, nargs='+', default=[512, 256, 1])
parser.add_argument("--num_field", type=int, default=26)
parser.add_argument("--sync_mode", type=str, default=None)
```

### 处理数据集

预处理

```python
# fllibs.py
def init_dataset():
	...
	from fedscale.dataloaders.dlrm_taobao import Taobao
	import pandas as pd
	def manual_train_test_split(df, test_size=0.2):
	    indices = df.index.tolist()
	    test_indices = random.sample(indices, int(len(indices) * test_size))
	
	    test_df = df.loc[test_indices]
	    train_df = df.drop(test_indices)
	    
	    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
	
	logging.info("Getting taobao dataset...")
	n_rows = 200000
	df_user_profile = pd.read_csv('/Users/haoransong/Downloads/archive/user_profile.csv', nrows=n_rows)
	df_raw_sample = pd.read_csv('/Users/haoransong/Downloads/archive/raw_sample.csv', nrows=n_rows)
	df_ad_feature = pd.read_csv('/Users/haoransong/Downloads/archive/ad_feature.csv', nrows=n_rows)
	df_raw_sample.rename(columns={'user': 'userid'}, inplace=True)
	df_merged = pd.merge(df_raw_sample, df_user_profile, how='left', on='userid')
	df_merged = pd.merge(df_merged, df_ad_feature, how='left', on='adgroup_id')
	df_merged.columns = df_merged.columns.str.strip()
	
	missing_values = df_merged.isna().any()
	columns_with_nan = missing_values[missing_values].index.tolist()
	
	for column in columns_with_nan:
	    mode_value = df_merged[column].mode()[0]
	    df_merged[column].fillna(mode_value, inplace=True)
	
	train_df, test_df = manual_train_test_split(df_merged, test_size=0.2)
	logging.info('Before Taobao')
	train_dataset = Taobao(train_df)
	test_dataset = Taobao(test_df)
	logging.info('Got dataset!')
```

分别处理稠密以及稀疏数据集

```python
# dataloaders/dlrm_taobao.py
class Taobao(Dataset):
    def __init__(self, df):
        self.data_frame = df

        logging.info("init taobao...")

        self.sparse_features = [
            'userid', 'adgroup_id', 'cms_segid', 
            'cate_id', 'campaign_id', 'customer', 'brand'
        ]
        self.index_mappings = {
            feature: {v: k for k, v in enumerate(self.data_frame[feature].unique())} 
            for feature in self.sparse_features
        }

        dense_features = ['final_gender_code', 'pid', 'age_level','cms_group_id', 'shopping_level', 'occupation', 'new_user_class_level', 'time_stamp', 'price', 'pvalue_level']
        self.dense_features = dense_features
        for feature in self.dense_features:
            self.data_frame[feature] = self.data_frame[feature].astype(np.float32)
        
        self.targets = self.data_frame['clk'].values.astype(np.float32)
        logging.info("init taobao done...")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sparse_feature_values = [
            self.index_mappings[feature][self.data_frame.loc[idx, feature]]
            for feature in self.index_mappings
        ]
        sparse_x = np.array(sparse_feature_values, dtype=np.int64)
        dense_x_values = [
            self.data_frame.loc[idx, feature]
            for feature in self.dense_features
        ]
        dense_x = np.array(dense_x_values, dtype=np.float32)
        label = self.targets[idx].astype(np.float32)
    
        return dense_x, sparse_x, label
```

### 自定义 model

```python
# utils/models/recommendation/dlrm.py
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

MIN_FLOAT = torch.finfo(torch.float32).min / 100.0

class DLRM(nn.Module):
    def __init__(self,
                 args,
                 sync_mode=None):
        super(DLRM, self).__init__()
        logging.info('Init DLRM')
        self.dense_feature_dim = args.dense_feature_dim
        self.bot_layer_sizes = args.bot_layer_sizes
        self.sparse_feature_number = args.sparse_feature_number
        self.sparse_feature_dim = args.sparse_feature_dim
        self.top_layer_sizes = args.top_layer_sizes
        self.num_field = args.num_field

        self.bot_mlp = MLPLayer(input_shape=self.dense_feature_dim,
                                units_list=self.bot_layer_sizes,
                                last_action="relu")

        self.top_mlp = MLPLayer(input_shape=int(self.num_field * (self.num_field + 1) / 2) + self.sparse_feature_dim,
                                units_list=self.top_layer_sizes,last_action='sigmoid')
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=self.sparse_feature_dim)
            for size in self.sparse_feature_number
        ])

    def forward(self, sparse_inputs, dense_inputs):
        x = self.bot_mlp(dense_inputs)

        batch_size, d = x.shape
        sparse_embs = [self.embeddings[i](sparse_inputs[:, i]).view(-1, self.sparse_feature_dim) for i in range(sparse_inputs.shape[1])]

        T = torch.cat(sparse_embs + [x], axis=1).view(batch_size, -1, d)

        Z = torch.bmm(T, T.transpose(1, 2))
        Zflat = torch.triu(Z, diagonal=1) + torch.tril(torch.ones_like(Z) * MIN_FLOAT, diagonal=0)
        Zflat = Zflat.masked_select(Zflat > MIN_FLOAT).view(batch_size, -1)

        R = torch.cat([x] + [Zflat], axis=1)

        y = self.top_mlp(R)
        return y

class MLPLayer(nn.Module):
    def __init__(self, input_shape, units_list=None, l2=0.01, last_action=None):
        super(MLPLayer, self).__init__()

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.l2 = l2
        self.last_action = last_action
        self.mlp = nn.Sequential()

        for i in range(len(units_list)-1):
            self.mlp.add_module('dense_%d' % i, nn.Linear(units_list[i], units_list[i + 1]))
            if i != len(units_list) - 2 or last_action is not None:
                self.mlp.add_module('relu_%d' % i, nn.ReLU())
            self.mlp.add_module('norm_%d' % i, nn.BatchNorm1d(units_list[i + 1]))
        if last_action == 'sigmoid':
            self.mlp.add_module('sigmoid', nn.Sigmoid())

    def forward(self, inputs):
        return self.mlp(inputs)
```

修改客户端相应文件承接新模型

```python
# fllibs.py
def import_libs():
	...
	elif parser.args.task == 'recommendation':
        global DLRM
        from fedscale.utils.models.recommendation.dlrm import DLRM

def init_model():
	...
	elif parser.args.task == 'recommendation':
        if parser.args.model == 'dlrm':
            model = DLRM(parser.args)
            logging.info('Got DLRM！')
        else:
            logging.info('Recommendation model does not exist!')
```

```python
# torch_client.py
def get_criterion(self, conf):
    criterion = None
		# new
    elif conf.task == 'recommendation':
        criterion = torch.nn.BCEWithLogitsLoss()
    return criterion

def train_step(self, client_data, conf, model, optimizer, criterion):
    logging.info("start training step....")
    for data_pair in client_data:
        if conf.task == 'nlp':
            ...
				# new
        elif conf.task == 'recommendation':
            dense_x, sparse_x, target = data_pair
       
				...

        if conf.task == "detection":
            ...
        elif conf.task == 'recommendation' and conf.model == 'dlrm':
            logging.info("start dlrm training step....")
            dense_features = dense_x.float()
            sparse_features = sparse_x.long()
            target = target.float().view(-1, 1)  
        ...

        target = Variable(target).to(device=self.device)

        if conf.task == 'nlp':
            ...
        elif conf.task == 'recommendation':
            outputs = model(sparse_features, dense_features)
            loss = criterion(outputs, target)
        ...

        # ======== collect training feedback for other decision components [e.g., oort selector] ======

        if conf.task == 'nlp' or (conf.task == 'text_clf' and conf.model == 'albert-base-v2') or conf.task == 'recommendation':
            loss_list = [loss.item()]  # [loss.mean().data.item()]

        ...
@overrides
def test(self, client_data, model, conf):
    """
    Perform a testing task.
    :param client_data: client evaluation dataset
    :param model: the framework-specific model
    :param conf: job config
    :return: testing results
    """
    evalStart = time.time()
    if self.args.task == 'voice':
        criterion = CTCLoss(reduction='mean').to(device=self.device)
    else:
        logging.info("start testing...")
        criterion = torch.nn.CrossEntropyLoss().to(device=self.device)
   ...
```

```python
# model_test_module.py
elif parser.args.task == 'recommendation':
    logging.info("Testing for dlrm...")
    total_loss = 0.0
    total_examples = 0
    correct_predictions = 0
    for batch in test_data:
        dense_x, sparse_x, labels = batch
        dense_x = dense_x.float()
        sparse_x = sparse_x.long()
        labels = labels.float().view(-1, 1)

        outputs = model(sparse_x, dense_x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)  
        total_examples += labels.size(0)

        predicted_probs = torch.sigmoid(outputs)  
        predicted_labels = (predicted_probs > 0.5).float() 
        correct_predictions += (predicted_labels == labels).sum().item()

    logging.info(f'Test set: Loss: {total_loss:.4f}')
    return correct_predictions,correct_predictions,total_loss,{'top_1': correct_predictions, 'top_5': correct_predictions, 'test_loss': total_loss, 'test_len': total_examples}
```

### 运行

```bash
python driver.py start benchmark/configs/taobao/conf.yml
```

### tensorboard 查看结果

![dlrm_train.png](https://github.com/RohanYim/FedScale-DLRM-RunGuide/blob/main/dlrm_train_result.png)

### 不足及未来工作

1. 模型准确度问题

    本篇只是尝试在本地搭建基于 FedScale 的自定义模型训练，并未考虑训练的结果。后续会考虑云端训练，提高训练效率，完成对模型准确度的优化。

2. DLRM 的优化：
    - 在 [AdaEmbed: Adaptive Embedding for Large-Scale Recommendation Models](https://www.usenix.org/conference/osdi23/presentation/lai) 提到了通过在训练过程中动态修剪嵌入表来优化 (DLRM) 的效率，后续可以通过实现 AdaEmbed 表过大的问题。
    - 另外，本篇的模型中 embedding 表是模型参数的一部分，并跟随其他参数传回给服务器端，这不仅会造成传输空间的大规模占用，更会暴露用户的隐私信息。另外，embedding 表属于用户的个性化信息，并不适用于服务端聚合参数并分发给各个客户端。这些是后续需要关注的问题。

# FedScale 中的 FL 优化策略

## **Oort 采样器**：

Oort 优先选择那些既能快速完成训练又能提供最大模型精度提升的数据的客户端。

`init_client_manager` 方法根据提供的参数（通过 `args.sample_mode`）来决定使用哪一种管理器，并返回初始化的客户端管理器实例。

## 优化器：

### q-fedavg: [Fair Resource Allocation in Federated Learning](https://arxiv.org/pdf/1905.10497.pdf)

考虑到了客户端之间的不公平性，通过引入一个超参数 **`q`** 来控制更新的聚合方式，每个客户端的更新被其损失的幂次加权,以期实现更公平的资源分配.

### **FedYogi: [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf)**

自适应的优化方法，服务器端的优化器会使用客户端计算出的模型更新来调整全局模型。