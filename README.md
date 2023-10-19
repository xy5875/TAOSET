# TAOSET

***

![81d82bdb9e6ce89b0bdd6fbcd2cb6ed](https://github.com/xy5875/TAOSET/assets/63028603/c21fbf83-711c-4259-8bc4-b7c3d39afb9f)


TAOSet工具集能够支持并加速深度学习模型与算法在异构AIoT设备上高效部署。通过用户给定性能需求，结合硬件资源智能感知，TAOSet能够自动选择并组合跨层加速策略，实现情境动态的自适应跨层协同训练或推理任务。

TAOSet核心模块包含单机优化模块、异构集群训练模块与异构集群推理模块。单机优化模块作为自主可控的泛在异构存算一体架构，包括算法层、算子编译层和内存编译层，能够分别从模型部署的不同阶段对物联网深度学习任务进行加速。异构集群模块基于AIoT设备集群的异构系统环境与硬件资源进行控制与调度，实现群智能体增强计算与协同优化。异构集群训练模块包括自适应资源计算和智能化数据通信，异构集群推理模块包括自适应计算卸载与智能化模型分割。通过单机优化模块对异构集群模块的支撑，能够进一步提高集群单个AIoT设备的计算效率。 

## 使用说明：

***

### 1.算法层优化

#### 运行：

`cd sparse_training`

`python train.py --sparse_training --autocast`

#### 参数信息：

**--sparse_training:**稀疏更新功能

**--autocast:**量化功能

### 2.算子编译层优化

#### 运行：

`cd op_fuse`

`python new.py`

#### 参数信息：

**run_iter:**每次测试模型运行的轮数

**model:**所选择的模型，本代码默认为Resnet18

### 3.内存编译层优化

#### 运行——内存分配：

`cd mem_alloc`

`python train.py --network --empty_cache`

#### 参数信息：

**--network:**选择神经网络

**--empty_cache:**是否启用缓存复用功能

#### 运行——重计算：

`cd recomputation`

`python checkpoint_test.py --network --empty_cache`

#### 参数信息：

**input：**可根据不同的模型输入对应大小的张量

**model：**选择要优化的模型

### 4.异构集群分布式训练

#### 运行(AR架构：ar_dist)：

> 其余类似

`cd ar_dist`

`python example_server.py `

> 服务器端

`python example_client.py`

> 客户端1

`python example_client2.py`

> 客户端2

#### 参数信息：

**cluster_conf：**包含**ps**和**workers**的ip地址，修改成对应的即可

## 环境要求：

***

`argparse`

`pytorch`

`grpc`

`pandas`

`numpy`

`pyplot`

`logging`

