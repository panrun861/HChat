## 实验简介 
基础模型为ChatGLM3-6b，微调方法为LoRA，优化方法为Adamw_torch，
## 实验过程
### 基础模型
ChatGLM3-6b是一款基于TransFormer
### 微调方法
本次模型采用原生LoRA进行微调，其中LoRA可以注入TransForm中embadding，self-attention，feedback中的7个层数，LoRA矩阵的秩为8，缩放系数为16
### 优化器
本次模型采用Adamw_torch进行优化。相对于传统的梯度下降法，adam对梯度的一阶矩估计和二阶矩估计进行综合考虑，计算出更新步长。Adamw_torch通过修正了权重衰减的实现，从而较 Adam 更进一步地提高了性能，在 Adam 中，权重衰减是在计算梯度之前应用的，这会导致次优结果。AdamW 在计算梯度后才应用权重衰减。
本次微调采用了两个数据集，自我认知和心理健康对话，设置的训练参数如下，Adamw优化器的初始学习率为5e-5，训练轮数为50，最大梯度范数为1，最大样本数为100000，计算类型为fp16，截断长度为1024，批处理大小为2，梯度累计为8，验证集比例为0， 学习率调节器为cosine，训练方法为Supervised Fine-Tunin
### 训练
此外本次微调使用的是LLama-Factory平台，是一款开源的低代码大模型训练框架，提供了可视化训练、推理平台，一键配置模型训练。

三 改进措施：
由于一开始对模型进行微调缺乏相应的基础知识，在后续的学习中发现还可以使用rslora和DoRA方法对lora进行改进。
  rslora使用秩稳定缩放的方法，但在本次实验中，LoRA矩阵的秩为8，相对来说不大，也可以使用原生LORA方法进行微调。
  DoRA 将预先训练好的权矩阵分解为幅度向量和方向矩阵，然后将LoRA运用于方向矩阵，可以增强 LoRA 的学习能力和训练稳定性，同时避免了任何额外的推理开销。
