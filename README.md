## 实验简介 
基础模型为ChatGLM3-6b，微调方法为LoRA，优化方法为Adamw_torch，
## 实验过程
### 基础模型
ChatGLM3-6b是一款基于TransFormer大型预训练语言模型，它拥有60亿个参数，能够在理解自然语言的基础上，进行文本生成、对话、问答等多种任务。
### 微调方法
本次模型采用原生LoRA进行微调，其中LoRA可以注入TransForm中embadding，self-attention，feedback中的7个层数，LoRA矩阵的秩为8，缩放系数为16
### 优化器
本次模型采用Adamw_torch进行优化。相对于传统的梯度下降法，adam对梯度的一阶矩估计和二阶矩估计进行综合考虑，计算出更新步长。Adamw_torch通过修正了权重衰减的实现，从而较 Adam 更进一步地提高了性能，在 Adam 中，权重衰减是在计算梯度之前应用的，这会导致次优结果。AdamW 在计算梯度后才应用权重衰减。
  
本次微调采用了两个数据集，自我认知和心理健康对话，设置的训练参数如下，Adamw优化器的初始学习率为5e-5，训练轮数为50，最大梯度范数为1，最大样本数为100000，计算类型为fp16，截断长度为1024,批处理大小为2，梯度累计为8，验证集比例为0， 学习率调节器为cosine
### 训练
此外本次微调使用的是LLama-Factory平台，是一款开源的低代码大模型训练框架，提供了可视化训练、推理平台，一键配置模型训练。

页面截图如下:
<image src='images/1.png'>

<image src='images/2.png'>

<image src='images/3.png'>


## 改进措施：
由于一开始对模型进行微调缺乏相应的基础知识，在后续的学习中发现还可以使用rslora和DoRA方法对lora进行改进。
  
rslora使用秩稳定缩放的方法，但在本次实验中，LoRA矩阵的秩为8，相对来说不大，也可以使用原生LORA方法进行微调。
  
DoRA 将预先训练好的权矩阵分解为幅度向量和方向矩阵，然后将LoRA运用于方向矩阵，可以增强 LoRA 的学习能力和训练稳定性，同时避免了任何额外的推理开销。

## 代码实现
由于本次微调采用的是可视化操作界面，后续学习了如何实现代码

### 微调训练
```
python3 finetune_hf.py data/self_cognition /root/models/chatglm3-6b configs/lora.yaml 
```
- 数据集： data/self_cognition
- 基础模型： /root/models/chatglm3-6b
- 配置参数： configs/lora.yaml

### 推理测试结果
```
python3 inference_hf.py output/checkpoint-3000/ --prompt "你是谁?"
```
- 预训练模型： output/checkpoint-3000
训练按照 configs/lora.yaml 的配置参数训练完成，保存到 output目录。(./output/checkpoint-3000)

#### 模型合并导出
```
python3 model_export_hf.py ./output/checkpoint-3000/ --out-dir ./chatglm3-6b-01
```
- 预训练模型目录（lora）： ./output/checkpoint-3000/
- 合并后模型输出目录： --out-dir ./chatglm3-6b-01

## 代码解读
### 训练模型
```
  def main(
        data_dir: Annotated[str, typer.Argument(help='')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='')],
):
```
 参数行，定义数据集的目录，预训练文件目录，配置文件目录
```
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)
```
从配置文件加载微调配置，加载分词器和模型。
创建一个 DataManager 对象，用于管理数据集。它使用 data_dir 和微调配置中的 data_config 作为输入。
```
    train_dataset = data_manager.get_dataset(
        Split.TRAIN, 
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length, 
            max_output_length=ft_config.max_output_length, 
        ),
        batched=True,
    )
    print('train_dataset:', train_dataset)

    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)
    # checks encoded dataset
    # _sanity_check(
    #     train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer
    # )

    # turn model to fp32
```
使用 data_manager.get_dataset 方法获取训练、验证和测试数据集。
对于每个数据集，它接受一个数据拆分类型，一个处理批次的函数，以及是否批处理的标志。
处理批次的函数使用了分词器、最大输入长度和最大输出长度作为参数。

```
    _prepare_model_for_training(model)
```
将模型设置为训练模式
```
    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
```
生成配置中的填充（pad）标记ID和结束（eos）标记ID
```
    model.gradient_checkpointing_enable()
 ```
 启用模型的梯度检查点功能（梯度检查点是一种节省显存的技术，它只在反向传播时保存中间激活值，而不是在整个前向传播过程中都保存。这有助于训练大型模型时减少显存使用）
 ``` 
    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(list(range(50))),
        tokenizer=tokenizer,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )
```
创建一个 Seq2SeqTrainer 对象，用于模型的训练，使用了模型的配置、数据集、分词器和其他参数
eval_dataset 被限制为验证数据集中的前50个样本，为了快速验证或节省计算资源。
```
    trainer.train()

    # test stage
    if test_dataset is not None:
        trainer.predict(test_dataset)
```
训练，如果存在测试集，用test_dataset方法对测试数据集进行预测