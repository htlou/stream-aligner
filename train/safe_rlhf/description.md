## Safe-RLHF

1. The File Tree

```
.
├── algorithms
│   ├── dpo
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── main.py
│   │   └── trainer.py
│   ├── __init__.py
│   ├── ppo
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── main.py
│   │   └── trainer.py
│   ├── ppo_lag
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── main.py
│   │   └── trainer.py
│   └── ppo_reward_shaping
│       ├── __init__.py
│       ├── __main__.py
│       ├── main.py
│       └── trainer.py
├── configs
│   ├── constants.py
│   ├── deepspeed_config.py
│   ├── ds_eval_config_template.json
│   ├── ds_train_config_template.json
│   ├── fsdp_config.json
│   └── __init__.py
├── datasets
│   ├── base.py
│   ├── __init__.py
│   ├── preference.py
│   ├── prompt_only.py
│   ├── raw
│   │   ├── alpaca.py
│   │   ├── correction.py
│   │   ├── empathy.py
│   │   ├── firefly.py
│   │   ├── hh_rlhf.py
│   │   ├── __init__.py
│   │   ├── moss.py
│   │   └── safe_rlhf.py
│   ├── safety_preference.py
│   ├── supervised.py
│   └── utils.py
├── evaluate
│   ├── arena.py
│   ├── bigbench
│   │   ├── eval.py
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── model.py
│   ├── cost.py
│   ├── gpt4
│   │   ├── eval.py
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── problem.json
│   ├── __init__.py
│   └── reward.py
├── finetune
│   ├── deepspeed.py
│   ├── huggingface.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py
│   └── trainer.py
SupervisedFinetuneTrainer
├── __init__.py
├── logger.py
├── models
│   ├── __init__.py
│   ├── normalizer.py
基类：Normalizer
Normalizer 是一个抽象基类，它扩展了 torch.nn.Module，表示它是一个可用于PyTorch网络的模块。
类中定义了 mean, var, count 和 normalize_function 作为模块的状态。
构造函数中，根据传入的 normalize_function（可以是 'affine', 'scale', 'translate', 或 'identity'），初始化 mean, var, 和 count。
forward 方法负责处理传入的数据，根据模型的训练状态调用 update 方法更新统计值，然后调用 normalize 方法来规范化数据。
normalize 方法根据 normalize_function 的类型来选择不同的规范化操作。
set_mean_var 方法允许手动设置 mean 和 var 的值。
派生类：RunningMeanStd 和 ExponentialMovingAverage
RunningMeanStd 类实现了一个运行平均的统计更新策略，适用于在数据流中动态更新数据的统计特性。
ExponentialMovingAverage 类使用指数移动平均来更新统计特性，这种方法对最近的观测赋予更高的权重。
这两个类都重写了 update 方法来实现它们特定的更新逻辑。
辅助类：IdentityNormalizer
IdentityNormalizer 是一个特殊的规范器，实际上并不改变输入数据，只是通过统计数据流的数量。
工厂方法：instantiate
instantiate 类方法根据传入的参数创建并返回一个合适的 Normalizer 对象。这使得用户可以方便地根据配置动态创建不同类型的规范器。

│   ├── pretrained.py
1. resize_tokenizer_embedding 函数：这个函数用于调整分词器和模型的词嵌入大小以匹配。如果模 型和分词器的词汇大小不一致，此函数还会发出警告。它还包括一个初始化新嵌入的功能，使新添加的词汇使用现有词汇的平均嵌入值初始化。
2. verify_vocabulary_embedding_sizes 子函数：验证分词器的词汇量和模型嵌入层的大小是否一致。
3. init_new_embeddings 子函数：初始化新词汇的嵌入，使用现有嵌入的均值来填充新增加的嵌入。
4. load_pretrained_models 函数：这是一个加载预训练模型和分词器的函数，它支持多种参数设置，如模型最大长度、填充方式、是否自动映射到多设备上、数据类型等。此函数首先加载模型和分词器，然后调用resize_tokenizer_embedding来调整嵌入层大小，确保分词器和模型的词汇量一致。

│   └── score_model 
│       ├── bloom
│       │   ├── __init__.py
│       │   └── modeling_bloom.py
│       ├── gpt2
│       │   ├── __init__.py
│       │   └── modeling_gpt2.py
│       ├── gptj
│       │   ├── __init__.py
│       │   └── modeling_gptj.py
│       ├── gpt_neo
│       │   ├── __init__.py
│       │   └── modeling_gpt_neo.py
│       ├── gpt_neox
│       │   ├── __init__.py
│       │   └── modeling_gpt_neox.py
│       ├── __init__.py
│       ├── llama
│       │   ├── __init__.py
│       │   └── modeling_llama.py
│       └── opt
│           ├── __init__.py
│           └── modeling_opt.py
├── serve
│   ├── arena.py
│   ├── chatbot.py
│   ├── cli.py
│   └── __init__.py
├── trainers
│   ├── base.py
│   ├── __init__.py
│   ├── rl_trainer.py
│   └── supervised_trainer.py
├── utils.py
├── values
│   ├── cost
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── main.py
│   │   └── trainer.py
│   ├── __init__.py
│   └── reward
│       ├── __init__.py
│       ├── __main__.py
│       ├── main.py
│       └── trainer.py
└── version.py
```