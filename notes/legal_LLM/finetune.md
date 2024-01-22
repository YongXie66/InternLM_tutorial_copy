# XTuner大模型单卡低成本微调实战

## 1. 数据预处理

1. 数据来源链接: [百度网盘](https://pan.baidu.com/s/1sKXbbEntvrI68m68u8bz6A?pwd=mbij)  提取码:mbij

   选择公开数据集中的5个chat数据集，用于指令跟随微调

   ```python
   input_file_name1 = 'CrimeKgAssitant清洗后_52k.json'
   input_file_name2 = 'legal_article/article.txt'
   input_file_name3 = 'DISC-Law-SFT/DISC-Law-SFT-Pair.jsonl'
   input_file_name4 = 'hanfei/data/zh_contract_instruction.json'
   input_file_name6 = 'hanfei/data/zh_law_instruction.json'
   ```

   执行python脚本，获得格式化后的数据集



### 1. 安装

```bash
conda create --name xtuner0.1.9 python=3.10 -y

conda activate xtuner0.1.9
cd ~
mkdir xtuner019 && cd xtuner019

# 无法访问github的用户请从 gitee 拉取:
git clone -b v0.1.9 https://gitee.com/Internlm/xtuner

cd xtuner
# 从源码安装 XTuner
pip install -e '.[all]'

# 创建一个微调 oasst1 数据集的工作路径，进入
mkdir ~/ft-oasst1 && cd ~/ft-oasst1
```

### 2. 微调

#### 2.1 准备配置文件

```bash
# 列出所有开箱即用的配置
xtuner list-cfg

# 拷贝一个配置文件到当前目录，选择internlm_chat_7b_qlora_oasst1_e3的配置文件
cd ~/ft-oasst1
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
```

#### 2.2 模型 & 数据集下载

- internlm-chat-7b模型：直接复制平台已有模型，或者可以手动从ModelScope下载模型

- openassistant-guanaco(oasst1)数据集：由于 huggingface 网络问题，也直接复制平台提前下载的数据

```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b ~/ft-oasst1/
cp -r /root/share/temp/datasets/openassistant-guanaco .
```

#### 2.3 修改配置文件

即 修改配置文件 internlm_chat_7b_qlora_oasst1_e3_copy.py 中的**模型&数据集的本地路径**

核心超参：

| 参数名              | 解释                                                   |
| ------------------- | ------------------------------------------------------ |
| **data_path**       | 数据路径或 HuggingFace 仓库名                          |
| max_length          | 单条数据最大 Token 数，超过则截断                      |
| pack_to_max_length  | 是否将多条短数据拼接到 max_length，提高 GPU 利用率     |
| accumulative_counts | 梯度累积，每多少次 backward 更新一次参数               |
| evaluation_inputs   | 训练过程中，会根据给定的问题进行推理，便于观测训练状态 |
| evaluation_freq     | Evaluation 的评测间隔 iter 数                          |
| ......              | ......                                                 |

如果想把显卡的现存吃满，充分利用显卡资源，可以将 `max_length` 和 `batch_size` 这两个参数调大

#### 2.4 开始微调

##### 训练

```bash
# 单卡，用刚才改好的config文件训练，并用deepspeed加速
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2
```

###### tmux (Terminal MUltipleXer) 终端复用器   [Tmux 使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)

```bash
apt update -y
apt install tmux -y
tmux new -s finetune  # 新建名为<finetune>的tmux会话窗口，按住ctrl+B,再按D可返回原始终端
tmux attach -t finetune  # 继续回到tmux虚拟窗口

```

##### 将得到的 .pth 模型转换为 HuggingFace 模型，**即：生成 Adapter 文件夹**

```bash
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
# xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}
xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf
```

**此时，hf 文件夹即为我们平时所理解的“LoRA 模型文件 (Adapter)”**

生成的**.safetensors**文件 (以前是.bin文件) 即为微调过后的LoRA模型

### 3. 部署与测试

####  3.1 将 HuggingFace adapter 合并到大语言模型

```bash
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB 分块保存
```

#### 3.2 与合并后的模型对话

```bash
# 加载 Adapter 模型对话（Float 16），注意底座模型不一样，对应的prompt-template就不同
xtuner chat ./merged --prompt-template internlm_chat
# 4 bit 量化加载
# xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```

==注：==也可以选择不融合 basemodel和adapter，而直接使用语句

```bash
xtuner chat ./internlm-chat-7b --adapter internlm-7b-qlora-msagent-react
```



#### 3.3 微调效果比较

加载模型时，选择加载 `internlm-chat-7b` 或者微调后的 `merged` 来比较两者效果

```bash
xtuner chat ./merged --prompt-template internlm_chat
xtuner chat ./internlm-chat-7b --prompt-template internlm_chat
```

`xtuner chat`的启动参数

| 启动参数              | 干哈滴                                                       |
| --------------------- | ------------------------------------------------------------ |
| **--prompt-template** | 指定对话模板                                                 |
| --system              | 指定SYSTEM文本                                               |
| --system-template     | 指定SYSTEM模板                                               |
| -**-bits**            | LLM位数                                                      |
| --bot-name            | bot名称                                                      |
| --with-plugins        | 指定要使用的插件                                             |
| **--no-streamer**     | 是否启用流式传输                                             |
| **--lagent**          | 是否使用lagent                                               |
| --command-stop-word   | 命令停止词                                                   |
| --answer-stop-word    | 回答停止词                                                   |
| --offload-folder      | 存放模型权重的文件夹（或者已经卸载模型权重的文件夹）         |
| --max-new-tokens      | 生成文本中允许的最大 `token` 数量                            |
| **--temperature**     | 温度值                                                       |
| --top-k               | 保留用于顶k筛选的最高概率词汇标记数                          |
| --top-p               | 如果设置为小于1的浮点数，仅保留概率相加高于 `top_p` 的最小一组最有可能的标记 |
| --seed                | 用于可重现文本生成的随机种子                                 |

### 4. 自定义微调

以 **[Medication QA](https://github.com/abachaa/Medication_QA_MedInfo2019)** **数据集**为例，将其往`医学问答`领域对齐

| 问题                                                  | 答案                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| What are ketorolac eye drops?（什么是酮咯酸滴眼液？） | Ophthalmic ketorolac is used to treat itchy eyes caused by allergies. It also is used to treat swelling and redness (inflammation) that can occur after cataract surgery. Ketorolac is in a class of medications called nonsteroidal anti-inflammatory drugs (NSAIDs). It works by stopping the release of substances that cause allergy symptoms and inflammation. |

#### 4.1 数据准备

原格式: (.xlsx)

| **==问题==** | 药物类型 | 问题类型 | ==**回答**== | 主题 | URL  |
| ------------ | -------- | -------- | ------------ | ---- | ---- |
| aaa          | bbb      | ccc      | ddd          | eee  | fff  |

##### 4.1.1 将原格式数据转为XTuner的.jsonL数据格式

ChatGPT 生成的 python 代码见本仓库的 [xlsx2jsonl.py](https://github.com/InternLM/tutorial/blob/main/xtuner/xlsx2jsonl.py)  ==有空再跑一遍==

```
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

#### 4.2 开始自定义微调

```bash
mkdir ~/ft-medqa && cd ~/ft-medqa
cp -r ~/ft-oasst1/internlm-chat-7b .  # 基座模型
git clone https://github.com/InternLM/tutorial  # 开发机用不了，只能upload
cp ~/tutorial/xtuner/MedQA2019-structured-train.jsonl .
```

##### 4.2.1 准备配置文件

```bash
# 复制配置文件到当前目录
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
# 改个文件名，对应medqa数据集
mv internlm_chat_7b_qlora_oasst1_e3_copy.py internlm_chat_7b_qlora_medqa2019_e3.py
```

修改配置文件.py的内容

```bash
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据为 MedQA2019-structured-train.jsonl 路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = 'MedQA2019-structured-train.jsonl'

# 修改 train_dataset 对象
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
```

##### 4.2.2 启动！

先老样子 `tmux new -s medqa`

```bash
xtuner train internlm_chat_7b_qlora_medqa2019_e3.py --deepspeed deepspeed_zero2
```

##### 4.2.3 .pth转.safetensors

跟2.4节一样 

##### 4.2.4 部署与测试

和第3节一样，InternStudio提醒我内存不足了，就先不部署了

### 【补充】用MS-Agent数据集，赋予LLM以Agent能力

已阅读，内存不足暂未实战