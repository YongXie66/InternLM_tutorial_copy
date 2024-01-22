# Legal_LLM微调

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

   执行[python](https://github.com/YongXie66/InternLM_tutorial_copy/blob/main/notes/legal_LLM/data2jsonl.py)脚本，获得格式化后的数据集
   
   执行[python](https://github.com/YongXie66/InternLM_tutorial_copy/blob/main/notes/legal_LLM/merge.py)脚本，合并数据集

## 2. 安装环境

```bash
conda create --name xtuner0.1.9 python=3.10 -y
conda activate xtuner0.1.9
cd ~
mkdir xtuner019 && cd xtuner019
git clone -b v0.1.9 https://gitee.com/Internlm/xtuner
cd xtuner
pip install -e '.[all]'
mkdir ~/ft-law && cd ~/ft-law
```

## 3. 微调

### 3.1 准备配置文件

```bash
cd ~/ft-oasst1
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
mv internlm_chat_7b_qlora_oasst1_e3_copy.py internlm_chat_7b_qlora_law_e3.py
```

### 3.2 模型 & 数据集下载

- internlm-chat-7b模型：直接复制平台已有模型
- 手动导入数据集

```bash
cp -r /root/share/temp/model_repos/internlm-chat-7b ~/ft-oasst1/
cp -r /root/share/temp/datasets/openassistant-guanaco .
```

### 3.3 修改配置文件

即 修改配置文件 internlm_chat_7b_qlora_law_e3_copy.py 中的**模型&数据集的本地路径等**

```bash
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = 'law_merged2.jsonl'

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

### 3.4 开始微调

#### 训练

```bash
# 1/4 A100，并用deepspeed加速
apt update -y
apt install tmux -y
tmux new -s finetune
xtuner train ./internlm_chat_7b_qlora_law_e3_copy.py --deepspeed deepspeed_zero2
```

#### 将得到的 .pth 模型转换为 HuggingFace 模型，**即：生成 Adapter 文件夹**

```bash
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert pth_to_hf ./internlm_chat_7b_qlora_law_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_law_e3_copy/epoch_1.pth ./hf
```

## 4. 部署与测试

### 4.1 将 HuggingFace adapter 合并到大语言模型

```bash
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
```

### 4.2 与合并后的模型对话

```bash
# 加载 Adapter 模型对话（Float 16）
# xtuner chat ./merged --prompt-template internlm_chat
# 4 bit 量化加载
xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```

