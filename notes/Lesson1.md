# 书生·浦语大模型全链路开源开放体系

视频链接：[bilibili](https://www.bilibili.com/video/BV1Rc411b7ns/)

2023年7月6日，上海人工智能实验室推出全面开源免费商用的模型InternLM-7B模型。InternLM包括7B、20B、123B各个级别模型，前两者开源可用。20B模型可达到Llama2-70B的水平。

该体系具体包括：

- 数据（书生万卷）
- 预训练（InternLM-Train）
- 微调XTuner+部署LMDeploy
- 评测OpenCompass
- 应用Lagent&AgentLego

## 数据: 书生·万卷

链接：[书生·万卷 1.0](https://github.com/opendatalab/WanJuan1.0)

1. 文本数据：数据量超1TB
2. 图像-文本数据集：数据量超140GB
3. 视频数据：数据量超900GB

总数据量：2TB

更多OpenDataLab数据集80TB：[OpenDataLab](https://opendatalab.com)


## 预训练: InternLM-Train

1. 高可拓展：支持从8卡到千卡训练，千卡加速效率达92%；
2. 极致性能优化：Hybrid Zero独特技术 + 极致优化，加速50%；
3. 兼容主流：无缝衔接HuggingFace等技术生态，支持各类轻量化技术；
4. 开箱即用：支持多种规格语言模型，修改配置即可训练。

## 微调: XTuner

### 增量续训

1. 使用场景：让基座模型学习到一些新知识，如某个垂类领域知识；
2. 训练数据：文章、书籍、代码等。

### 有监督微调

1. 使用场景：让模型学会理解和遵循各类指令，或者注入少量领域知识；
2. 训练数据：高质量的对话、问答数据。

### XTuner:

- 适配多种生态
- 适配多种硬件

## 评测: OpenCompass

1. 丰富模型支持：开源模型、API模型一站式评测；
2. 分布式高效评测：支持千亿参数模型在海量数据集上分布式评测；
3. 便捷的数据集接口：支持社区用户根据自身需求快速添加自定义数据集；
4. 敏捷的能力迭代：每周更新大模型能力榜单，每月提升评测工具能力。

## 部署: LMDeploy

### 高效推理引擎

1. 持续批处理技巧
2. 深度优化的低比特计算kernel
3. 模型并行
4. 高效的k/v缓存管理机制

### 完善易用的工具链

1. 量化、推理、服务全流程
2. 无缝衔接OpenCompass评测推理精度
3. 和OpenAI接口高度兼容的API server

## 应用: Lagent Agentlego 

### 智能体框架Lagent

1. 支持多种类型的智能体能力
2. 灵活支持多种大语言模型
3. 简单易拓展，支持丰富的工具

### 多模态智能体工具箱AgentLego

1. 丰富的工具集合，尤其是提供了大量视觉、多模态相关领域的前沿算法功能
2. 支持多个主流智能体系统，如LangChain, Transformers Agent, Lagent等
3. 灵活的多模态工具调用接口，可以轻松支持各类输入输出格式的工具函数
4. 一键式远程工具部署，轻松使用和调试大模型智能体
