## 安装
```text
dev 通过一下方式安装 , 注意顺序
pip uninstall aigc_zoo
pip install -U git+https://github.com/ssbuild/aigc_zoo#egg=aigc_zoo
pip uninstall deep_training
pip install -U git+https://github.com/ssbuild/deep_training.git
pip install -U transformers>=4.30 deepspeed transformers_stream_generator bitsandbytes>=0.39 accelerate>=0.20
```

## 数据

[ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)

wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
解压后 数据放置 aigc_eval/data 目录

## 本仓库部分代码参考 C-Eval
    

## ceval排行榜

#### Zero-shot
| Model              | STEM | Social Science | Humanities | Other | Average |
|--------------------|:----:|:--------------:|:----------:|:-----:|:-------:|
| CHATGLM2-6B        | 42.7 |      60.6      |    58.1    | 49.9  |  52.8   |
| Baichuan-13b-Chat  | 43.2 |      50.9      |    52.7    | 42.0  |  47.2   |
| Baichuan-7B        | 31.2 |      50.8      |    41.1    | 42.4  |  41.4   |
| CHATGLM-6B         | 33.4 |      43.9      |    36.4    | 32.2  |  36.4   |




#### Five-shot
| Model             | STEM | Social Science | Humanities | Other | Average |
|-------------------|:----:|:--------------:|:----------:|:-----:|:-------:|
| CHATGLM2-6B       | 46.9 |      57.8      |    54.7    | 46.5  |  51.6   |
| Baichuan-7B       | 36.6 |      53.7      |    45.1    | 44.7  |  45.0   |
| Baichuan-13b-Chat | 30.2 |      45.3      |    47.1    | 32.9  |  38.9   |
| CHATGLM-6B        | 33.3 |      43.2      |    39.7    | 35.5  |  38.0   |



## cmmlu排行榜

#### Zero-shot
| Model             | STEM | Social Science | Humanities | Other | China specific | Average |
|-------------------|:----:|:--------------:|:----------:|:-----:|:--------------:|:-------:|
| CHATGLM2-6B       | 41.7 |      51.4      |    53.0    | 52.3  |      49.6      |  49.6   |
| Baichuan-13b-Chat | 34.7 |      50.4      |    50.2    | 50.1  |      48.7      |  46.8   |
| Baichuan-7B       | 40.0 |      45.9      |    43.6    | 43.9  |      42.4      |  41.6   |
| CHATGLM-6B        | 30.8 |      42.7      |    40.4    | 39.8  |      39.4      |  38.6   |




#### Five-shot
| Model              | STEM | Social Science | Humanities | Other | China specific | Average |
|--------------------|:----:|:--------------:|:----------:|:-----:|:--------------:|:-------:|
| CHATGLM2-6B        | 41.9 |      50.0      |    50.5    | 50.2  |      47.8      |  48.0   |
| CHATGLM-6B         | 32.8 |      41.2      |    38.2    | 37.8  |      38.7      |  37.8   |
| Baichuan-13b-Chat  | 29.1 |      40.0      |    38.6    | 41.8  |      35.2      |  37.0   |



## 前言

怎么去评估一个大语言模型呢？

在广泛的NLP任务上进行评估。
在高级LLM能力上进行评估，比如推理、解决困难的数学问题、写代码。
在英文中，已经有不少评测基准：

传统英语基准：GLUE，是NLU任务的的评测基准。
MMLU基准（Hendrycks等人，2021a）提供了从真实世界的考试和书籍中收集的多领域和多任务评价。
BIG-bench基准（Srivastava等人，2022年）包括204个不同的任务，其中一些任务被认为超出了当前LLM的能力。
HELM基准（Liang等人，2022年）汇总了42个不同的任务，用从准确性到鲁棒性的7个指标来评估LLMs。
中文评测基准：

CLUE基准（Xu等人，2020）是第一个大规模的中文NLU基准，现在仍然是使用最广泛和最好的中文基准。
AGIEval基准（Zhong等人，2023）包含了来自中国高考、中国律师资格考试和中国公务员考试的数据。
MMCU基准（Zeng，2023）包括来自医学、法律、心理学和教育等四大领域的测试，这些数据也是从中国高考、资格考试以及大学考试中收集的。


C-EVAL 与上述评估基准的区别： 覆盖更广泛的领域。
具有四种不同的难度--特别是C-EVAL HARD基准是中国第一个提供复杂推理问题的基准。
努力减少数据泄漏--作者的问题大多来自模拟考试的PDF或Microsoft Word文件，这些文件由作者进一步处理，而AGIEval和MMCU收集的是中国过去国家考试的确切题目。

## 如何在llm-eval上测试

通常你可以直接从模型的生成中使用正则表达式提取出答案选项（A,B,C,D)。在少样本测试中，模型通常会遵循少样本给出的固定格式，所以提取答案很简单。然而有时候，特别是零样本测试和面对没有做过指令微调的模型时，模型可能无法很好的理解指令，甚至有时不会回答问题。这种情况下我们推荐直接计算下一个预测token等于"A", "B", "C", "D"的概率，然后以概率最大的选项作为答案 -- 这是一种受限解码生成的方法，MMLU的[官方测试代码](https://github.com/hendrycks/test/blob/4450500f923c49f1fb1dd3d99108a0bd9717b660/evaluate.py#L88)中是使用了这种方法进行测试。注意这种概率方法对思维链的测试不适用。[更加详细的评测教程](resources/tutorial.md)。

在我们最初发布时，我们自己用了以下prompt进行测试：
#### 仅预测答案的prompt
```
以下是中国关于{科目}考试的单项选择题，请选出其中的正确答案。

{题目1}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
答案：A

[k-shot demo, note that k is 0 in the zero-shot case]

{测试题目}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
答案：
```

#### 思维链prompt
```
以下是中国关于{科目}考试的单项选择题，请选出其中的正确答案。

{题目1}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
答案：让我们一步一步思考，
1. {解析过程步骤1}
2. {解析过程步骤2}
3. {解析过程步骤3}
所以答案是A。

[k-shot demo, note that k is 0 in the zero-shot case]

{测试题目}
A. {选项A}
B. {选项B}
C. {选项C}
D. {选项D}
答案：让我们一步一步思考，
1. 
```



## 如何提交

您首先需要准备一个 UTF-8 编码的 JSON 文件，并按照以下格式编写。详情请参考[submission_example.json](http://101.42.176.124:8080/data_share/n)。

```
## 每个学科内部的键名是数据集中的"id"字段
{
    "chinese_language_and_literature": {
        "0": "A",
        "1": "B",
        "2": "B",
        ...
    },
    
    "学科名称":{
    "0":"答案1",
    "1":"答案2",
    ...
    }
    ....
}
```

然后你可以将准备好的JSON文件提交到[这里](http://101.42.176.124:8080/data_share/)，**请注意，你需要先登录才能访问提交页面**。



## TODO

- [x] 添加zero-shot结果
- [ ] 集成到openai eval



## Licenses

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

本项目遵循 [MIT License](https://lbesson.mit-license.org/).

[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

C-Eval数据集遵循 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

