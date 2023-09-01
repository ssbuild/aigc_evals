## aigc_evals

 aigc_evals 是在openai/evals基础上修改而来，用于评估基于 aigc_serving 等仿openai 接口开源模型服务的脚本。<br />
部署开源模型移步至[aigc_serving](https://github.com/ssbuild/aigc_serving) 


## 当前支持评估数据集
目前支持 cmmlu , ceval,  mmlu 和 翻译数据集评估


## 安装

```commandline
pip install aigc_evals>=0.0.2

# 源码安装
git clone -b dev https://github.com/ssbuild/aigc_evals.git
pip install -e .
```
## 修改环境变量

```commandline
auto_eval/config.py
设置openai url等参数

```

## 修改自定义评估模型

修改 registry/completion_fns/langchain_aigc_serving

```text
langchain/chat_model/chatglm2-6b-int4:
  class: aigc_evals.completion_fns.langchain_llm:LangChainChatModelCompletionFn
  args:
    llm: ChatOpenAI
    chat_model_kwargs:
      model_name: chatglm2-6b-int4
      model_kwargs: # langchain 未明确实现的参数
        adapter_model: default
      max_retries: 10
      top_p: 0.7
      temperature: 0.95
      max_tokens: 2000
```

修改替换 chatglm2-6b-int4 成自己的开放模型，chat_model_kwargs 为 langchain ChatOpenAI 参数


## 一键评估
```commandline
cd auto_eval
python ceval.py
```

```commandline
cd auto_eval
python cmmlu.py
```

```commandline
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
下载mmlu数据集
cd auto_eval
python mmlu.py
```

```commandline
cd auto_eval
python translate_zh_en.py
```

## exec_aigc_evals 使用帮助

```text
exec_aigc_evals --help
usage: exec_aigc_evals [-h] [--extra_eval_params EXTRA_EVAL_PARAMS] [--max_samples MAX_SAMPLES] [--cache CACHE]
                       [--visible VISIBLE] [--seed SEED] [--user USER] [--record_path RECORD_PATH]
                       [--log_to_file LOG_TO_FILE] [--registry_path REGISTRY_PATH] [--debug DEBUG]
                       [--local-run LOCAL_RUN] [--http-run HTTP_RUN] [--http-run-url HTTP_RUN_URL]
                       [--http-batch-size HTTP_BATCH_SIZE] [--http-fail-percent-threshold HTTP_FAIL_PERCENT_THRESHOLD]
                       [--dry-run DRY_RUN] [--dry-run-logging DRY_RUN_LOGGING]
                       completion_fn eval

Run evals through the API

positional arguments:
  completion_fn         One or more CompletionFn URLs, separated by commas (,). A CompletionFn can either be the name
                        of a model available in the OpenAI API or a key in the registry (see
                        evals/registry/completion_fns).
  eval                  Name of an eval. See registry.

optional arguments:
  -h, --help            show this help message and exit
  --extra_eval_params EXTRA_EVAL_PARAMS
  --max_samples MAX_SAMPLES
  --cache CACHE
  --visible VISIBLE
  --seed SEED
  --user USER
  --record_path RECORD_PATH
  --log_to_file LOG_TO_FILE
                        Log to a file instead of stdout
  --registry_path REGISTRY_PATH
                        Path to the registry
  --debug DEBUG
  --local-run LOCAL_RUN
                        Enable local mode for running evaluations. In this mode, the evaluation results are stored
                        locally in a JSON file. This mode is enabled by default.
  --http-run HTTP_RUN   Enable HTTP mode for running evaluations. In this mode, the evaluation results are sent to a
                        specified URL rather than being stored locally or in Snowflake. This mode should be used in
                        conjunction with the '--http-run-url' and '--http-batch-size' arguments.
  --http-run-url HTTP_RUN_URL
                        URL to send the evaluation results when in HTTP mode. This option should be used in
                        conjunction with the '--http-run' flag.
  --http-batch-size HTTP_BATCH_SIZE
                        Number of events to send in each HTTP request when in HTTP mode. Default is 1, i.e., send
                        events individually. Set to a larger number to send events in batches. This option should be
                        used in conjunction with the '--http-run' flag.
  --http-fail-percent-threshold HTTP_FAIL_PERCENT_THRESHOLD
                        The acceptable percentage threshold of HTTP requests that can fail. Default is 5, meaning 5%
                        of total HTTP requests can fail without causing any issues. If the failure rate goes beyond
                        this threshold, suitable action should be taken or the process will be deemed as failing, but
                        still stored locally.
  --dry-run DRY_RUN
  --dry-run-logging DRY_RUN_LOGGING

```


## Licenses

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

本项目遵循 [MIT License](https://lbesson.mit-license.org/).

[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

