# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/1 9:23
import os

EVALS_THREADS = 2 # 根据服务的并发设置， 避免读取超时
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://192.168.2.180:8081/v1"

MODEL = "langchain/chat_model/chatglm2-6b-int4"
FORCE_EVAL = False # 评估结果存在，是否重新评估

def env_setting():
    # 限制并发数目, 避免等待超时
    if os.environ.get("EVALS_THREADS", None) is None:
        os.environ['EVALS_THREADS'] = str(EVALS_THREADS)

    if os.environ.get("OPENAI_API_KEY", None) is None:
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    if os.environ.get("OPENAI_API_BASE",None) is None:
        os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE


    os.environ['PYTHONPATH'] = os.path.abspath(os.path.dirname(__file__))

    registry_path = get_registry_path()
    p = os.path.join(registry_path,'data')
    os.makedirs(p,exist_ok=True)
    p = os.path.join(registry_path, 'evals')
    os.makedirs(p,exist_ok=True)
    os.makedirs(get_output_path(), exist_ok=True)



def get_registry_path():
    # 注册路径，不建议更改
    return os.path.join(os.path.dirname(__file__), "../registry")


def get_output_path():
    output_path = os.path.join(os.path.dirname(__file__), '../outputs')
    return output_path

def get_output_path_metric():
    output_path = os.path.join(os.path.dirname(__file__), '../model_result')
    return output_path