# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/1 9:23
import os




# MODEL = "langchain/chat_model/qwen-7b-chat-int4"
# MODEL = "langchain/chat_model/baichuan-13b-chat-int4"
# MODEL = "langchain/chat_model/xverse-13b-chat-int4"
# MODEL = "langchain/chat_model/moss-moon-003-sft-int4"
# MODEL = "langchain/chat_model/internlm-chat-7b-int4"
# MODEL = "langchain/chat_model/llama2-70b-hf-int4"
# MODEL = "langchain/chat_model/baichuan2-7b-chat-int4"
# MODEL = "langchain/chat_model/baichuan2-13b-chat-int4"
# MODEL = "langchain/chat_model/tigerbot-70b-chat"
# MODEL = "langchain/chat_model/internlm-chat-20b"
# MODEL = "langchain/chat_model/ChatYuan-large-v2"
# MODEL = "langchain/chat_model/chatglm2-6b-int4"
# MODEL = "langchain/chat_model/chatglm3-6b"
# MODEL = "langchain/chat_model/CausalLM-14B"
# MODEL = "langchain/chat_model/Yi-34B-Chat"
# MODEL = "langchain/chat_model/Baichuan2-13B-Chat"
# MODEL = "langchain/chat_model/Baichuan2-7B-Chat"
# MODEL = "langchain/chat_model/Baichuan-13B-Chat"
# MODEL = "langchain/chat_model/chatglm2-6b"
# MODEL = "langchain/chat_model/XVERSE-13B-Chat"
# MODEL = "langchain/chat_model/internlm-chat-7b"
# MODEL = "langchain/chat_model/internlm-chat-20b"
# MODEL = "langchain/chat_model/chatglm3-6b"
# MODEL = "langchain/chat_model/Qwen-14B-Chat"
MODEL = "langchain/chat_model/Qwen-1_8B-Chat"
MODEL = "langchain/chat_model/Qwen-72B-Chat"


EVALS_THREADS = 2 # 根据服务的并发设置， 避免读取超时
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://192.168.2.180:8081/v1"
FORCE_EVAL = False # 评估结果存在，是否重新评估

def env_setting():
    # 限制并发数目, 避免等待超时
    if os.environ.get("EVALS_THREADS", None) is None:
        os.environ['EVALS_THREADS'] = str(EVALS_THREADS)

    if os.environ.get("OPENAI_API_KEY", None) is None:
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    if os.environ.get("OPENAI_API_BASE",None) is None:
        os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE



    os.environ['PYTHONPATH'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),'..')

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
    output_path = os.path.join(os.path.dirname(__file__), 'outputs')
    return output_path

def get_output_path_metric():
    output_path = os.path.join(os.path.dirname(__file__), 'model_result')
    return output_path