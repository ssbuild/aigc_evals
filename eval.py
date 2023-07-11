# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/11 10:16
import argparse
import os
from enum import Enum
import time

import pandas as pd

choices = ["A", "B", "C", "D"]

class ModelType(Enum):
    BAICHUAN = 0
    BAICHUAN2 = 1
    CHATGLM = 2
    CHATGLM2 = 3
    LLM = 4
    MOSS = 5
    RWKV = 6

model_base = ModelType.CHATGLM

if model_base == ModelType.BAICHUAN:
    from evaluate.baichuan.prompt import EvaluateBuilder
elif model_base == ModelType.BAICHUAN2:
    from evaluate.baichuan2.prompt import EvaluateBuilder
elif model_base == ModelType.CHATGLM:
    from evaluate.chatglm.prompt import EvaluateBuilder
elif model_base == ModelType.CHATGLM2:
    from evaluate.chatglm2.prompt import EvaluateBuilder
elif model_base == ModelType.LLM:
    from evaluate.llm.prompt import EvaluateBuilder
elif model_base == ModelType.MOSS:
    from evaluate.moss.prompt import EvaluateBuilder
elif model_base == ModelType.RWKV:
    from evaluate.rwkv.prompt import EvaluateBuilder
else:
    raise ValueError('not support yet')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--openai_key", type=str, default="xxx")
    parser.add_argument("--minimax_group_id", type=str, default="xxx")
    parser.add_argument("--minimax_key", type=str, default="xxx")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--subject", "-s", type=str, default="operating_system")
    parser.add_argument("--cuda_device", type=str)
    args = parser.parse_args()

    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    subject_name = args.subject
    if not os.path.exists(r"logs"):
        os.mkdir(r"logs")
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_result_dir = os.path.join(r"logs", f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)
    print(subject_name)
    val_file_path = os.path.join('data/val', f'{subject_name}_val.csv')
    val_df = pd.read_csv(val_file_path)
    if args.few_shot:
        dev_file_path = os.path.join('data/dev', f'{subject_name}_dev.csv')
        dev_df = pd.read_csv(dev_file_path)
        correct_ratio = EvaluateBuilder.eval_subject(subject_name, val_df, dev_df, few_shot=args.few_shot,
                                               save_result_dir=save_result_dir, cot=args.cot)
    else:
        correct_ratio = EvaluateBuilder.eval_subject(subject_name, val_df, few_shot=args.few_shot,
                                               save_result_dir=save_result_dir)
    print("Acc:", correct_ratio)