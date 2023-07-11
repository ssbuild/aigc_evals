# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/11 10:16
import sys

import numpy as np

sys.path.append('.')
import argparse
import os
from enum import Enum
import time
from evaluate.constant_map import train_info_args

import pandas as pd

choices = ["A", "B", "C", "D"]

task_list=[
"computer_network",
"operating_system",
"computer_architecture",
"college_programming",
"college_physics",
"college_chemistry",
"advanced_mathematics",
"probability_and_statistics",
"discrete_mathematics",
"electrical_engineer",
"metrology_engineer",
"high_school_mathematics",
"high_school_physics",
"high_school_chemistry",
"high_school_biology",
"middle_school_mathematics",
"middle_school_biology",
"middle_school_physics",
"middle_school_chemistry",
"veterinary_medicine",
"college_economics",
"business_administration",
"marxism",
"mao_zedong_thought",
"education_science",
"teacher_qualification",
"high_school_politics",
"high_school_geography",
"middle_school_politics",
"middle_school_geography",
"modern_chinese_history",
"ideological_and_moral_cultivation",
"logic",
"law",
"chinese_language_and_literature",
"art_studies",
"professional_tour_guide",
"legal_professional",
"high_school_chinese",
"high_school_history",
"middle_school_history",
"civil_servant",
"sports_science",
"plant_protection",
"basic_medicine",
"clinical_medicine",
"urban_and_rural_planner",
"accountant",
"fire_engineer",
"environmental_impact_assessment_engineer",
"tax_accountant",
"physician",]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--openai_key", type=str, default="xxx")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--subject", "-s", type=str, default=None)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()


    model_name = args.model_name.lower()
    if model_name.startswith("baichuan2"):
        from evaluate.baichuan2.prompt import EvaluateBuilder
        evaluator = EvaluateBuilder(choices, args.model_name, args.k)
    elif model_name.startswith("baichuan"):
        from evaluate.baichuan.prompt import EvaluateBuilder
        evaluator = EvaluateBuilder(choices, args.model_name, args.k)
    elif model_name.startswith("chatglm2"):
        from evaluate.chatglm2.prompt import EvaluateBuilder
        evaluator = EvaluateBuilder(choices, args.model_name, args.k)
    elif model_name.startswith("chatglm"):
        from evaluate.chatglm.prompt import EvaluateBuilder
        evaluator = EvaluateBuilder(choices, args.model_name, args.k)
    elif model_name.startswith("llama") or model_name.startswith("opt") or model_name.startswith("bloom"):
        from evaluate.llm.prompt import EvaluateBuilder
        evaluator = EvaluateBuilder(choices, args.model_name, args.k)
    elif model_name.startswith("moss"):
        from evaluate.moss.prompt import EvaluateBuilder
        evaluator = EvaluateBuilder(choices, args.model_name, args.k)
    elif model_name.startswith("rwkv"):
        from evaluate.rwkv.prompt import EvaluateBuilder
        evaluator = EvaluateBuilder(choices, args.model_name, args.k)
    elif model_name.startswith("chatgpt"):
        from evaluate.chatgpt.prompt import EvaluateBuilder
        evaluator = EvaluateBuilder(choices, args.model_name, args.k,args.openai_key)
    else:
        raise ValueError('not support yet')

    assert args.model_name in train_info_args,ValueError("{} is not in ".format(args.model_name,str(train_info_args.keys())))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    evaluator.init()
    subject_name = args.subject


    cur_path = os.path.dirname(__file__)
    log_dir = os.path.join(cur_path, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_result_dir = os.path.join(log_dir, f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)

    f_out = open(os.path.join(save_result_dir,'summary.txt'),mode='w',encoding='utf-8')
    acc_all = []
    for task_name in task_list:
        if subject_name is not None:
            if subject_name.lower() != task_name.lower():
                continue
        val_file_path = os.path.join(cur_path,'data/val', f'{task_name}_val.csv')
        val_df = pd.read_csv(val_file_path)

        if args.few_shot:
            dev_file_path = os.path.join(cur_path,'data/dev', f'{task_name}_dev.csv')
            dev_df = pd.read_csv(dev_file_path)
            correct_ratio = evaluator.eval_subject(task_name, val_df, dev_df, few_shot=args.few_shot,
                                                   save_result_dir=save_result_dir, cot=args.cot)
        else:
            correct_ratio = evaluator.eval_subject(task_name, val_df, few_shot=args.few_shot,
                                                   save_result_dir=save_result_dir)
        print(task_name,"Acc:", correct_ratio)
        acc_all.append(correct_ratio)
        f_out.write('{},"Acc:", {}\n'.format(task_name,correct_ratio))


    print('avg',np.average(acc_all))
    f_out.write('avg,"Acc:", {}\n'.format(np.average(acc_all)))
    f_out.close()