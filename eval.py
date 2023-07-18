# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/11 10:16

import sys
sys.path.append('.')
import numpy as np
import json
import argparse
import os
import time
from evaluate.constant_map import train_info_args

import pandas as pd

choices = ["A", "B", "C", "D"]


def eval_ceval(evaluator,subject_name,save_result_dir):
    with open(os.path.join(os.path.dirname(__file__), 'ceval_categories.json'), mode='r', encoding='utf-8') as f:
        tasks: dict = json.loads(f.read())

    f_out = open(os.path.join(save_result_dir, 'summary.txt'), mode='w', encoding='utf-8')
    acc_classify = set([task[-1] for task in tasks.values()])
    acc_classify = {k: [] for k in acc_classify}
    for task_name in tasks:
        if subject_name is not None:
            if subject_name.lower() != task_name.lower():
                continue
        task = tasks[task_name]
        val_file_path = os.path.join(cur_path, 'ceval_data/val', f'{task_name}_val.csv')
        val_df = pd.read_csv(val_file_path)

        if args.few_shot:
            dev_file_path = os.path.join(cur_path, 'ceval_data/dev', f'{task_name}_dev.csv')
            dev_df = pd.read_csv(dev_file_path)
            correct_ratio = evaluator.eval_subject(task_name, val_df, dev_df, few_shot=args.few_shot,
                                                   save_result_dir=save_result_dir, cot=args.cot)
        else:
            correct_ratio = evaluator.eval_subject(task_name, val_df, few_shot=args.few_shot,
                                                   save_result_dir=save_result_dir)

        acc_classify[task[-1]].append(correct_ratio)
        print(task_name, *task, "Acc:", correct_ratio)
        f_out.write('{} {}_{}_{},"Acc:", {}\n'.format(task_name, task[0], task[1], task[2], correct_ratio))

    acc_avg = {k: np.average(v) for k, v in acc_classify.items()}
    acc_avg_all = np.average([np.average(v) for k, v in acc_classify.items()])
    f_out.write('{}\n'.format(json.dumps(acc_classify, ensure_ascii=True, indent=2)))
    f_out.write('{}\n'.format(json.dumps(acc_avg, ensure_ascii=True, indent=2)))
    f_out.write('***avg acc {}***\n'.format(acc_avg_all))
    print(acc_classify)
    print(acc_avg)
    print('***avg acc {}***\n'.format(acc_avg_all))
    f_out.close()

def eval_cmmlu(evaluator,subject_name,save_result_dir):
    from cmmlu_categories import name_en2zh,subcategories,categories

    categories_new = {}
    for k,v in categories.items():
        for sub_k in v:
            categories_new[sub_k] = k
    tasks = {}
    for k,parent_cate in subcategories.items():
        parent_cate = subcategories[k]
        root_cat = [ categories_new[cat] for cat in parent_cate]
        tasks[k] = (name_en2zh[k],parent_cate,root_cat)

    f_out = open(os.path.join(save_result_dir, 'summary.txt'), mode='w', encoding='utf-8')
    acc_classify = {root: [] for root in categories.keys()}
    for task_name in tasks:
        if subject_name is not None:
            if subject_name.lower() != task_name.lower():
                continue
        task = tasks[task_name]
        val_file_path = os.path.join(cur_path, 'cmmlu_data/test', f'{task_name}.csv')
        val_df = pd.read_csv(val_file_path)
        val_df.columns = val_df.columns.str.replace('Question', 'question')
        val_df.columns = val_df.columns.str.replace('Answer', 'answer')
        val_df.loc[:, 'explanation'] = ''
        if args.few_shot:
            dev_file_path = os.path.join(cur_path, 'cmmlu_data/dev', f'{task_name}.csv')
            dev_df = pd.read_csv(dev_file_path)
            dev_df.columns = dev_df.columns.str.replace('Question', 'question')
            dev_df.columns = dev_df.columns.str.replace('Answer', 'answer')
            dev_df.loc[:, 'explanation'] = ''
            correct_ratio = evaluator.eval_subject(task_name, val_df, dev_df, few_shot=args.few_shot,
                                                   save_result_dir=save_result_dir, cot=args.cot)
        else:
            correct_ratio = evaluator.eval_subject(task_name, val_df, few_shot=args.few_shot,
                                                   save_result_dir=save_result_dir)
        roots = task[-1]
        for root in roots:
            acc_classify[root].append(correct_ratio)
        print(task_name, *task, "Acc:", correct_ratio)
        f_out.write('{} {}_{}_{},"Acc:", {}\n'.format(task_name, task[0], ','.join(task[1]), ','.join(task[2]), correct_ratio))

    acc_avg = {k: np.average(v) for k, v in acc_classify.items()}
    acc_avg_all = np.average([np.average(v) for k, v in acc_classify.items()])
    f_out.write('{}\n'.format(json.dumps(acc_classify, ensure_ascii=True, indent=2)))
    f_out.write('{}\n'.format(json.dumps(acc_avg, ensure_ascii=True, indent=2)))
    f_out.write('***avg acc {}***\n'.format(acc_avg_all))
    print(acc_classify)
    print(acc_avg)
    print('***avg acc {}***\n'.format(acc_avg_all))
    f_out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--openai_key", type=str, default="xxx")
    parser.add_argument("--data_type", type=str, default="ceval")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--subject", "-s", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    assert args.data_type in ['ceval','cmmlu']
    if args.device is not None:
        print("seting up device",args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_name: str = args.model_name.lower()
    if model_name.startswith("baichuan"):
        if model_name.find('13b') != -1:
            from evaluate.baichuan2.prompt import EvaluateBuilder
            evaluator = EvaluateBuilder(choices, args.model_name, args.k)
        else:
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
    elif model_name.startswith("internlm"):
        from evaluate.internlm.prompt import EvaluateBuilder
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


    evaluator.init()
    subject_name = args.subject
    cur_path = os.path.dirname(__file__)
    log_dir = os.path.join(cur_path, f"logs_{args.data_type}")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_result_dir = os.path.join(log_dir, f"{args.model_name}_{'few_shot_' + str(args.k) if args.few_shot else 'zero_shot'}_{run_date}")
    os.mkdir(save_result_dir)

    if args.data_type == "ceval":
        eval_ceval(evaluator,subject_name,save_result_dir)
    else:
        eval_cmmlu(evaluator,subject_name,save_result_dir)