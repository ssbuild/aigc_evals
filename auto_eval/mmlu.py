#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('curl -O https://people.eecs.berkeley.edu/~hendrycks/data.tar')
# get_ipython().system('tar -xf data.tar')

import os
import sys
sys.path.append(os.path.dirname(__file__))

from tqdm import tqdm
from auto_eval.mymetrics import compute_match_metric
from data_utils import build_mmlu_data
from config import env_setting, get_registry_path, get_output_path, MODEL, FORCE_EVAL


def do_eval(subjects,output_path,data_type):
    # 评估主题
    for subject in tqdm(subjects):
        print(subject)
        log_path = os.path.join(output_path,data_type)
        os.makedirs(log_path,exist_ok=True)
        log_file = os.path.join(log_path, subject + '.log')
        record_path = os.path.join(log_path, subject + '.event')
        if os.path.exists(record_path) and not FORCE_EVAL:
            continue
        run_string = 'exec_aigc_evals {} match_{}_{} --debug=1 --registry_path={} --log_to_file={} --record_path={}'.format(
            MODEL,
            data_type,
            subject,
            registry_path,
            log_file,
            record_path
        )
        #启动评估脚本
        ret = os.system(run_string)
        if ret != 0:
            break




if __name__ == '__main__':
    env_setting()

    registry_path = get_registry_path()
    output_path = get_output_path()
    data_path = r'F:\nlpdata_2023\openai\evals\data'

    data_type = "mmlu"
    # 构建数据
    subjects = build_mmlu_data(data_path, registry_path, data_type=data_type, few_shot=5)

    do_eval(subjects,output_path,data_type)
    compute_match_metric(subjects,output_path,data_type)

