# -*- coding: utf-8 -*-
# @Time:  21:16
# @Author: tk
# @File：translate
import json
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))

from tqdm import tqdm
from auto_eval.mymetrics import compute_rouge_metric
from auto_eval.data_utils import build_rouge_data
from auto_eval.config import env_setting, get_registry_path, get_output_path, MODEL, FORCE_EVAL, get_output_path_metric


def do_eval(subjects,output_path,data_type):
    # 评估主题
    for subject in tqdm(subjects):
        print(subject)
        log_path = os.path.join(output_path,data_type,MODEL.rsplit('/')[-1])
        os.makedirs(log_path,exist_ok=True)
        log_file = os.path.join(log_path, subject + '.log')
        record_path = os.path.join(log_path, subject + '.event')
        if os.path.exists(record_path) and not FORCE_EVAL:
            continue
        run_string = 'exec_aigc_evals {} translate_{}_{} --debug=1 --registry_path={} --log_to_file={} --record_path={}'.format(
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
    output_path_metric = get_output_path_metric()
    data_path = r'../assets/rouge_data'

    data_type = "rouge"
    # 构建数据
    subjects = build_rouge_data(data_path, registry_path, data_type=data_type)
    do_eval(subjects,output_path,data_type)
    compute_rouge_metric(subjects,output_path,output_path_metric,data_type,MODEL)

