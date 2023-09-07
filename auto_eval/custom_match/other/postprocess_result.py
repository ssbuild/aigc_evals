# -*- coding: utf-8 -*-
# @Time:  18:16
# @Author: tk
# @File：reprocess_result
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import pandas as pd
from tqdm import tqdm
from auto_eval.config import env_setting, get_registry_path, get_output_path, get_output_path_metric, MODEL
from auto_eval.data_utils import build_ceval_data
from auto_eval.custom_match.choice_match import extract_answer


def rematch(subjects, output_path, new_data_path, data_type, MODEL):
    for subject in subjects:
        log_path = os.path.join(output_path, data_type, MODEL.rsplit('/')[-1])
        record_path = os.path.join(log_path, subject + '.event')
        if not os.path.exists(record_path):
            continue
        with open(record_path, "r") as f:
            events_df = pd.read_json(f, lines=True)

        for idx, d in events_df.iterrows():
            if d["type"] == "match":
                data = d["data"]
                if data["picked"] is None:
                    data["picked"] = extract_answer(data["sampled"])



        record_path_new = os.path.join(new_data_path, subject + '.event')
        events_df.to_json(record_path_new, lines=True, orient="records",force_ascii=False)



if __name__ == '__main__':
    env_setting()

    registry_path = get_registry_path()
    output_path = get_output_path()
    output_path_metric = get_output_path_metric()
    # 数据路径
    data_path = r'../../../assets/ceval_data'

    data_type = "ceval"

    new_data_path = '/home/test/aigc_evals/outputs/ceval/llama2-70b-hf-int4-new'
    # 构建数据
    subjects = build_ceval_data(data_path, registry_path, data_type=data_type, few_shot=5)




    rematch(subjects, output_path, new_data_path, data_type, MODEL)