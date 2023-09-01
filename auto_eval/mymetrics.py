# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/1 9:30
import json
import os
import pandas as pd


def compute_match_metric(subjects,output_path,data_type,model):
    acc_all = {}
    for subject in subjects:
        log_path = os.path.join(output_path,data_type,model.rsplit('/')[-1])
        record_path = os.path.join(log_path, subject + '.event')
        if not os.path.exists(record_path):
            continue
        with open(record_path, "r") as f:
            events_df = pd.read_json(f, lines=True)

        # matches_df = events_df[events_df.type == "match"].reset_index(drop=True)
        # matches_df = matches_df.join(pd.json_normalize(matches_df.data))
        # matches_df.correct.value_counts().plot.bar(title="Correctness of generated answers", xlabel="Correctness", ylabel="Count")

        # "correct": true, "expected": "C", "picked": "C", "sampled": "慢性胃溃疡最常见的并发症是出血。\n\n解题过程：\n\n根据对慢性胃溃疡的了解，它是一种消化性溃疡，常常会引起上消化道出血。出血是溃疡最常见的并发症，发生率为25%～30%。出血可以表现为呕血、黑便、咯血，严重者甚至会引起失血性休克。\n\n分析选项：\n\nA. 幽门狭窄：幽门狭窄是胃溃疡的严重并发症，但不是最常见的。\n\nB. 穿孔：穿孔是胃溃疡的严重并发症，但不是最常见的。\n\nC. 出血：根据上述分析，出血是慢性胃溃疡最常见的并发症。\n\nD. 癌变：癌变是胃溃疡的严重并发症，但不是最常见的。\n\n综上所述，正确答案是C. 出血。", "options": [
        #     "C"]}, "created_by": "", "created_at": "2023-09-01 01:05:20.893104+00:00"}

        acc_num = 0
        total_num = 0
        for i, r in pd.json_normalize(events_df[events_df.type == "match"].data).iterrows():
            total_num += 1
            if r.correct:
                acc_num += 1
        acc_all[subject] = acc_num / total_num if total_num >0 else 0

    print(acc_all)
    with open(os.path.join(output_path,'metric.json'),mode='w',encoding='utf-8') as f:
        f.write(json.dumps(acc_all,ensure_ascii=False,indent=2))
    return acc_all


def compute_bleu_metric(subjects,output_path,data_type,model):
    acc_all = {}
    for subject in subjects:
        log_path = os.path.join(output_path,data_type,model.rsplit('/')[-1])
        record_path = os.path.join(log_path, subject + '.event')
        if not os.path.exists(record_path):
            continue
        with open(record_path, "r") as f:
            events_df = pd.read_json(f, lines=True)

        # {"run_id": "2308310700296J5KV3PJ", "event_id": 58, "sample_id": "translate_zh-en_samples.test.1", "type": "metrics", "data": {"sacrebleu_sentence_score": 1.958224832501124}, "created_by": "", "created_at": "2023-08-31 07:03:03.500952+00:00"}


        total_num = 0
        bleu = 0
        for i, r in pd.json_normalize(events_df[events_df.type == "metrics"].data).iterrows():
            total_num += 1
            bleu += r.sacrebleu_sentence_score

        acc_all[subject] = bleu / total_num if total_num >0 else 0

    print(acc_all)
    with open(os.path.join(output_path,'metric.json'),mode='w',encoding='utf-8') as f:
        f.write(json.dumps(acc_all,ensure_ascii=False,indent=2))

    return acc_all