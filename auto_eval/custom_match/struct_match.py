# -*- coding: utf-8 -*-
# @Time:  19:04
# @Author: tk
# @File：struct_match
import json
from typing import Any, List, Union, Tuple, Callable, Optional, Dict

import aigc_evals
import aigc_evals.metrics
from aigc_evals.api import CompletionFn
from aigc_evals.prompt.base import is_chat_prompt
from aigc_evals.record import record_match


class Element(object):
    def __init__(self,value: str):
        self.key,self.value  = value.split('_',1)

    def __eq__(self, other):
        return self.key == other.key and self.value == other.value

class StructMatch(aigc_evals.Eval):
    def __init__(
        self,
        completion_fns: List[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        label_file=None,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Match only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

        if label_file is not None:
            with open(label_file,mode='r',encoding='utf-8') as f:
                labels = f.readlines()
            labels = [l.replace('\r\n','').replace('\n','') for l in labels if l]
            labels = [l for l in labels if l]
            self.labels = list(sorted(labels))
        else:
            self.labels = None

    def _evaluate(self,
                  sample: Dict,
                  expect: Dict,
                  ):
        """评测函数
        """
        R,T = [],[]
        for k,v in sample.items():
            RESULT = R
            if isinstance(v, dict):
                RESULT.extend([k + _ for _ in list(v.values()) if _])
            elif isinstance(v, list):
                for _ in v:
                    if not isinstance(_,dict):
                        continue
                    RESULT.extend([k + '_' +  value for value in list(_.values()) if value])
            else:
                if v:
                    RESULT.append(k + '_' + v)

        for k, v in expect.items():
            RESULT = T
            if isinstance(v, dict):
                RESULT.extend([k + _ for _ in list(v.values()) if _])
            elif isinstance(v, list):
                for _ in v:
                    if not isinstance(_, dict):
                        continue
                    RESULT.extend([k + '_' + value for value in list(_.values()) if value])
            else:
                if v:
                    RESULT.append(k + '_' + v)

        R = set([Element(i) for i in R])
        T = set([Element(i) for i in T])
        tp = len(R & T)
        fp = len(R) - tp
        fn = len(T) - tp

        return tp, fp, fn

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"
        assert isinstance(sample["ideal"], dict) or isinstance(
            sample["ideal"], dict
        ), "sample['ideal'] must be dict"

        prompt = sample["input"]


        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]

        try:
            jd = json.loads(sampled.strip())

        except:
            jd = {}
            pass

        metric = {}
        tp_all, fp_all, fn_all = 0,0,0
        if not self.labels:
            tp_all, fp_all, fn_all = self._evaluate(sample=jd, expect=sample["ideal"])
        else:
            for label in self.labels:
                tp, fp, fn = self._evaluate(sample=jd.get(label,{}), expect=sample["ideal"].get(label,{}))
                metric[label] = (tp, fp, fn)
                tp_all += tp
                fp_all += fp
                fn_all += fn

        result = {
            "index" : sample.get("id",None),
            # "prompt": prompt,
            "sampled": sampled,
            "metric_all": json.dumps((tp_all, fp_all, fn_all),ensure_ascii=False),
            "metric": json.dumps(metric,ensure_ascii=False)
        }
        result["expected"] = json.dumps(sample["ideal"],ensure_ascii=False)
        aigc_evals.record.record_sampling(prompt, sampled, index=sample.get("id",None))
        aigc_evals.record.record_metrics(**result)

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("metrics")
        tp = sum(int(json.loads(event.data["metric_all"])[0]) for event in events)
        fp = sum(int(json.loads(event.data["metric_all"])[1]) for event in events)
        fn = sum(int(json.loads(event.data["metric_all"])[2]) for event in events)
        precision,recall = tp / (tp + fp + 1e-10),tp / (tp + fn + 1e-10)
        f1 = 2*(precision*recall)/(precision+recall +1e-10)

        metric_avg = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if self.labels is not None:
            result = {k: [0, 0, 0] for k in self.labels}
            metric = {k: {
                "precision": 0,
                "recall": 0,
                "f1": 0,
            } for k in self.labels}

            events = recorder.get_events("metrics")
            for event in events:
                for k,(tp,fp,fn) in json.loads(event.data["metric"]).items():
                    result[k][0] += tp
                    result[k][1] += fp
                    result[k][2] += fn

            for k in metric:
                tp,fp,fn = result[k][0],result[k][1],result[k][2]
                precision, recall = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                metric[k]["precision"] = precision
                metric[k]["recall"] = recall
                metric[k]["f1"] = f1
        else:
            metric = None

        return {
            "metric_avg" : metric_avg,
            "metric": metric
        }
