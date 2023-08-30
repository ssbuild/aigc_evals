import json
from typing import Any, List

import aigc_evals
import aigc_evals.metrics
import aigc_evals.record
from aigc_evals.api import CompletionFn


def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


class JsonValidator(aigc_evals.Eval):
    def __init__(
        self,
        completion_fns: List[CompletionFn],
        samples_jsonl: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "JsonValidator only supports one completion fn"
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"

        prompt = sample["input"]
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]
        return aigc_evals.record.record_match(is_valid_json(sampled), expected=None, picked=sampled)

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": aigc_evals.metrics.get_accuracy(events),
            "boostrap_std": aigc_evals.metrics.get_bootstrap_accuracy_std(events),
        }
