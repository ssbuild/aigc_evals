from typing import Optional, List
from urllib.parse import parse_qs, urlparse
from pydantic import BaseModel
import aigc_evals
import aigc_evals.metrics
from aigc_evals.api import CompletionFn
from aigc_evals.formatting import make_abc
from aigc_evals.record import RecorderBase
from datasets import load_dataset

class Sample(BaseModel):
    question: str
    answers: List[str]
    label: int

def get_dataset(url: str) -> List[Sample]:
    parsed = urlparse(url)
    if parsed.scheme == "hf":
        query = parse_qs(parsed.query)
        query = {k: v[0] for k, v in query.items()}

        path = parsed.netloc

        dataset = load_dataset(path, **query)
        if path == "hellaswag":
            return [
                Sample(
                    question=sample["ctx"],
                    answers=sample["endings"],
                    label=int(sample["label"]),
                )
                for sample in dataset
            ]
        elif path == "hendrycks_test":
            return [
                Sample(
                    question=sample["question"],
                    answers=sample["choices"],
                    label=sample["answer"],
                )
                for sample in dataset
            ]

    raise ValueError(f"Unknown question dataset {url}")

class MultipleChoice(aigc_evals.Eval):
    def __init__(
        self,
        completion_fns: List[CompletionFn],
        dataset: str,
        *args,
        instructions: Optional[str] = "",
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "MultipleChoice only supports one completion fn"
        self.dataset = dataset
        self.instructions = instructions

    def eval_sample(self, sample, rng):
        assert isinstance(sample, Sample)

        options, correct_answer = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=rng,
        )

        prompt = self.instructions + "\nPlease answer with the letter of the correct answer." + "\n\n" + sample.question + "\n" + options
        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1,
        )
        sampled = result.get_completions()[0]

        aigc_evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=correct_answer,
        )


    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.dataset)
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": aigc_evals.metrics.get_accuracy(recorder.get_events("match")),
        }
