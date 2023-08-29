import aigc_evals
import aigc_evals.metrics
from aigc_evals.api import CompletionFn
from aigc_evals.record import RecorderBase
from datasets import load_dataset

class Lambada(aigc_evals.Eval):
    def __init__(
        self,
        completion_fns: List[CompletionFn],
        subset: str,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "MultipleChoice only supports one completion fn"
        self.subset = subset

    def eval_sample(self, sample, rng):
        del rng

        words = sample["text"].split()
        context = " ".join(words[:-1])
        a = words[-1]

        prompt = "Please answer with the word which is most likely to follow:" + "\n\n" + context
        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
            max_tokens=8,
        )
        sampled = result.get_completions()[0]

        aigc_evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=a,
        )


    def run(self, recorder: RecorderBase):
        samples = load_dataset("EleutherAI/lambada_openai", self.subset, split="test")
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": aigc_evals.metrics.get_accuracy(recorder.get_events("match")),
        }
