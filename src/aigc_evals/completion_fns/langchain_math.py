import importlib
from typing import Optional, List

from openai import Completion
from aigc_evals.api import CompletionResult

from langchain import OpenAI, LLMMathChain

from aigc_evals.prompt.base import CompletionPrompt
# from aigc_evals.record import record_sampling


class LangChainCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> List[str]:
        return [self.response.strip()]


class LangChainMathChainCompletionFn(Completion):
    def __init__(self, **kwargs) -> None:
        llm = OpenAI(temperature=0)
        self.llm_math = LLMMathChain(llm=llm)

    def __call__(self, prompt, **kwargs) -> LangChainCompletionResult:

        prompt = CompletionPrompt(prompt).to_formatted_prompt()
        response = self.llm_math.run(prompt)
        # The LangChain response comes with `Answer: ` ahead of this, let's strip it out
        response = response.strip("Answer:").strip()
        # record_sampling(prompt=prompt, sampled=response)
        return LangChainCompletionResult(response)
