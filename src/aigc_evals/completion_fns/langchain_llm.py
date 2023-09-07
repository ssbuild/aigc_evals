import importlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable, List

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.chat_models import ChatOpenAI


from aigc_evals.prompt.base import CompletionPrompt, is_chat_prompt, OpenAICreateChatPrompt
# from aigc_evals.record import record_sampling


class CompletionResult(ABC):
    @abstractmethod
    def get_completions(self) -> List[str]:
        pass

@runtime_checkable
class CompletionFn(Protocol):
    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> CompletionResult:
        """
        ARGS
        ====
        `prompt`: Either a `Prompt` object or a raw prompt that will get wrapped in
            the appropriate `Prompt` class.
        `kwargs`: Other arguments passed to the API.

        RETURNS
        =======
        The result of the API call.
        The prompt that was fed into the API call as a str.
        """


class LangChainLLMCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> List[str]:
        return [self.response.strip()]


class LangChainLLMCompletionFn(CompletionFn):
    def __init__(self, llm: str,  **kwargs) -> None:
        # Import and resolve self.llm to an instance of llm argument here,
        # assuming it's always a subclass of BaseLLM
        model_kwargs = kwargs.get('chat_model_kwargs',None) or kwargs.get('llm_kwargs',None) or {}
        module = importlib.import_module("langchain.llms")
        LLMClass = getattr(module, llm)

        if issubclass(LLMClass, BaseLLM):
            self.llm = LLMClass(**kwargs)
        else:
            raise ValueError(f"{llm} is not a subclass of BaseLLM")

    def __call__(self, prompt, **kwargs) -> LangChainLLMCompletionResult:
        prompt = CompletionPrompt(prompt).to_formatted_prompt()
        response = self.llm(prompt)
        # record_sampling(prompt=prompt, sampled=response)
        return LangChainLLMCompletionResult(response)


def _convert_dict_to_langchain_message(_dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict["content"] or ""  # OpenAI returns None for tool invocations
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


class LangChainChatModelCompletionFn(CompletionFn):
    def __init__(self, llm: str,  **kwargs) -> None:
        # Import and resolve self.llm to an instance of llm argument here,
        # assuming it's always a subclass of BaseLLM
        model_kwargs = kwargs.get('chat_model_kwargs',None) or kwargs.get('llm_kwargs',None) or {}
        module = importlib.import_module("langchain.chat_models")
        LLMClass = getattr(module, llm)

        if issubclass(LLMClass, BaseChatModel):
            self.llm = LLMClass(**model_kwargs)
        else:
            raise ValueError(f"{llm} is not a subclass of BaseChatModel")

    def __call__(self, prompt, **kwargs) -> LangChainLLMCompletionResult:
        if is_chat_prompt(prompt):
            messages = [_convert_dict_to_langchain_message(message) for message in prompt]
        else:
            messages = [HumanMessage(content=prompt)]
        response = self.llm(messages).content
        # record_sampling(prompt=prompt, sampled=response)
        return LangChainLLMCompletionResult(response)
