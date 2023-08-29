from typing import TYPE_CHECKING, Optional, Union, List, Dict

from aigc_evals.prompt.base import OpenAICreateChatPrompt

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


@dataclass
class ModelGradedSpec:
    # must have
    prompt: Union[str, OpenAICreateChatPrompt]
    choice_strings: Union[List[str], str]
    input_outputs: Dict[str, str]

    # optional
    eval_type: Optional[str] = None
    choice_scores: Optional[Union[Dict[str, float], str]] = None
    output_template: Optional[str] = None

    # unused
    key: Optional[str] = None
    group: Optional[str] = None
