# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/30 17:09
import re
from typing import Any, Union, List, Tuple, Callable, Optional
from aigc_evals.elsuite.basic.match import Match
from aigc_evals.prompt.base import is_chat_prompt
from aigc_evals.record import record_match, record_sampling


def extract_answer(gen_ans):
    m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
    if len(m) > 0:
        return m[-1]
    answer_patterns = [
        r'([ABCD])\s*是正确的',
        r'选项\s*([ABCD])正确',
        r'答案为\s*([ABCD])',
        r'答案是\s*([ABCD])',
        r'答案\s*([ABCD])',
        r'选择\s*([ABCD])',
        r'答案：\s*([ABCD])',
        r'选择答案\s*([ABCD])',
        r'正确答案为\s*([ABCD])',
        r'正确答案应该是\s*([ABCD])',
    ]
    # RE extraction
    for answer_pattern in answer_patterns:
        m = re.search(answer_pattern, gen_ans, re.M)
        if m:
            answer = m.group(1)
            return answer
    # only containing one choice-character
    m = re.findall(r'[ABCD]', gen_ans, re.M)
    if len(m) == 1:
        answer = m[0]
        return answer
    return None

def choice_record_and_match(
    prompt: Any,
    sampled: str,
    expected: Union[str, List[str], Tuple[str]],
    separator: Callable[[str], bool] = None,
    options: Optional[List[str]] = None,
):
    """
    Records and checks if a sampled response from a CompletionFn matches the expected result.

    Args:
        prompt: The input prompt.
        sampled: The sampled response from the model.
        expected: The expected response or list of responses.
        separator: Optional function to check if a character is a separator.
        options: Optional list of options to match against the sampled response.

    Returns:
        The matched option or None if no match found.
    """

    if isinstance(expected, tuple):
        expected = list(expected)
    elif isinstance(expected, str) and expected.startswith('[') and expected.endswith(']'):
        expected = eval(expected)
    elif not isinstance(expected, list):
        expected = [expected]


    if options is None:
        options = expected

    picked = None
    for option in options:
        if not sampled.startswith(option):
            continue
        if (
            separator is not None
            and len(sampled) > len(option)
            and not separator(sampled[len(option)])
        ):
            continue
        picked = option
        break
    if picked is None:
        picked = extract_answer(sampled)
    match = picked in expected
    record_sampling(prompt,sampled)
    record_match(match, expected=expected, picked=picked,options=options)
    return picked


class ChoiceMatch(Match):
    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert "input" in sample, "sample must have an 'input' key"
        assert "ideal" in sample, "sample must have an 'ideal' key"
        assert isinstance(sample["ideal"], str) or isinstance(
            sample["ideal"], list
        ), "sample['ideal'] must be a string or list of strings"

        prompt = sample["input"]
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]
        return choice_record_and_match(
            prompt=prompt,
            sampled=sampled,
            expected=sample["ideal"],
        )

if __name__ =='__main__':
    gen_ans = "根据中国《建筑设计防火规范》（GB50016-2014）的规定，两座多层建筑，其防火间距可以按规定减少 25% 的条件是相邻两面外墙均为不燃性墙体，且无外露的可燃性屋檐，每面外墙上无防火保护的门、窗、洞口不正对开设，同时门、窗、洞口面积之和各不大于该外墙面积的 6%。因此，正确答案为C. 6%。\n</s>"
    d = extract_answer(gen_ans=gen_ans)

    print(d)