# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/11 10:41

import os
import re
import typing

import torch
from tqdm import tqdm
from evaluate.prompt.prompt import EvaluateBuilderBase
from evaluate.moss.infer import Engine_API

class EvaluateBuilder(EvaluateBuilderBase):
    def __init__(self,choices, model_name, k):
        super().__init__(choices, model_name, k)
        self.api_client: typing.Optional[Engine_API] = None

    def init(self):
        self.api_client = Engine_API()
        self.api_client.init(self.model_name)
        A = self.api_client.tokenizer.encode("A", add_special_tokens=False)[0]
        B = self.api_client.tokenizer.encode("B", add_special_tokens=False)[0]
        C = self.api_client.tokenizer.encode("C", add_special_tokens=False)[0]
        D = self.api_client.tokenizer.encode("D", add_special_tokens=False)[0]
        self.choices_ids = [A,B,C,D]

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None, cot=False):
        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            few_shot_prompt = f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n"
        answers = list(test_df['answer'])
        message_list = []
        tar_list = []
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + "<|Human|>: " + question[0]['content'] + " <eoh>\n<|MOSS|>:"
            message_list.append(full_prompt)
            tar_list.append(answers[row_index])
            if len(message_list) % 1 == 0 or row_index == len(test_df) - 1:
                inputs = self.api_client.tokenizer(message_list, return_tensors="pt", padding=True)
                for k in inputs:
                    inputs[k] = inputs[k].cuda()
                max_tok = 2040 - inputs.input_ids.shape[1]
                outputs = self.api_client.generate(**inputs, do_sample=True, temperature=0.2, top_p=0.8,
                                              repetition_penalty=1.02,
                                              max_new_tokens=max_tok)
                input_len = torch.max(torch.sum(inputs.attention_mask, axis=1))
                response_list = [
                    self.api_client.tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
                    for i in range(outputs.shape[0])
                ]
                for i, response_str in enumerate(response_list):
                    # print(response_str)
                    if cot:
                        ans_list = re.findall(r"答案是(.+?)。", response_str)
                        if len(ans_list) == 0:
                            ans_list = re.findall(r"答案为(.+?)。", response_str)
                        if len(ans_list) == 0:
                            ans_list = re.findall(r"选项(.+?)是正确的。", response_str)

                        if len(ans_list) == 0:
                            correct = 0
                        else:
                            if self.exact_match(ans_list[-1], tar_list[i]):
                                correct_num += 1
                                correct = 1
                            else:
                                correct = 0
                    else:
                        response_str = response_str.strip()
                        if few_shot:
                            if self.exact_match(response_str, tar_list[i]):
                                correct_num += 1
                                correct = 1
                            else:
                                correct = 0
                        else:
                            if response_str[0] == tar_list[i]:
                                correct_num += 1
                                correct = 1
                            else:
                                correct = 0

                    if save_result_dir:
                        result.append(response_str)
                        score.append(correct)
                message_list = []
                tar_list = []

        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            test_df['model_output'] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'), encoding="utf-8", index=False)
        return correct_ratio

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
            if i == 0:
                tmp[0]["content"] = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n" + tmp[0][
                    "content"]
            user = tmp[0]['content']
            moss = tmp[1]['content']
            prompt += f"<|Human|>: {user} <eoh>\n<|MOSS|>: {moss} <eom>\n"
        return prompt

    def format_example(self, line, include_answer=True, cot=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += '\n答案：'
        if include_answer:
            if cot:
                ans = line["answer"]
                content = "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{ans}。"
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": content}
                ]
            else:
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": line["answer"]}
                ]
        else:
            return [
                {"role": "user", "content": example},
            ]

    def extract_cot_answer(self, line, gen_ans):
        m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True
        answer_patterns = [
            r'([ABCD])是正确的',
            r'选项([ABCD])正确',
            r'答案为([ABCD])',
            r'答案是([ABCD])',
            r'答案([ABCD])',
            r'选择([ABCD])',
            r'答案：([ABCD])',
            r'选择答案([ABCD])'
        ]
        # RE extraction
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r'[ABCD]', gen_ans, re.M)
        if len(m) == 1:
            answer = m[0]
            return answer, False
        answer_word_counter = 0
        # only containing one choice-context
        for c in self.choices:
            if str(line[f'{c}']) in gen_ans:
                answer = c
                answer_word_counter += 1
        if answer_word_counter == 1:
            return answer, False
        return '-', False

    def generate_dist(self, query, num_beams=1, max_length=2048,
                      do_sample=False, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):

        gen_kwargs = {"num_beams": num_beams, "do_sample": do_sample, "top_p": top_p, "max_length": max_length,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        scores = self.api_client.infer(query, return_dict_in_generate=True, output_scores=True, **gen_kwargs)
        score = scores[0].tolist()

        choice_score = [score[self.choices_ids[0]], score[self.choices_ids[1]],
                        score[self.choices_ids[2]], score[self.choices_ids[3]]]
        ranked_index = [index for index, value in
                        sorted(list(enumerate(choice_score)), key=lambda x: x[1], reverse=True)]
        return self.choices[ranked_index[0]]