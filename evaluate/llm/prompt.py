# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/11 10:40

import os
import re
import typing
from tqdm import tqdm
from evaluate.prompt.prompt import EvaluateBuilderBase
from evaluate.llm.infer import Engine_API

class EvaluateBuilder(EvaluateBuilderBase):
    def __init__(self,choices, model_name, k):
        super().__init__(choices, model_name, k)
        self.api_client: typing.Optional[Engine_API] = None

    def init(self):
        self.api_client = Engine_API()
        self.api_client.init(self.model_name)
        A = self.api_client.tokenizer.encode("A", add_special_tokens=False)[-1]
        B = self.api_client.tokenizer.encode("B", add_special_tokens=False)[-1]
        C = self.api_client.tokenizer.encode("C", add_special_tokens=False)[-1]
        D = self.api_client.tokenizer.encode("D", add_special_tokens=False)[-1]
        self.choices_ids = [A,B,C,D]

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, cot=False, save_result_dir=None):
        correct_num = 0
        if save_result_dir:
            if few_shot:
                response_list = []
            result = []
            score = []
        few_shot_prompt = self.generate_few_shot_prompt(
            subject_name, dev_df, cot=cot) if few_shot else ""
        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            if few_shot:
                full_prompt = few_shot_prompt + question
                response = self.api_client.infer(full_prompt, repetition_penalty=1.01, max_length=2048, do_sample=False)
                response = response.strip()
                ans, direct_extract = self.extract_cot_answer(row, response)
            else:  # zero-shot by extracting answer from distribution
                ans = self.generate_dist(question, do_sample=False, repetition_penalty=1.01, max_new_tokens=1)

            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            if save_result_dir:
                if few_shot:
                    response_list.append(response)
                result.append(ans)
                score.append(correct)
        correct_ratio = 100 * correct_num / len(answers)

        if save_result_dir:
            if few_shot:
                test_df['model_response'] = response_list
            test_df['model_output'] = result
            test_df['correctness'] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

        return correct_ratio

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                dev_df.iloc[i, :],
                include_answer=True,
                cot=cot
            )
        return prompt

    def format_example(self, line, include_answer=True, cot=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + \
                           line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += '\n答案：' + line["answer"] + '\n\n'
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += '\n答案：'
        return example

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

    def generate_dist(self, query, num_beams=1, 
                      do_sample=False, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):

        gen_kwargs = {"num_beams": num_beams, "do_sample": do_sample, "top_p": top_p, 
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        scores = self.api_client.infer(query, return_dict_in_generate=True, output_scores=True, **gen_kwargs)
        score = scores[0].tolist()

        choice_score = [score[self.choices_ids[0]], score[self.choices_ids[1]],
                        score[self.choices_ids[2]], score[self.choices_ids[3]]]
        ranked_index = [index for index, value in
                        sorted(list(enumerate(choice_score)), key=lambda x: x[1], reverse=True)]
        return self.choices[ranked_index[0]]