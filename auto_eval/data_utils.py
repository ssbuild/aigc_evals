# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/30 12:39
import json
import os
import pandas as pd
import yaml


def build_ceval_data(data_path,registry_path,data_type = "ceval",few_shot=5):
    choices = ["A", "B", "C", "D"]
    sys_msg = "以下是中国关于{}考试的单项选择题，请选出其中的正确答案。\n\n"
    def create_chat_prompt(sys_msg, question, answers, subject):
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)]) + "\n答案："
        return [
            {"role": "system", "content": sys_msg.format(subject)},
            {"role": "user", "content": user_prompt}
        ]

    def create_chat_example(question, answers, correct_answer):
        """
        Form few-shot prompts in the recommended format: https://github.com/openai/openai-python/blob/main/chatml.md#few-shot-prompting
        """
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)]) + "\n答案："
        return [

            {"role": "user", "content": user_prompt},
            {"role": "system", "content": correct_answer},
        ]

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_path, "test")) if "_test.csv" in f])
    print(subjects)

    registry_yaml = {}

    for subject in subjects:
        subject_path = os.path.join(registry_path, "data", data_type, subject)
        os.makedirs(subject_path, exist_ok=True)

        #id,question,A,B,C,D,answer,explanation
        # Create few-shot prompts

        if few_shot > 0:
            dev_df = pd.read_csv(os.path.join(data_path, "dev", subject + "_dev.csv"),header=0, names=("id","question", "A", "B", "C", "D", "answer","explanation"))
            dev_df["sample"] = dev_df.apply(lambda x: create_chat_example(x["question"], x[["A", "B", "C", "D"]], x["answer"]), axis=1)
            few_shot_path = os.path.join(subject_path, "few_shot.jsonl")
            dev_df[["sample"]].head(few_shot).to_json(few_shot_path, lines=True, orient="records",force_ascii=False)

        # Create test prompts and ideal completions
        test_df = pd.read_csv(os.path.join(data_path, "val", subject + "_val.csv"),header=0, names=("id","question", "A", "B", "C", "D", "answer"))
        test_df["input"] = test_df.apply(lambda x: create_chat_prompt(sys_msg, x["question"], x[["A", "B", "C", "D"]], subject), axis=1)
        test_df["ideal"] = test_df.answer
        samples_path = os.path.join(subject_path, "samples.jsonl")
        test_df[["input", "ideal"]].to_json(samples_path, lines=True, orient="records",force_ascii=False)

        eval_id = f"match_{data_type}_{subject}"

        registry_yaml[eval_id] = {
            "id": f"{eval_id}.test.v1",
            "metrics": ["accuracy"]
        }

        d = {
            "class": "custom_match.choice_match:ChoiceMatch",
            "args": {
                "samples_jsonl": samples_path,
                # "few_shot_jsonl": few_shot_path,
                "num_few_shot": 4,
            }
        }
        if few_shot > 0:
            d["args"]["few_shot_jsonl"] = few_shot_path
        registry_yaml[f"{eval_id}.test.v1"] = d

    with open(os.path.join(registry_path, "evals", data_type + ".yaml"), "w") as f:
        yaml.dump(registry_yaml, f)

    return subjects


def build_cmmlu_data(data_path,registry_path,data_type = "cmmlu",few_shot=True):
    choices = ["A", "B", "C", "D"]
    sys_msg = "以下是中国关于{}考试的单项选择题，请选出其中的正确答案。\n\n"
    def create_chat_prompt(sys_msg, question, answers, subject):
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)]) + "\n答案："
        return [
            {"role": "system", "content": sys_msg.format(subject)},
            {"role": "user", "content": user_prompt}
        ]

    def create_chat_example(question, answers, correct_answer):
        """
        Form few-shot prompts in the recommended format: https://github.com/openai/openai-python/blob/main/chatml.md#few-shot-prompting
        """
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)]) + "\nAnswer:"
        return [
            # {"role": "system", "content": user_prompt, "name": "example_user"},
            # {"role": "system", "content": correct_answer, "name": "example_assistant"},

            {"role": "user", "content": user_prompt},
            {"role": "system", "content": correct_answer},
        ]




    subjects = sorted([f.split(".csv")[0] for f in os.listdir(os.path.join(data_path, "test")) if ".csv" in f])
    print(subjects)
    registry_yaml = {}

    for subject in subjects:
        subject_path = os.path.join(registry_path, "data", data_type, subject)
        os.makedirs(subject_path, exist_ok=True)

        # Create few-shot prompts
        if few_shot>0:
            dev_df = pd.read_csv(os.path.join(data_path, "dev", subject + ".csv"),header=0, names=("Question", "A", "B", "C", "D", "Answer"))
            dev_df["sample"] = dev_df.apply(lambda x: create_chat_example(x["Question"], x[["A", "B", "C", "D"]], x["Answer"]), axis=1)
            few_shot_path = os.path.join(subject_path, "few_shot.jsonl")
            dev_df[["sample"]].head(few_shot).to_json(few_shot_path, lines=True, orient="records",force_ascii=False)

        # Create test prompts and ideal completions
        test_df = pd.read_csv(os.path.join(data_path, "test", subject + ".csv"), header=0, names=("Question", "A", "B", "C", "D", "Answer"))
        test_df["input"] = test_df.apply(lambda x: create_chat_prompt(sys_msg, x["Question"], x[["A", "B", "C", "D"]], subject), axis=1)
        test_df["ideal"] = test_df.Answer
        samples_path = os.path.join(subject_path, "samples.jsonl")
        test_df[["input", "ideal"]].to_json(samples_path, lines=True, orient="records",force_ascii=False)

        eval_id = f"match_{data_type}_{subject}"

        registry_yaml[eval_id] = {
            "id": f"{eval_id}.test.v1",
            "metrics": ["accuracy"]
        }
        d = {
            "class": "custom_match.choice_match:ChoiceMatch",
            "args": {
                "samples_jsonl": samples_path,
                # "few_shot_jsonl": few_shot_path,
                "num_few_shot": 4,
            }
        }
        if few_shot>0:
            d["args"]["few_shot_jsonl"] = few_shot_path
        registry_yaml[f"{eval_id}.test.v1"] = d

    with open(os.path.join(registry_path, "evals", data_type + ".yaml"), "w") as f:
        yaml.dump(registry_yaml, f)

    return subjects



def build_mmlu_data(data_path,registry_path,data_type = "mmlu",few_shot=True):
    choices = ["A", "B", "C", "D"]
    sys_msg = "The following are multiple choice questions (with answers) about {}."
    def create_chat_prompt(sys_msg, question, answers, subject):
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)]) + "\nAnswer:"
        return [
            {"role": "system", "content": sys_msg.format(subject)},
            {"role": "user", "content": user_prompt}
        ]

    def create_chat_example(question, answers, correct_answer):
        """
        Form few-shot prompts in the recommended format: https://github.com/openai/openai-python/blob/main/chatml.md#few-shot-prompting
        """
        user_prompt = f"{question}\n" + "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, answers)]) + "\nAnswer:"
        return [
            # {"role": "system", "content": user_prompt, "name": "example_user"},
            # {"role": "system", "content": correct_answer, "name": "example_assistant"},

            {"role": "user", "content": user_prompt},
            {"role": "system", "content": correct_answer},
        ]


    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_path, "test")) if "_test.csv" in f])
    print(subjects)

    registry_yaml = {}

    for subject in subjects:
        subject_path = os.path.join(registry_path, "data", data_type, subject)
        os.makedirs(subject_path, exist_ok=True)

        if few_shot >0:
            # Create few-shot prompts
            dev_df = pd.read_csv(os.path.join(data_path, "dev", subject + "_dev.csv"), names=("Question", "A", "B", "C", "D", "Answer"))
            dev_df["sample"] = dev_df.apply(lambda x: create_chat_example(x["Question"], x[["A", "B", "C", "D"]], x["Answer"]), axis=1)
            few_shot_path = os.path.join(subject_path, "few_shot.jsonl")
            dev_df[["sample"]].head(few_shot).to_json(few_shot_path, lines=True, orient="records")

        # Create test prompts and ideal completions
        test_df = pd.read_csv(os.path.join(data_path, "test", subject + "_test.csv"), names=("Question", "A", "B", "C", "D", "Answer"))
        test_df["input"] = test_df.apply(lambda x: create_chat_prompt(sys_msg, x["Question"], x[["A", "B", "C", "D"]], subject), axis=1)
        test_df["ideal"] = test_df.Answer
        samples_path = os.path.join(subject_path, "samples.jsonl")
        test_df[["input", "ideal"]].to_json(samples_path, lines=True, orient="records")

        eval_id = f"match_{data_type}_{subject}"

        registry_yaml[eval_id] = {
            "id": f"{eval_id}.test.v1",
            "metrics": ["accuracy"]
        }
        d = {
            "class": "custom_match.choice_match:ChoiceMatch",
            "args": {
                "samples_jsonl": samples_path,
            }
        }
        registry_yaml[f"{eval_id}.test.v1"] = d

    with open(os.path.join(registry_path, "evals", data_type + ".yaml"), "w") as f:
        yaml.dump(registry_yaml, f)

    return subjects










def build_translate_data(data_path,registry_path,data_type="zh-en"):
    sys_msg = "给定一个表示的文本，提供该文本的英文翻译。除了翻译本身之外，您必须在输出中提供任何解释。您必须释义而不是逐字翻译，并保留所有的原始含义。"
    def create_chat_prompt(sys_msg, question):
        user_prompt = f"{question}"
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt}
        ]

    subjects = []

    registry_yaml = {}

    subject_path = os.path.join(registry_path, "data", data_type)
    os.makedirs(subject_path, exist_ok=True)

    fs_list = os.listdir(data_path)
    for file in fs_list:
        if file.endswith('.json') or file.endswith('.JSON'):
            subject = file.rsplit('.json')[0]
            subjects.append(subject)
            with open(os.path.join(data_path,file),mode='r',encoding='utf=-8') as f:
                lines = f.readlines()

            D = []
            for line in lines:
                jd = json.loads(line)
                if not jd:
                    continue
                D.append({
                    "input": create_chat_prompt(sys_msg,jd["src"]),
                    "ideal": jd["dst"] if isinstance(jd["dst"],list) else [jd['dst']]
                })
            samples_path = os.path.join(subject_path, "samples.jsonl")
            test_df = pd.DataFrame(D)
            test_df[["input", "ideal"]].to_json(samples_path, lines=True, orient="records",force_ascii=False)

            eval_id = f"translate_{data_type}_{subject}"

            registry_yaml[eval_id] = {
                "id": f"{eval_id}.test.v1",
                "metrics": ["accuracy"]
            }
            d = {
                "class": "aigc_evals.elsuite.translate:Translate",
                "args": {
                    "samples_jsonl": samples_path,
                }
            }
            registry_yaml[f"{eval_id}.test.v1"] = d


    with open(os.path.join(registry_path, "evals", data_type + ".yaml"), "w") as f:
        yaml.dump(registry_yaml, f)

    return subjects