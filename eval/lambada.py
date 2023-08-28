#!/usr/bin/env python
# coding: utf-8

# ## Building an eval for LAMBADA
# 
# We show how to build an eval for the LAMBADA dataset

# In[ ]:


# Download LAMBADA from https://zenodo.org/record/2630551 and place in examples/lambada-dataset
get_ipython().system('curl -O https://zenodo.org/record/2630551/files/lambada-dataset.tar.gz')
get_ipython().system('tar -xzf lambada-dataset.tar.gz --one-top-level')
get_ipython().system('ls lambada-dataset')
import os
import pandas as pd

registry_path = os.path.join("..", "evals", "registry")
os.makedirs(os.path.join(registry_path, "data", "lambada"), exist_ok=True)

def create_chat_prompt(text):
    return [
        {"role": "system", "content": "Please complete the passages with the correct next word."}, 
        {"role": "user", "content": text}
    ]

df = pd.read_csv('lambada-dataset/lambada_test_plain_text.txt', sep="\t", names=["text"])
df["text"] = df["text"].str.split(" ")
df["input"], df["ideal"] = df["text"].str[:-1].str.join(" ").apply(create_chat_prompt), df["text"].str[-1]
df = df[["input", "ideal"]]
df.to_json(os.path.join(registry_path, "data/lambada/samples.jsonl"), orient="records", lines=True)
display(df.head())

eval_yaml = """
lambada:
  id: lambada.test.v1
  metrics: [accuracy]
lambada.test.v1:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: lambada/samples.jsonl
""".strip()
with open(os.path.join(registry_path, "evals", "lambada.yaml"), "w") as f:
    f.write(eval_yaml)


# In[ ]:


get_ipython().system('oaieval gpt-3.5-turbo lambada --max_samples 20')


# In[ ]:


# Inspect samples
log_path = None # Set to jsonl path to logs from oaieval
events = f"/tmp/evallogs/{log_path}"
with open(events, "r") as f:
    events_df = pd.read_json(f, lines=True)
for i, r in pd.json_normalize(events_df[events_df.type == "sampling"].data).iterrows():
    print(r)
    print(f"Prompt: {r.prompt}")
    print(f"Sampled: {r.sampled}")
    print("-" * 25)

