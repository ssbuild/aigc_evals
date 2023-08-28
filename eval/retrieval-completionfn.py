#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
We show here how to use the retrieval completion function to add context from documents to any OpenAI Evals task
The toy example here will be to augment our Born-First task with a dataset of presidential birthdays
"""

# Download the dataset manually, or use curl
get_ipython().system('curl -O https://people.math.sc.edu/Burkardt/datasets/presidents/president_birthdays.csv')


# In[ ]:


import os
import openai
import pandas as pd

df = pd.read_csv("president_birthdays.csv").rename(columns={" \"Name\"": "Name", " \"Month\"": "Month", " \"Day\"": "Day", " \"Year\"": "Year"}).set_index("Index")
df["text"] = df.apply(lambda r: f"{r['Name']} was born on {r['Month']}/{r['Day']}/{r['Year']}", axis=1)
display(df.head())

def embed(text):
    return openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )["data"][0]["embedding"]

df["embedding"] = df['text'].apply(embed)
df[["text", "embedding"]].to_csv("presidents_embeddings.csv")


# In[ ]:


"""
We create a registry entry here in code. Notice we set number of retrieved documents k=2.
"""

registry_yaml = f"""
retrieval/presidents/gpt-3.5-turbo:
  class: evals.completion_fns.retrieval:RetrievalCompletionFn
  args:
    completion_fn: gpt-3.5-turbo
    embeddings_and_text_path: {os.path.abspath('presidents_embeddings.csv')}
    k: 2

retrieval/presidents/cot/gpt-3.5-turbo:
  class: evals.completion_fns.retrieval:RetrievalCompletionFn
  args:
    completion_fn: cot/gpt-3.5-turbo
    embeddings_and_text_path: {os.path.abspath('presidents_embeddings.csv')}
    k: 2
""".strip()

# Replace with path to your registry
os.makedirs("completion_fns", exist_ok=True)
with open("completion_fns/retrieval.yaml", "w") as f:
    f.write(registry_yaml)

# GPT-3.5-turbo base: accuracy 0.7
get_ipython().system('oaieval gpt-3.5-turbo born-first --max_samples 10 --registry_path .')

# GPT-3.5-turbo with retrieval: accuracy 0.9 -> The failure mode here is the retrieved president is incorrect: Andrew Johnson vs Andrew Jackson
get_ipython().system('oaieval retrieval/presidents/gpt-3.5-turbo born-first --max_samples 10 --registry_path .')

# GPT-3.5-turbo with retrieval and chain-of-thought: accuracy 1.0
get_ipython().system('oaieval retrieval/presidents/cot/gpt-3.5-turbo born-first --max_samples 10 --registry_path .')


# In[ ]:




