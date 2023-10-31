# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/31 11:05
# Libraries
import pandas as pd
import json
import os
import numpy as np
with open("../../assets/ceval_data/ceval_categories.json",mode="r",encoding="utf-8") as f:
   category_all = json.loads(f.read())


category = {k: [] for k in set([v[-1] for k,v in category_all.items()])}
for k,v in category_all.items():
    category[v[-1]].append(k)
for k,v in category.items():
    category[k] = list(set(v))

def get_data(data: dict):
    data_new = {_: [] for _ in category}
    for k,v in data.items():
        data_new[category_all[k][-1]].append(v)

    data_new = {k: np.mean(v)  for k,v in data_new.items()}
    data_new = dict(sorted(data_new.items(),key=lambda x: x[0]))
    return data_new

data_map = {}
files = os.listdir("ceval")
files = sorted(files)
for file in files:
    with open(os.path.join('ceval',file,'metric.json'),mode='r',encoding='utf-8') as f:
        data_map[file.replace('-int4','')] = get_data(json.loads(f.read()))


df = pd.DataFrame(data_map)
df = df.T
print(df)



import pandas as pd
from math import pi




import matplotlib.pyplot as plt

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

df_al = df["Humanities"] + df["Other"] + df["STEM"] + df["Social Science"]

avg = df.mean(axis=1)
avg = pd.DataFrame(avg)
avg.columns = ["avg"]

df = df.merge(avg,right_index=True,left_index=True)

df = df.sort_values(by=["avg"],ascending=False)
print(df)
df = df.head(6)
del df["avg"]
labels = list(df.keys())
# 使用ggplot的绘图风格
# plt.style.use('ggplot')

# 构造数据

# 绘图
fig = plt.figure(num=1,figsize=(16, 12))
ax = fig.add_subplot(111, polar=True)
for i,d in enumerate(df.iterrows()):
    d: pd.Series = d[1]
    label = d.name
    values = d.values * 100
    print(label, values)

    N = len(labels)
    # 设置雷达图的角度，用于平分切开一个圆面
    angles=np.linspace(0, 2*np.pi, N, endpoint=False)
    # 为了使雷达图一圈封闭起来，需要下面的步骤
    values=np.concatenate((values,[values[0]]))

    angles=np.concatenate((angles,[angles[0]]))


    # 绘制折线图
    ax.plot(angles, values, 'o-', linewidth=1,label=label)
    # 填充颜色
    ax.fill(angles, values, alpha=0.25)

    # 添加每个特征的标签
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    # 设置雷达图的范围
    ax.set_ylim(0,100)
    # 添加网格线
    ax.grid(True)
    # 设置图例
    plt.legend(loc="center", fontsize=9,bbox_to_anchor=(0.9,0.9), ncol=1)

plt.title('ceval five shot avg for top5',loc="center")
# 显示图形
plt.savefig(f"../../assets/imgs/img_avg_top6.jpg", bbox_inches='tight')