import json
import os
import numpy as np
with open("../assets/ceval_data/ceval_categories.json",mode="r",encoding="utf-8") as f:
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
    return data_new
with open("ceval/baichuan-13b-chat-int4/metric.json",mode="r",encoding="utf-8") as f:
    baichuan = get_data(json.loads(f.read()))

with open("ceval/chatglm2-6b-int4/metric.json",mode="r",encoding="utf-8") as f:
    chatglm2 =  get_data(json.loads(f.read()))

with open("ceval/qwen-7b-chat-int4/metric.json",mode="r",encoding="utf-8") as f:
    qwen =  get_data(json.loads(f.read()))

with open("ceval/xverse-13b-chat-int4/metric.json",mode="r",encoding="utf-8") as f:
    xverse =  get_data(json.loads(f.read()))

with open("ceval/moss-moon-003-sft-int4/metric.json",mode="r",encoding="utf-8") as f:
    moss =  get_data(json.loads(f.read()))

import matplotlib.pyplot as plt   #导入包
fig = plt.figure()              #创建空图
x_label = list(baichuan.keys())     #x轴的坐标
y_label = list(baichuan.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'r',linewidth=1.0,linestyle='dashed',label="baichuan-13b")  #构建折线图，可以设置线宽，颜色属性

x_label = list(chatglm2.keys())     #x轴的坐标
y_label = list(chatglm2.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'g',linewidth=1.0,linestyle='solid',label="chatglm2-6b")  #构建折线图，可以设置线宽，颜色属性

x_label = list(qwen.keys())     #x轴的坐标
y_label = list(qwen.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'b',linewidth=1.0,linestyle='dotted',label="qweb-7b")  #构建折线图，可以设置线宽，颜色属性

x_label = list(xverse.keys())     #x轴的坐标
y_label = list(xverse.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'black',linewidth=1.0,linestyle='dashdot',label="xverse-13b")  #构建折线图，可以设置线宽，颜色属性

x_label = list(moss.keys())     #x轴的坐标
y_label = list(moss.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'yellow',linewidth=1.0,linestyle='dashdot',label="moss-16b")  #构建折线图，可以设置线宽，颜色属性

plt.title("ceval")                 #设置标题，这里只能显示英文，中文显示乱码
plt.ylabel("acc")            #设置y轴名称
plt.xlabel("ceval category")            #设置x轴名称


plt.legend()
plt.show()                       #将图形显示出来