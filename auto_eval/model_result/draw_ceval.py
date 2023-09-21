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


import matplotlib.pyplot as plt   #导入包
plt.figure(figsize=(18, 12))
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

colors = [
    'r','g','b','black','yellow','grey', 'orange','silver','purple','brown','beige','golden'
]
styles = [
    'dashed','solid','dotted','dashdot','dashdot','solid','solid','dashdot','solid','solid','solid',
]
idx = -1
for k,v in data_map.items():
    idx += 1
    x_label = list(v.keys())  # x轴的坐标
    y_label = list(v.values())
    plt.plot(x_label, y_label, color=colors[idx], linewidth=1.0, linestyle=styles[idx], label=k)



plt.title("ceval 5 shot")                 #设置标题，这里只能显示英文，中文显示乱码
plt.ylabel("acc")            #设置y轴名称
plt.xlabel("category")            #设置x轴名称


plt.legend()

plt.savefig(f"../../assets/imgs/img.jpg")

import matplotlib.pyplot as plt
plt.tick_params(axis='x', labelsize=8)




metric = list(data_map.values())[0]
fig: plt.Figure
axs: plt.axes
for idx in range(len(metric)):
    fig , axs = plt.subplots(figsize=(18, 4))
    # axs.tick_params(axis='x', labelrotation=-80, gridOn=True)
    axs.tick_params(axis='x', gridOn=True)

    for k,v in data_map.items():
        x_label = list(v.keys())[idx]  # x轴的坐标
        y_label = list(v.values())[idx]
        axs.bar(k, y_label,width=0.2)


    fig.suptitle(list(metric.keys())[idx] + ' 5 shot')
    fig.supxlabel("model")
    fig.supylabel("acc")
    fig.savefig(f"../../assets/imgs/img_{idx}.jpg", bbox_inches='tight')



# avg
avg_map = {k: np.average(list(v.values())) for k,v in data_map.items()}
fig, axs = plt.subplots(figsize=(18, 4))
# axs.tick_params(axis='x', labelrotation=-80, gridOn=True)
axs.tick_params(axis='x', gridOn=True)
axs.bar(avg_map.keys(), avg_map.values(),width=0.2)
fig.suptitle(' 5 shot avg')
fig.supxlabel("model")
fig.supylabel("acc")
fig.savefig(f"../../assets/imgs/img_avg.jpg", bbox_inches='tight')