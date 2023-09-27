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
plt.figure(figsize=(24, 12))
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

colors = [
    'r','g','b','black','yellow','grey', 'orange','silver','purple','brown','beige','salmon','pink','pulm'
]
styles = [
    'dashed','solid','dotted','dashdot','dashdot','solid','solid','dashdot','solid','solid','solid','dashed','dashed',
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

plt.close()

import matplotlib.pyplot as plt
plt.tick_params(axis='x', labelsize=8)




metric = list(data_map.values())[0]
for idx in range(len(metric)):
    plt.figure(figsize=(24, 9), dpi=80)
    # axs.tick_params(axis='x', labelrotation=-80, gridOn=True)
    # axs.tick_params(axis='x', gridOn=True)

    for i,(k,v) in enumerate(data_map.items()):
        x_label = list(v.keys())[idx]  # x轴的坐标
        y_label = list(v.values())[idx]
        plt.bar(k, y_label,width=0.2)
        plt.text(i, y_label, "{:0.1f}".format(y_label * 100), weight="bold", ha="center", va="bottom")


    plt.title(list(metric.keys())[idx] + ' 5 shot')
    plt.xlabel("model")
    plt.ylabel("acc")
    plt.savefig(f"../../assets/imgs/img_{idx}.jpg", bbox_inches='tight')

plt.close()

# avg
avg_map = {k: np.average(list(v.values())) for k,v in data_map.items()}
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(figsize=(24, 9), dpi=80)

for k,v in zip(avg_map.keys(), avg_map.values()):
    plt.bar(k, v,width=0.2)

for i,pos in enumerate(list(avg_map.values())):
    plt.text(i, pos, "{:0.1f}".format(pos * 100), weight="bold",  ha="center", va="bottom")

plt.title('5 shot avg')
plt.xlabel("model")
plt.ylabel("acc")

plt.savefig(f"../../assets/imgs/img_avg.jpg", bbox_inches='tight')