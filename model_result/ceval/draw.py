import json
import os


with open("baichuan-13b-chat-int4/metric.json",mode="r",encoding="utf-8") as f:
    baichuan = json.loads(f.read())

with open("chatglm2-6b-int4/metric.json",mode="r",encoding="utf-8") as f:
    chatglm2 = json.loads(f.read())

with open("qwen-7b-chat-int4/metric.json",mode="r",encoding="utf-8") as f:
    qwen = json.loads(f.read())

with open("xverse-13b-chat-int4/metric.json",mode="r",encoding="utf-8") as f:
    xverse = json.loads(f.read())

print(qwen)

import matplotlib.pyplot as plt   #导入包
fig = plt.figure()              #创建空图
x_label = list(range(len(baichuan.keys())))     #x轴的坐标
y_label = list(baichuan.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'r',linewidth=1.0,linestyle='dashed',label="baichuan-13b")  #构建折线图，可以设置线宽，颜色属性

x_label = list(range(len(chatglm2.keys())))     #x轴的坐标
y_label = list(chatglm2.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'g',linewidth=1.0,linestyle='solid',label="chatglm2-6b")  #构建折线图，可以设置线宽，颜色属性

x_label = list(range(len(qwen.keys())))     #x轴的坐标
y_label = list(qwen.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'b',linewidth=1.0,linestyle='dotted',label="qweb-7b")  #构建折线图，可以设置线宽，颜色属性

x_label = list(range(len(xverse.keys())))     #x轴的坐标
y_label = list(xverse.values())     #y轴坐标
plt.plot(x_label,y_label,color = 'black',linewidth=1.0,linestyle='dashdot',label="xverse-13b")  #构建折线图，可以设置线宽，颜色属性


plt.title("line")                 #设置标题，这里只能显示英文，中文显示乱码
plt.ylabel("acc")            #设置y轴名称
plt.xlabel("ceval category")            #设置x轴名称


plt.legend()
plt.show()                       #将图形显示出来