#coding:utf-8
import pandas as pd

fpath = "D://数据集大荟萃/周杰伦新歌《Mojito》豆瓣短评数据集/mojito6931/1Mojito豆瓣短评数据6.12.csv"

df = pd.read_csv(fpath,engine='python')

# 还是屈服于分词啊哈哈哈
import jieba

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

df['cut_content'] = df.content.apply(chinese_word_cut)

# res = df['attitude'].value_counts(normalize=True)
res = df['attitude'].value_counts()
label_dict = {0:"不喜欢",1:"中立",2:"喜欢"}
x = []
y = []

for i,v in res.items():
    print('index:',i,'value:',v)
    x.append(label_dict[i])
    y.append(v)

# 绘制 三种态度 饼图
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

fig1, ax1 = plt.subplots()

ax1.pie(y,labels=x,autopct='%1.2f%%')

# x,y轴长度相同
ax1.axis('equal')

plt.savefig('D:/mojito/atti_bing.jpg',dpi=300)
plt.show()


vote_dict = {0:0,1:0,2:0}
for index,row in df.iterrows():
#     print(row['vote_count'],row['attitude'])
    vote_dict[row['attitude']]+=row['vote_count']
    
vote_dict

# 绘制饼图：统计各类投票数占比

fig2, ax2 = plt.subplots()

ax2.pie(vote_dict.values(),labels=x,autopct='%1.2f%%')

# x,y轴长度相同
ax2.axis('equal')
plt.savefig('D:/mojito/atti_bing.jpg',dpi=300)
plt.show()

# 得到最高赞的10个评论和态度
df.sort_values(by="vote_count",ascending=False)[:10]

# 得到最高赞的10个评论和态度
maxvote_df_10 = df.sort_values(by="vote_count",ascending=False)[:10]

maxvote_dict = {}

for index,row in maxvote_df_10.iterrows():
    print(row["vote_count"],row["attitude"])
    maxvote_dict[row["vote_count"]] = label_dict[row["attitude"]]

# 绘制柱状图
labels = maxvote_dict.values()
votes = maxvote_dict.keys()

x = np.arange(10)
width = 0.35

fig3,ax3 = plt.subplots()
rects = ax3.bar(x,votes,width,label = '投票数')

ax3.set_ylabel('votes')
ax3.set_title('前十名高赞评论')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend()

# 在柱顶加具体投票数
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax3.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



autolabel(rects)
    
fig3.tight_layout()

plt.savefig('D:/mojito/max10_zhu.jpg',dpi=300)
plt.show()


allcontent = ""
for index,row in df.iterrows():
    allcontent+=row['content']

# [所有content]绘制词云
from wordcloud import WordCloud as wc

text_cut = jieba.lcut(allcontent)
text_cut = ' '.join(text_cut)

stop_words_file = 'D:\文本数据预处理常用工具\stopwords-master\hit_stopwords.txt'
stop_words = open(stop_words_file,encoding="utf8").read().split("\n")

word_cloud = wc(font_path="simsun.ttc",
               background_color="white", stopwords=stop_words,scale=4)

word_cloud.generate(text_cut)

plt.subplots(figsize=(72,64))

plt.imshow(word_cloud)

plt.axis("off")

plt.savefig('D:/all_wordcloud.jpg')


max10_content = ""

for index,row in maxvote_df_10.iterrows():
    max10_content+=row["content"]


text_cut = jieba.lcut(max10_content)
text_cut = ' '.join(text_cut)

stop_words_file = 'D:\文本数据预处理常用工具\stopwords-master\hit_stopwords.txt'
stop_words = open(stop_words_file,encoding="utf8").read().split("\n")

word_cloud = wc(font_path="simsun.ttc",
               background_color="white", stopwords=stop_words,scale=4)

word_cloud.generate(text_cut)

plt.subplots(figsize=(72,64))

plt.imshow(word_cloud)

plt.axis("off")

plt.savefig('D:/max10_wordcloud.jpg')


# 填充缺失值
df['rating_num'] = df['rating_num'].fillna(1)

import numpy as np

df["rating_num"].count()

for index,row in df.iterrows():
    rate_num = row["rating_num"]
    if rate_num==3:
        continue
    else:
        rate_num = int(rate_num[18])
        print(rate_num)
        #以下方式不会报警，且避免了df.ix[index,column]已过时的警告
        df.iloc[index,df.columns.get_loc('rating_num')]=rate_num


res = df['rating_num'].value_counts(ascending=False)
label_dict = {1:"一星",2:"二星",3:"三星",4:"四星",5:"五星"}
x = []
y = []

for i,v in res.items():
    print('index:',i,'value:',v)
    x.append(label_dict[i])
    y.append(v)
    print(x,y)


# 绘制 星级 饼图
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

fig4, ax4 = plt.subplots()

ax4.pie(y,labels=x,autopct='%1.2f%%')

# x,y轴长度相同
ax4.axis('equal')
plt.savefig('D:/mojito/rating_count_bing.jpg',dpi=300)

plt.show()


# 绘制柱状图
labels = x
votes = y

x = np.arange(1,6)
width = 0.35

fig5,ax5 = plt.subplots()
rects = ax5.bar(x,y,width,label = '星级统计')

ax5.set_ylabel('count')
ax5.set_title('星级统计')
ax5.set_xticks(x)
ax5.set_xticklabels(labels)
ax5.legend()

# 在柱顶加具体投票数
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax5.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



autolabel(rects)
    
fig5.tight_layout()

plt.savefig('D:/mojito/rating_count_zhu.jpg',dpi=300)

plt.show()