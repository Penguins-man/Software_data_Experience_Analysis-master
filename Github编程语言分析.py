#!/usr/bin/env python
# coding: utf-8

# In[352]:


#-*-coding:utf-8-*-
#环境及依赖项导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

import folium

from matplotlib.font_manager import _rebuild
_rebuild()
# 设置中文字体 kesci 专用代码

def insertFonts():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    sns.set(context='notebook', style='ticks', font='SimHei')
    
print('成功建立开发环境')


# In[353]:


#加载数据集
issues = pd.read_csv('../数据/issues.csv')
prs = pd.read_csv('../数据/prs.csv')
repos = pd.read_csv('../数据/repos.csv')
print('数据集加载成功')


# In[354]:


# 读取各个数据集的基本情况
print('-----------------issues数据集基本情况-----------------')
issues.dropna().info()
print('-------------------prs数据集基本情况------------------')
prs.dropna().info()
print('------------------repos数据集基本情况-----------------')
repos.dropna().info()


# In[355]:


#保存原始数据
issues_df = issues.copy()
prs_df = prs.copy()


# In[356]:


# 通过连接年份和季度添加日期
pd.options.mode.chained_assignment = None 
def adjust_date(df):
    df['date'] = df['year'].astype('str') + '-' + (df['quarter']*3-2).astype('str').str.pad(2,fillchar='0')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
    df.drop(['year','quarter'],axis=1,inplace=True)
    df = df[df.date<'2022-01-01']
    return df
issues = adjust_date(issues)
prs = adjust_date(prs)


# In[357]:


issues.head()


# In[358]:


issues.describe()


# In[359]:


prs.head()


# In[360]:


prs.describe()


# In[361]:


repos.head()


# In[362]:


repos.describe()


# In[363]:


#绘制词云分布图
text=""
for i,lan in enumerate(repos.language):
    text = "".join([text,("".join(lan.split()) + " ")*repos.num_repos[i]])
word_cloud = WordCloud(collocations = False,max_font_size=50, background_color="white").generate(text)
plt.figure(figsize=(10,5))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[364]:


#排名前50的编程语言
top = 50
plt.figure(figsize=(10,15))
repos.num_repos = repos.num_repos/1000
sns.set_color_codes("pastel")
ax = sns.barplot(x="num_repos", y="language", data=repos.iloc[:top+1,:],label="Total", color="b")
ax.axes.set_title(f'Repos of Top {top} Languages',fontsize=18)
ax.set_xlabel("Number of Repos (in thousands)",fontsize=15)
ax.set_ylabel("Language",fontsize=15);


# In[365]:


#当前最流行的5种编程语言
top = 5
name_list = repos.sort_values(by='num_repos', ascending=False)[:top]['language']
name_list


# In[366]:


#所有语言总的issues变化
total_by_date = issues.groupby(['date']).sum()/1000
insertFonts()
total_by_date.plot.line(figsize=(10,5), marker='o')
plt.grid()
plt.title('Total Issues Count',fontsize=18)
plt.xlabel('Year')
plt.ylabel('Number of Issues(in thousand)')


# In[367]:


#最流行的5种编程语言其issues变化趋势
top_issues = issues[issues.name.isin(name_list)]
top_issues['count'] = top_issues['count']/1000
insertFonts()
plt.figure(figsize=(10,5))
sns.set(style='whitegrid')
ax = sns.lineplot(x='date',y='count',hue='name',data=top_issues)
ax.axes.set_title(f'Issues of Top {top} Language',fontsize=15)
ax.set_xlabel("Year",fontsize=12)
ax.set_ylabel("Issues Count (in thousand)",fontsize=12);


# In[368]:


#最流行的5种编程语言每一年的issues变化
def create_issue_line_charts(year):
    issues_ = issues_df[(issues_df['year'] == year) & (issues_df['name'].isin(name_list))].sort_values(by='count', ascending=False).pivot(columns='name', index='quarter', values='count').dropna(axis=1)
    plt = issues_.plot.line(legend=True, title=f'{year} Language Issues', figsize=(10,5), xticks=[1,2,3,4])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[369]:


create_issue_line_charts(2019)


# In[370]:


create_issue_line_charts(2020)


# In[371]:


create_issue_line_charts(2021)


# In[372]:


#最流行的5种编程语言整体的pull request变化
top_prs = prs[prs.name.isin(name_list)]
top_prs['count'] = top_prs['count']/1000
plt.figure(figsize=(10,5))
insertFonts()
sns.set(style='whitegrid')
ax = sns.lineplot(x='date',y='count',hue='name',data=top_prs)
ax.axes.set_title(f'Prs of Top {top} Language',fontsize=15)
ax.set_xlabel("Year",fontsize=12)
ax.set_ylabel("Prs Count (in thousand)",fontsize=12);


# In[373]:


#最流行的5种编程语言每一年的pull request变化
def create_prs_line_charts(year):
    prs_ = prs_df[(prs_df['year'] == year) & (prs_df['name'].isin(name_list))].sort_values(by='count', ascending=False).pivot(columns='name', index='quarter', values='count').dropna(axis=1)
    plt = prs_.plot.line(legend=True, title=f'{year} Language PRs', figsize=(10,5), xticks=[1,2,3,4])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[374]:


create_prs_line_charts(2019)


# In[375]:


create_prs_line_charts(2020)


# In[376]:


create_prs_line_charts(2021)


# In[377]:


#编程语言JavaScript、Python prs和issues比较
def create_line_plot(year, name):
    issues_ = issues_df[(issues_df['year'] == year) & (issues_df['name'] == name)].replace(to_replace=name, value=f'{name} Issues')
    prs_ = prs_df[(prs_df['year'] == year) & (prs_df['name'] == name)].replace(to_replace=name, value=f'{name} PRs')
    df = prs_.append(issues_).pivot(columns='name', index='quarter', values='count').dropna(axis=1)
    plt = df.plot.line(legend=True, title=f'{year} {name} PRs vs Issues Per Quarter',figsize=(10,5),xticks=[1,2,3,4])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    return plt


# In[378]:


plt_JavaScript_2019 = create_line_plot(2019, 'JavaScript')


# In[379]:


plt_JavaScript_2020 = create_line_plot(2020, 'JavaScript')


# In[380]:


plt_JavaScript_2021 = create_line_plot(2021, 'JavaScript')


# In[381]:


plt_Python_2021 = create_line_plot(2019, 'Python')


# In[382]:


plt_Python_2021 = create_line_plot(2020, 'Python')


# In[383]:


plt_Python_2021 = create_line_plot(2021, 'Python')


# In[384]:


#Comparing Total Issues and PRs per Year Regardless of Language
#比较每一年相关语言总的的发布数和prs数量
pr_sums = prs_df.drop(columns='quarter').groupby('year').sum()
issue_sums = issues_df.drop(columns='quarter').groupby('year').sum()
combined_sums = pr_sums.merge(right=issue_sums, on='year')
combined_sums.rename(columns={'count_x': 'Count PRs', 'count_y': 'Count Issues'}, inplace=True)
combined_sums.plot.bar(figsize=(10,5))


# In[389]:


#加载数据包及格式化
def load_dataset(df,name):
    dataframe = df[df['name'] == name]['count']/1000
    dataset = dataframe.values
    dataset = dataset.astype('float32')[:,None]
    return dataset


# In[398]:


from scipy.optimize import curve_fit
#Logistic函数拟合曲线
def logistic_increase_function(t,K,P0,r):
    # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
    t0 = 0
    r = 0.2 
    exp_value=np.exp(r*(t-t0))
    return (K*exp_value*P0)/(K+(exp_value-1)*P0)


# In[399]:


def draw_Logistc(dataset):
    train_size = int(len(dataset) * 0.8)  # 80%的训练集，剩下测试集
    test_size = len(dataset) - train_size
    # 将整型变为float
    t=np.arange(train_size,dtype=float)
    P= [i for arr in dataset[:train_size] for i in arr]
    # 用最小二乘法估计拟合
    popt, pcov = curve_fit(logistic_increase_function, t, P)
    #获取popt里面是拟合系数
    print("K:capacity  P0:initial_value   r:increase_rate   t:year")
    print(popt)
    #拟合后预测的P值
    P_predict = logistic_increase_function(t,popt[0],popt[1],popt[2])
    #未来预测
    future=np.arange(int(len(dataset)*0.8),int(len(dataset)),1)
    future=np.array(future)
    future_predict=logistic_increase_function(future,popt[0],popt[1],popt[2])
    #近期情况预测
    tomorrow=np.arange(int(len(dataset)*0.8),int(len(dataset)),1)
    # tomorrow=np.array(tomorrow)
    tomorrow_predict=logistic_increase_function(tomorrow,popt[0],popt[1],popt[2])

    #绘图
    plot1 = plt.plot(t, P, 's',label="confimed infected people number")
    plot2 = plt.plot(t, P_predict, 'r',label='fit infected people number')
    plot3 = plt.plot(tomorrow, tomorrow_predict, 'b',label='predict infected people number')
    plt.xlabel('time')
    plt.ylabel('confimed infected people number')

    plt.legend(loc=0) #指定legend的位置右下角

    print(logistic_increase_function(np.array(28),popt[0],popt[1],popt[2]))
    print(logistic_increase_function(np.array(29),popt[0],popt[1],popt[2]))


# In[400]:


#选取python这些年的发布数量拟合预测
draw_Logistc(load_dataset(issues_df,'Python'))
#可以看出使用logistic拟合的并不是很好，所以下面采用LSTM拟合


# In[402]:


#选取python这些年的pull request数拟合预测
draw_Logistc(load_dataset(prs_df,'Python'))
#可以看出使用logistic拟合的并不是很好，所以下面采用LSTM拟合


# In[403]:


import math
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model


# In[404]:


def LSTM_Predict(dataset):
    values1 = dataset
    dataset = values1.reshape(-1, 1)  # 注意将一维数组，转化为2维数组
    def create_dataset(dataset, look_back=1):  # 后一个数据和前look_back个数据有关系
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)  # .apeend方法追加元素
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)  # 生成输入数据和输出数据

    numpy.random.seed(7)  # 随机数生成时算法所用开始的整数值

    #正则化
    scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化0-1
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets  #训练集和测试集分割
    train_size = int(len(dataset) * 0.8)  # 80%的训练集，剩下测试集
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]  # 训练集和测试集

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)  # 训练输入输出
    testX, testY = create_dataset(test, look_back)  # 测试输入输出

    # [samples, time steps, features]注意转化数据维数
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # 建立LSTM模型
    model = Sequential()
    model.add(LSTM(200, input_shape=(1, look_back)))  # 隐层200个神经元 （可以断调整此参数提高预测精度）
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['acc'])  # 评价函数mse，优化器adam
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)  # 100次迭代
    model.save('./newmodel/newmodel.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # 数据反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    plt.figure(figsize=(10, 4))
    l1, = plt.plot(scaler.inverse_transform(dataset), color='red', linewidth=5, linestyle='--')
    l2, = plt.plot(trainPredictPlot, color='k', linewidth=4.5)
    l3, = plt.plot(testPredictPlot, color='g', linewidth=4.5)
    plt.ylabel('Height m')
    plt.legend([l1, l2, l3], ('raw-data', 'true-values', 'pre-values'), loc='best')
    plt.title('LSTM Gait Prediction')
    plt.show()


# In[405]:


LSTM_Predict(load_dataset(issues_df,'Python'))
#可以看出通过自己训练LSTM模型，能够较为准确地预测python的发布数


# In[406]:


LSTM_Predict(load_dataset(prs_df,'Python'))
#可以看出通过自己训练LSTM模型，能够较为准确地预测python的pr数






