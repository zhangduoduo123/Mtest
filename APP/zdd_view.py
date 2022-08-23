
from APP.models import File, User, Yuanshi,Xiangdui,NfYuanshi,NfXiangdui
from django.http import HttpResponse, request, response, JsonResponse, FileResponse, StreamingHttpResponse
import copy
import math
import pandas as pd
from django.core.paginator import Paginator
from django.db.models import Q, Avg, Sum, Max, F, Min
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
import numpy as np
from django.shortcuts import render
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances
from APP.models import NfXiangdui, Xiangdui, Res,NfYuanshi,Tempxiangdui,Tempyuanshi

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.datasets import load_iris
from sklearn.cross_decomposition import PLSRegression


import matplotlib.pyplot as plt
matplotlib.rc("font",family='YouYuan')
#plt.rcParams['axes.unicode_minus']=False
def compute_VIP(X, y, R, T, A):
    """
    计算模型中各预测变量的VIP值
    :param X: 数据集X
    :param y: 标签y
    :param R: A个PLS成分中，每个成分a都对应一套系数wa将X转换为成分得分，系数矩阵写作R，大小为p×A
    :param T: 得分矩阵记做T，大小为n×A，ta代表n个样本的第a个成分的得分列表
    :param A: PLS成分的总数
    :return: VIPs = np.zeros(p)
    """
    p = X.shape[1]
    Q2 = np.square(np.dot(y.T, T))

    VIPs = np.zeros(p)
    temp = np.zeros(A)
    for j in range(p):
        for a in range(A):
            temp[a] = Q2[a] * pow(R[j, a] / np.linalg.norm(R[:, a]), 2)
        VIPs[j] = np.sqrt(p * np.sum(temp) / np.sum(Q2))
    return VIPs

def charts_pro(all_naifen,charts,):


    X = all_naifen
    x = [ "age", "birth_jd","fenmian_way", "chanci","minzu","zaochan"]
    y = ['a','b']
    # x 影响因素 y 氨基酸
    g = sns.PairGrid(X,
                     x_vars=x,
                     y_vars=y, palette='GnBu_d',
                     )
    # 下三角绘多 sns.boxplot plt.scatter sns.boxplot sns.violinplot
    if charts == 'box':
        plotway = sns.boxplot
    # elif charts == 'vio':
    #     plotway = sns.violinplot
    else:
        plotway = sns.violinplot

    my_dict_x = {"city": "南方1北方2", "age": "年龄28以下1，28以上2", "birth_jd": "冬春2夏秋1",
                 "fenmian_way": "1，阴道分娩，2，阴道手术分娩，3，剖宫产", "chanci": "1胎2胎3胎","minzu":"1汉族2少数民族",
                 "zaochan":"1，足月，2，早产，3，过期产" ,
               }
    xlabels = []
    for i in range(len(x)):
        xlabels.append(my_dict_x[x[i]])
    my_dict_y = {"a": "α-乳白蛋白","b": "β-酪蛋白", }
    ylabels = []
    for i in range(len(y)):
        ylabels.append(my_dict_y[y[i]])
    g.map_diag(plotway)
    g.map_offdiag(plotway)
    for i in range(len(x)):
        for j in range(len(y)):
            g.axes[j, i].xaxis.set_label_text(xlabels[i])
            g.axes[j, i].yaxis.set_label_text(ylabels[j])
    plt.show()

    filepath = 'static/pic/'
    g.savefig(str(filepath) + str(charts) + '.jpg', dpi=400)
    filename2 = str(charts) + '.jpg'
    g.savefig('box.jpg', dpi=300)
    # savepath(filename1, filepath, uid)
    #写入CSV
    res = []
    res.append('蛋白质')
    res.append('置信区间下限制')
    res.append('置信区间上限制')
    res.append('箱型图下限')
    res.append('上四分')
    res.append('中位数')
    res.append('下四分')
    res.append('箱型图上限')
    my_dict_y = {"a": "α-乳白蛋白","b": "β-酪蛋白", }

    for ajs1 in ['a','b']:
        res.append(my_dict_y[ajs1])
        des = X[ajs1].describe()
        # 得到每列的平均值,是一维数组
        mean = des['mean']
        # 得到每列的标准差,是一维数组
        std = des['std']
        numll = des['count']
        numll = math.sqrt(numll)
        qz0 = mean - 1.96 * std / numll
        qz5 = mean + 1.96 * std / numll
        q1 = des['25%']
        q2 = des['50%']
        q3 = des['75%']
        IQR = q3 - q1
        q0 = q1 - 1.5 * IQR
        q5 = q3 + 1.5 * IQR
        # 置信区间上下限制
        res.append(qz0)
        res.append(qz5)
        # 箱型图下线
        res.append(q0)
        # 上四分卫 中位数 下四分
        res.append(q1)
        res.append(q2)
        res.append(q3)
        # 箱型图上线
        res.append(q5)
        # sql_query1 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' <= '+str(q0)+';'
        # X1 = pd.read_sql_query(sql_query1, engine)
        # X1 = np.array(X1)  # 先将数据框转换为数组
        # X1 = X1.tolist()  # 其次转换为列表
        # res.append(X1)
        # sql_query2 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' >= '+str(q5)+';'
        # X2 = pd.read_sql_query(sql_query2, engine)
        # X2 = np.array(X2)  # 先将数据框转换为数组
        # X2 = X2.tolist()
        # res.append(X2)

    X = [res[i:i + 8] for i in range(0, len(res), 8)]
    X1 = pd.DataFrame(X)
    a = None
    filepath = 'static/picdata/'
    X1.to_csv(str(filepath) + str(charts) + '.csv', encoding='utf-8-sig')
    filename1 = str(charts) + '.csv'
    return filename2,filename1


def plot_pro(iris_t2,inf):

        numlem = len(iris_t2)
        for i in range(numlem):
            if iris_t2[inf][i] == 1:
                plt.plot(iris_t2.iloc[i][-1], iris_t2.iloc[i][-2], 'o', color='green')
            elif iris_t2[inf][i] == 2:
                plt.plot(iris_t2.iloc[i][-1], iris_t2.iloc[i][-2], '*', color='red')
            else:
                plt.plot(iris_t2.iloc[i][-1], iris_t2.iloc[i][-2], '*', color='blue')
        for i in range(numlem):
            plt.text(iris_t2.iloc[i][-1], iris_t2.iloc[i][-2],iris_t2.iloc[i][0], fontsize=5)

        my_dict_x = {"city": "南方-绿色，北方-红色", "age": "年龄28以下-绿色，28以上-红色",
                     "birth_jd": "冬春-红色，夏秋-绿色",
                     "fenmian_way": "绿色-阴道分娩，红色-阴道手术分娩，蓝色-剖宫产", "chanci": "绿色-1胎，红色-2胎，蓝色-3胎次",
                     "minzu": "绿色-汉族，红色-少数民族",
                     "zaochan": "绿色-足月，红色-早产，蓝色-过期产",

                     }
        xl = my_dict_x[inf]
        plt.xlabel(xl)
        plt.savefig(str(xl) + '.jpg', dpi=300)
        plt.savefig('box.jpg', dpi=300)
        plt.show()
def distancepro(all_naifen,all_anjisuan):
    OUT = []
    Z2 = all_anjisuan['NO']


    al = len(all_anjisuan)

    for anjisuan in range(al):
        Y1 = list(all_anjisuan.iloc[anjisuan])
        for anjisuan in range(al):
            X = list(all_anjisuan.iloc[anjisuan])

            Z = []
            Z.append(X[-2:])
            Z.append(Y1[-2:])

            OUT.append((pairwise_distances(Z, metric="euclidean")[1][0]))
            OUT.append((pairwise_distances(Z, metric="cosine")[1][0]))
            OUT.append(pairwise_distances(Z, metric="braycurtis")[1][0])
            OUT.append((pairwise_distances(Z, metric="correlation")[1][0]))


    X = [OUT[i:i + 4] for i in range(0, len(OUT), 4)]
    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt + 1
        # 妈妈类型个数
        if cnt == len(all_anjisuan):
            temp.append(i)
            temp = list(zip(*temp))
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)

    new_list = Y
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)
    new_list = temp2
    new_list1 = []
    for i in new_list:
        temp = 0
        ilen = len(i)
        for j in range(len(i)):
            if j == len(i) - 1:
                new_list1.append(i[j])
                new_list1.append(temp)
            else:
                new_list1.append(i[j])
                temp = temp + i[j]
    iilen = ilen + 1
    new_list_t = [new_list1[i:i + iilen] for i in range(0, len(new_list1), iilen)]

    Z1 = [
        "欧氏距离",
        "余弦距离",
        "braycurtis",
        "correlation",

    ]
    count = -1
    zcnt = 0

    for i in new_list_t:
        count = count + 1
        # 12 距离种类个数
        if count == 3:
            i.insert(0, Z1[count])
            i.insert(0, Z2[zcnt])
            zcnt = zcnt + 1
            count = -1
        else:
            i.insert(0, Z1[count])
            i.insert(0, Z2[zcnt])
    df_data1 = pd.DataFrame(new_list_t)

    return  df_data1
def distanceComputemama_3(all_naifen,all_anjisuan,dim,ma):
    OUT = []
    a = []
    b = all_anjisuan.describe()
    a.append(list(b.iloc[1, :]))
    a.append(list(b.iloc[3, :]))
    a.append(list(b.iloc[5, :]))
    a.append(list(b.iloc[7, :]))

    return a
def distanceComputemama(all_naifen,all_anjisuan,dim,ma):
    OUT = []
    Z2 = []

    al = len(all_anjisuan)
    for naifen in range(al):
        Y1 = list(all_naifen.iloc[naifen,:])
        for naifen in range(al):
            X = list(all_naifen.iloc[naifen,:])

            Z = []
            Z.append(X)
            Z.append(Y1)
            OUT.append((pairwise_distances(Z, metric="euclidean")[1][0]))
            OUT.append((pairwise_distances(Z, metric="cosine")[1][0]))
            OUT.append(pairwise_distances(Z, metric="braycurtis")[1][0])
            OUT.append((pairwise_distances(Z, metric="correlation")[1][0]))

    X = [OUT[i:i + 4] for i in range(0, len(OUT), 4)]
    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt + 1
        # 妈妈类型个数
        if cnt == len(all_anjisuan):
            temp.append(i)
            temp = list(zip(*temp))
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)

    new_list = Y
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)
    new_list = temp2



    # ave
    new_list1 = []
    for i in new_list:
        temp = 0
        ilen = len(i)
        for j in range(len(i)):
            if j == len(i) - 1:
                new_list1.append(i[j])
                new_list1.append(temp)
            else:
                new_list1.append(i[j])
                temp = temp + i[j]
    iilen = ilen + 1
    new_list_t = [new_list1[i:i + iilen] for i in range(0, len(new_list1), iilen)]


    averange = []
    cnt = -1
    for i in new_list_t:
        cnt = cnt + 1
        averange.append(i[len(i) - 1])
        averange.append(ma[cnt//4])

    X = [averange[i:i + 2] for i in range(0, len(averange), 2)]

    Z1 = [
        "欧氏距离",
        "余弦距离",
        "braycurtis",
        "correlation",

    ]
    count = -1
    for i in X:
        count = count + 1
        # 12 距离种类个数
        if count == 3:
            i.insert(0, Z1[count])
            count = -1
        else:
            i.insert(0, Z1[count])

    return X
def distanceCompute(all_naifen,all_anjisuan,dim):
    OUT = []
    Z2 = []
    for naifen in all_naifen:
        Z2.append(naifen[0])
    al = len(all_anjisuan)

    for anjisuan in range(al):
        Y1 = list(all_anjisuan.iloc[anjisuan])
        for naifen in all_naifen:
            X = []
            b = list(naifen)
            lb = len(b)-1
            for i in range(lb):
                X.append(b[i+1])

            Z = []
            Z.append(X)
            Z.append(Y1)
            OUT.append((pairwise_distances(Z, metric="euclidean")[1][0]))
            OUT.append((pairwise_distances(Z, metric="cosine")[1][0]))
            OUT.append(pairwise_distances(Z, metric="braycurtis")[1][0])
            OUT.append((pairwise_distances(Z, metric="correlation")[1][0]))
            if dim >= 9:
                # 必需氨基酸的比值
                res = []
                for i in range(9):
                    t = Z[1][i] / Z[0][i]
                    res.append(t)
                # 必需氨基酸的比值的均值
                arr_mean = np.mean(res)
                # 比值系数
                res1 = []
                for i in range(9):
                    t = res[i] / arr_mean
                    res1.append(t)
                # 均值
                arr_mean1 = np.mean(res1)
                # 求标准差
                arr_std = np.std(res1, ddof=1)
                t = 100 - 100 * arr_std / arr_mean1
                OUT.append(t)
            else:
                OUT.append(0)

    X = [OUT[i:i + 5] for i in range(0, len(OUT), 5)]
    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt + 1
        # 妈妈类型个数
        if cnt == len(all_anjisuan):
            temp.append(i)
            temp = list(zip(*temp))
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)

    new_list = Y
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)
    new_list = temp2


    Z1 = [
        "欧氏距离",
        "余弦距离",
        "braycurtis",
        "correlation",
        "氨基酸比值系数",
    ]
    count = -1
    zcnt = 0
    for i in new_list:
        count = count + 1
        # 12 距离种类个数
        if count == 4:
            i.insert(0, Z1[count])
            i.insert(0, Z2[zcnt])
            zcnt = zcnt + 1
            count = -1
        else:
            i.insert(0, Z1[count])
            i.insert(0, Z2[zcnt])
    return new_list

def fundelmamaid(ajs,mr,per,dt):
    temp = ['mother_id', 'city', 'birth_jd', 'age1', 'fenmian_way', 'chanci']
    temp.append(ajs[0])
    for i in range(len(ajs)):
        if dt == "相对数据":
            all_anjisuan_temp = Xiangdui.objects.values('mother_id', 'city', 'birth_jd', 'age1', 'fenmian_way',
                                                    'chanci').filter(miruqi=mr).annotate(Avg(ajs[i]))
        else:
            all_anjisuan_temp = Yuanshi.objects.values('mother_id', 'city', 'birth_jd', 'age1', 'fenmian_way',
                                                        'chanci').filter(miruqi=mr).annotate(Avg(ajs[i]))

        all_anjisuan_temp=all_anjisuan_temp
        if i == 0:
            for anjisuan in all_anjisuan_temp:
                a = list(anjisuan.values())
                la = len(a)
                for i in range(la):
                    temp.append(a[i])
            all_anjisuan = [temp[i:i + 7] for i in range(0, len(temp), 7)]
            all_anjisuan[0].extend(ajs[1:])
        else:
            cnt = 1
            for anjisuan in all_anjisuan_temp:
                a = list(anjisuan.values())
                la = len(a) - 6
                for i in range(la):
                    all_anjisuan[cnt].append(a[i + 6])
                cnt = cnt + 1
    ma,delmamaid = commamadel(all_anjisuan, per)
    return ma,delmamaid
def delmamasql(ajs,mr,delmamaid,mama,plot,dt):

    temp = ['mother_id', 'city', 'birth_jd', 'age1', 'fenmian_way', 'chanci']
    temp.append(ajs[0])
    for i in range(len(ajs)):
        if dt == "相对数据":
            all_anjisuan_temp = Xiangdui.objects.values('mother_id', 'city', 'birth_jd', 'age1', 'fenmian_way',
                                                    'chanci').filter(miruqi=mr).exclude(
            mother_id__in=delmamaid).annotate(Avg(ajs[i]))
        else:
            all_anjisuan_temp = Yuanshi.objects.values('mother_id', 'city', 'birth_jd', 'age1', 'fenmian_way',
                                                        'chanci').filter(miruqi=mr).exclude(
            mother_id__in=delmamaid).annotate(Avg(ajs[i]))

        if i == 0:
            for anjisuan in all_anjisuan_temp:
                a = list(anjisuan.values())
                la = len(a)
                for i in range(la):
                    temp.append(a[i])
            all_anjisuan = [temp[i:i + 7] for i in range(0, len(temp), 7)]
            all_anjisuan[0].extend(ajs[1:])
        else:
            cnt = 1
            for anjisuan in all_anjisuan_temp:
                a = list(anjisuan.values())
                la = len(a) - 6
                for i in range(la):
                    all_anjisuan[cnt].append(a[i + 6])
                cnt = cnt + 1

    if plot == True:
        return all_anjisuan
    else:
        all_anjisuan = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
        all_anjisuan = all_anjisuan.iloc[:, 6:]
        return all_anjisuan
def delmamasql_inf(ajs, mr, delmamaid, inf, plot):
    temp = ['mother_id', 'city', 'birth_jd', 'age1', 'fenmian_way', 'chanci']
    temp.append(ajs[0])
    for i in range(len(ajs)):
        all_anjisuan_temp = Xiangdui.objects.values('mother_id', 'city', 'birth_jd', 'age1', 'fenmian_way',
                                                    'chanci').filter(miruqi=mr).exclude(
            mother_id__in=delmamaid).annotate(Avg(ajs[i]))
        if i == 0:
            for anjisuan in all_anjisuan_temp:
                a = list(anjisuan.values())
                la = len(a)
                for i in range(la):
                    temp.append(a[i])
            all_anjisuan = [temp[i:i + 7] for i in range(0, len(temp), 7)]
            all_anjisuan[0].extend(ajs[1:])
        else:
            cnt = 1
            for anjisuan in all_anjisuan_temp:
                a = list(anjisuan.values())
                la = len(a) - 6
                for i in range(la):
                    all_anjisuan[cnt].append(a[i + 6])
                cnt = cnt + 1

    if plot == True:
        return all_anjisuan
    else:

        return all_anjisuan
def chartssql_inf(ajs, mr,mama,plot,uid,inf):

    ma,delmamaid = fundelmamaid(ajs,mr,mama)
    all_anjisuan = delmamasql_inf(ajs, mr, delmamaid, inf, plot)

    if plot == True:
                res = []
                res.append('氨基酸')
                res.append('置信区间下限制')
                res.append('置信区间上限制')
                res.append('箱型图下限')
                res.append('上四分')
                res.append('中位数')
                res.append('下四分')
                res.append('箱型图上限')
                my_dict_y = {"fbaa": "非必须必总氨基酸","baa": "必须必总氨基酸", "lai": "赖氨酸", "su": "苏氨酸", "jie": "缬氨酸", "dan": "蛋氨酸", "yiliang": "异亮氨酸",
                             "liang": "亮氨酸",
                             "benbing": "苯丙氨酸", "se": "色氨酸", "zu": "组氨酸", "tiandong": "天冬氨酸", "si": "丝氨酸",
                             "gu": "谷氨酸", "gan": "甘氨酸", "bing": "丙氨酸", "lao": "酪氨酸", "jing": "精氨酸", "fu": "脯氨酸",
                             "banguang": "半胱氨酸",
                             }

                all_anjisuan1 = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
                for ajs1 in ajs:
                    res.append(my_dict_y[ajs1])
                    des = all_anjisuan1[ajs1].describe()
                    # 得到每列的平均值,是一维数组
                    mean = des['mean']
                    # 得到每列的标准差,是一维数组
                    std = des['std']
                    numll = des['count']
                    numll = math.sqrt(numll)
                    qz0 = mean - 1.96 * std / numll
                    qz5 = mean + 1.96 * std / numll
                    q1 = des['25%']
                    q2 = des['50%']
                    q3 = des['75%']
                    IQR = q3 - q1
                    q0 = q1 - 1.5 * IQR
                    q5 = q3 + 1.5 * IQR
                    # 置信区间上下限制
                    res.append(qz0)
                    res.append(qz5)
                    # 箱型图下线
                    res.append(q0)
                    # 上四分卫 中位数 下四分
                    res.append(q1)
                    res.append(q2)
                    res.append(q3)
                    # 箱型图上线
                    res.append(q5)
                    # sql_query1 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' <= '+str(q0)+';'
                    # X1 = pd.read_sql_query(sql_query1, engine)
                    # X1 = np.array(X1)  # 先将数据框转换为数组
                    # X1 = X1.tolist()  # 其次转换为列表
                    # res.append(X1)
                    # sql_query2 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' >= '+str(q5)+';'
                    # X2 = pd.read_sql_query(sql_query2, engine)
                    # X2 = np.array(X2)  # 先将数据框转换为数组
                    # X2 = X2.tolist()
                    # res.append(X2)

                X = [res[i:i + 8] for i in range(0, len(res), 8)]
                X1 = pd.DataFrame(X)
                a = None
                filename1 = english_chinese(mr, ajs, a, mama, '', '')
                filename1 = '删除妈妈后统计数据-'+filename1
                filepath = 'static/picdata/'
                X1.to_csv(str(filepath)  + str(filename1) + '.csv', encoding='utf-8-sig')

                filename1 = str(filename1) + '.csv'
                #savepath(filename1,filepath,uid)

    #奶粉
    ajs1 = copy.deepcopy(ajs)
    ajs1.insert(0, 'nf_name')
    all_naifen = NfXiangdui.objects.values_list(*ajs1)


    return all_naifen,all_anjisuan,ma
def chartssql(ajs, mr,mama,plot,uid,plot1,dt):

    ma,delmamaid = fundelmamaid(ajs,mr,mama,dt)
    all_anjisuan = delmamasql(ajs,mr,delmamaid,mama,plot,dt)
    if plot1 == True:
                res = []
                res.append('氨基酸')
                res.append('置信区间下限制')
                res.append('置信区间上限制')
                res.append('箱型图下限')
                res.append('上四分')
                res.append('中位数')
                res.append('下四分')
                res.append('箱型图上限')
                my_dict_y = {"fbaa": "非必须必总氨基酸","baa": "必须必总氨基酸", "lai": "赖氨酸", "su": "苏氨酸", "jie": "缬氨酸", "dan": "蛋氨酸", "yiliang": "异亮氨酸",
                             "liang": "亮氨酸",
                             "benbing": "苯丙氨酸", "se": "色氨酸", "zu": "组氨酸", "tiandong": "天冬氨酸", "si": "丝氨酸",
                             "gu": "谷氨酸", "gan": "甘氨酸", "bing": "丙氨酸", "lao": "酪氨酸", "jing": "精氨酸", "fu": "脯氨酸",
                             "banguang": "半胱氨酸",
                             }

                all_anjisuan1 = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
                for ajs1 in ajs:
                    res.append(my_dict_y[ajs1])
                    des = all_anjisuan1[ajs1].describe()
                    # 得到每列的平均值,是一维数组
                    mean = des['mean']
                    # 得到每列的标准差,是一维数组
                    std = des['std']
                    numll = des['count']
                    numll = math.sqrt(numll)
                    qz0 = mean - 1.96 * std / numll
                    qz5 = mean + 1.96 * std / numll
                    q1 = des['25%']
                    q2 = des['50%']
                    q3 = des['75%']
                    IQR = q3 - q1
                    q0 = q1 - 1.5 * IQR
                    q5 = q3 + 1.5 * IQR
                    # 置信区间上下限制
                    res.append(qz0)
                    res.append(qz5)
                    # 箱型图下线
                    res.append(q0)
                    # 上四分卫 中位数 下四分
                    res.append(q1)
                    res.append(q2)
                    res.append(q3)
                    # 箱型图上线
                    res.append(q5)
                    # sql_query1 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' <= '+str(q0)+';'
                    # X1 = pd.read_sql_query(sql_query1, engine)
                    # X1 = np.array(X1)  # 先将数据框转换为数组
                    # X1 = X1.tolist()  # 其次转换为列表
                    # res.append(X1)
                    # sql_query2 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' >= '+str(q5)+';'
                    # X2 = pd.read_sql_query(sql_query2, engine)
                    # X2 = np.array(X2)  # 先将数据框转换为数组
                    # X2 = X2.tolist()
                    # res.append(X2)

                X = [res[i:i + 8] for i in range(0, len(res), 8)]
                X1 = pd.DataFrame(X)
                a = None
                filename1 = english_chinese(mr, ajs, a, mama, '', '')
                filename1 = '删除妈妈后统计数据-'+filename1
                filepath = 'static/picdata/'
                X1.to_csv(str(filepath)  + str(filename1) + '.csv', encoding='utf-8-sig')

                filename1 = str(filename1) + '.csv'
                #savepath(filename1,filepath,uid)

    #奶粉
    ajs1 = copy.deepcopy(ajs)
    ajs1.insert(0, 'nf_name')
    if dt == "相对数据":
        all_naifen = NfXiangdui.objects.values_list(*ajs1)
    else:
        all_naifen = NfYuanshi.objects.values_list(*ajs1)



    return all_naifen,all_anjisuan,ma
def chartssql_box(ajs, mr,mama,plot,uid,plot1):

    ma,delmamaid = fundelmamaid(ajs,mr,mama,'原始数据')
    all_anjisuan = delmamasql(ajs,mr,delmamaid,mama,plot,'原始数据')
    if plot1 == True:
                res = []
                res.append('氨基酸')
                res.append('置信区间下限制')
                res.append('置信区间上限制')
                res.append('箱型图下限')
                res.append('上四分')
                res.append('中位数')
                res.append('下四分')
                res.append('箱型图上限')
                my_dict_y = {"fbaa": "非必须必总氨基酸","baa": "必须必总氨基酸", "lai": "赖氨酸", "su": "苏氨酸", "jie": "缬氨酸", "dan": "蛋氨酸", "yiliang": "异亮氨酸",
                             "liang": "亮氨酸",
                             "benbing": "苯丙氨酸", "se": "色氨酸", "zu": "组氨酸", "tiandong": "天冬氨酸", "si": "丝氨酸",
                             "gu": "谷氨酸", "gan": "甘氨酸", "bing": "丙氨酸", "lao": "酪氨酸", "jing": "精氨酸", "fu": "脯氨酸",
                             "banguang": "半胱氨酸",
                             }

                all_anjisuan1 = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
                for ajs1 in ajs:
                    res.append(my_dict_y[ajs1])
                    des = all_anjisuan1[ajs1].describe()
                    # 得到每列的平均值,是一维数组
                    mean = des['mean']
                    # 得到每列的标准差,是一维数组
                    std = des['std']
                    numll = des['count']
                    numll = math.sqrt(numll)
                    qz0 = mean - 1.96 * std / numll
                    qz5 = mean + 1.96 * std / numll
                    q1 = des['25%']
                    q2 = des['50%']
                    q3 = des['75%']
                    IQR = q3 - q1
                    q0 = q1 - 1.5 * IQR
                    q5 = q3 + 1.5 * IQR
                    # 置信区间上下限制
                    res.append(qz0)
                    res.append(qz5)
                    # 箱型图下线
                    res.append(q0)
                    # 上四分卫 中位数 下四分
                    res.append(q1)
                    res.append(q2)
                    res.append(q3)
                    # 箱型图上线
                    res.append(q5)
                    # sql_query1 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' <= '+str(q0)+';'
                    # X1 = pd.read_sql_query(sql_query1, engine)
                    # X1 = np.array(X1)  # 先将数据框转换为数组
                    # X1 = X1.tolist()  # 其次转换为列表
                    # res.append(X1)
                    # sql_query2 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' >= '+str(q5)+';'
                    # X2 = pd.read_sql_query(sql_query2, engine)
                    # X2 = np.array(X2)  # 先将数据框转换为数组
                    # X2 = X2.tolist()
                    # res.append(X2)

                X = [res[i:i + 8] for i in range(0, len(res), 8)]
                X1 = pd.DataFrame(X)
                a = None
                filename1 = english_chinese(mr, ajs, a, mama, '', '')
                filename1 = '删除妈妈后统计数据-'+filename1
                filepath = 'static/picdata/'
                X1.to_csv(str(filepath)  + str(filename1) + '.csv', encoding='utf-8-sig')

                filename1 = str(filename1) + '.csv'
                #(filename1,filepath,uid)

    #奶粉
    ajs1 = copy.deepcopy(ajs)
    ajs1.insert(0, 'nf_name')
    all_naifen = NfXiangdui.objects.values_list(*ajs1)


    return all_naifen,all_anjisuan,ma,filename1
def score():

    Res.objects.values('distancename','mama1').annotate(nax_mama=Max('mama1'))

def trans(m):
        return list(zip(*m))
def commamadel(X,per):
    mamaidcopy = X
    mamai = pd.DataFrame(mamaidcopy[1:], columns=mamaidcopy[0])
    mamaid =mamai['mother_id']
    X = X[1:]
    Y = X
    naifennum = len(X)
    OUT = []
    for j in range(naifennum):
        for i in range(naifennum):
            Z = []
            Z.append(X[j][6:])
            Z.append(Y[i][6:])
            if i == j:
                OUT.append(0)
                continue
            OUT.append(pairwise_distances(Z, metric="euclidean")[1][0])
    X = [OUT[i:i + 1] for i in range(0, len(OUT), 1)]
    X1 = []
    for i in X:
        i = list(map(float, i))
        X1.append(i)
    X = X1
    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt +1
        if cnt == naifennum:
            temp.append(i)
            temp = trans(temp)
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)
    new_list = Y
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)
    new_list = temp2
    #ave
    new_list1 = []
    for i in new_list:
        temp = 0
        ilen = len(i)
        for j in range(len(i)):
            if j == len(i) - 1:
                new_list1.append(i[j])
                new_list1.append(temp)
            else:
                new_list1.append(i[j])
                temp = temp + i[j]
    iilen = ilen + 1
    new_list_t = [new_list1[i:i + iilen] for i in range(0, len(new_list1), iilen)]
    #max
    averange = []
    delmama = len(new_list_t) * per
    cnt = 0
    for i in new_list_t:
        averange.append(i[len(i) - 1])
        averange.append(mamaid[cnt])
        cnt = cnt + 1

    X = [averange[i:i + 2] for i in range(0, len(averange), 2)]

    X.sort()
    X.reverse()
    delid = []
    cnt = 0
    for a in X:
        cnt = cnt + 1
        if cnt <= int(delmama):
            delid.append(a[1])
        else:
            break
    return mamaid,delid
def comMDS(X,para,naifen_mama):
    if naifen_mama == 'mama':
        X = X[1:]
    Y = X
    naifennum = len(X)
    OUT = []
    for j in range(naifennum):
        for i in range(naifennum):
            Z = []
            if naifen_mama == 'mama':
                Z.append(X[j][6:])
                Z.append(Y[i][6:])
            else:
                Z.append(list(X.iloc[j, :]))
                Z.append(list(Y.iloc[j, :]))

            if i == j:
                OUT.append(0)
                continue
            OUT.append(pairwise_distances(Z, metric="euclidean")[1][0])
    X = [OUT[i:i + 1] for i in range(0, len(OUT), 1)]
    X1 = []
    for i in X:
        i = list(map(float, i))
        X1.append(i)
    X = X1
    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt +1
        if cnt == naifennum:
            temp.append(i)
            temp = trans(temp)
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)
    new_list = Y
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)
    new_list = temp2
    clf2 = MDS(para)
    data = new_list
    clf2.fit(data)
    iris_t2=clf2.fit_transform(data)
    return iris_t2
def compca(X,para,naifen_mama):
    #X = X[,6:]
    if naifen_mama == 'mama':
        X = pd.DataFrame(X[1:], columns=X[0])
        X = X.iloc[:, 6:]




    bnum = para
    pca = PCA(n_components=bnum)
    m = pca.fit_transform(X)
    X = pca.inverse_transform(m)
    a = [[] for i in m[0]]
    for i in m:
        for j in range(len(i)):
            a[j].append(i[j])
    principalComponents = a
    importance = pca.explained_variance_ratio_
    for i in importance:
        print('importance')
        print(i)
    p = trans(principalComponents)
    return p
def plot_3d(all_anjisuan,iris_t2,filename,inf,naifen_mama):
    if naifen_mama == 'mama':
        iris_t2 = pd.DataFrame(iris_t2)
        all_anjisuan = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
        lnum = len(iris_t2)
        data = iris_t2
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        filename1 = filename
        #plt.title(filename1)
        for i in range(lnum):
            if all_anjisuan[inf[0]][i] == 1:
                ax.scatter(data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], marker='o', c='green', )
            else:
                ax.scatter(data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], marker='*', c='red', )
        # for i in range(lnum):
        #     plt.text(data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], all_anjisuan.iloc[i][0], fontsize=5)
        my_dict_x = {"city": "南方绿色 北方红色", "age1": "年龄30以下绿色，30以上红色", "birth_jd": "冬春绿色 夏秋红色",
                     "fenmian_way": "顺产绿色，剖腹产红色",
                     "chanci": "1胎绿色2胎红色"}
        xl = my_dict_x[inf[0]]
        plt.xlabel(xl)
        filepath1 = 'static/pic/'
        plt.savefig(str(filepath1) + '3d' + str(filename1) + '.jpg', dpi=300)
        filename1 = '3d'+str(filename1) + '.jpg'
        plt.savefig('PCA.jpg', dpi=300)

        plt.show()
    else:
        iris_t2 = pd.DataFrame(iris_t2)
        #all_anjisuan = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
        lnum = len(iris_t2)
        data = iris_t2
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        filename1 = filename
        # plt.title(filename1)
        for i in range(lnum):
                ax.scatter(data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], marker='*', c='red', )
        # for i in range(lnum):
        #     plt.text(data.iloc[i][0], data.iloc[i][1], data.iloc[i][2], all_anjisuan.iloc[i][0], fontsize=5)
        # my_dict_x = {"city": "南方绿色 北方红色", "age1": "年龄30以下绿色，30以上红色", "birth_jd": "冬春绿色 夏秋红色",
        #              "fenmian_way": "顺产绿色，剖腹产红色",
        #              "chanci": "1胎绿色2胎红色"}
        # xl = my_dict_x[inf[0]]
        # plt.xlabel(xl)
        filepath1 = 'static/pic/'
        plt.savefig(str(filepath1) + '3d' + str(filename1) + '.jpg', dpi=300)
        plt.savefig('PCA.jpg', dpi=300)
        filename1 = '3d' + str(filename1) + '.jpg'


        plt.show()


    return filename1

def savepath(filename1,filepath1,uid):

    qs = User.objects.filter(userid=uid).first()
    file = File.objects.create(userid = qs, filename=str(filename1) ,
                               filepath = str(filepath1) + str(filename1), active=1)
    file.save()
def plot_2d(all_anjisuan,iris_t2,filename,inf,naifen_mama):
    if naifen_mama == 'mama':
        filename1 = filename
        iris_t2 = pd.DataFrame(iris_t2)
        all_anjisuan = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
        numlem = len(iris_t2)
        for i in range(numlem):
            if all_anjisuan[inf[0]][i] == 1:
                plt.plot(iris_t2.iloc[i][0], iris_t2.iloc[i][1], 'o', color='green')
            else:
                plt.plot(iris_t2.iloc[i][0], iris_t2.iloc[i][1], '*', color='red')
        for i in range(numlem):
            plt.text(iris_t2.iloc[i][0], iris_t2.iloc[i][1],all_anjisuan.iloc[i][0], fontsize=5)

        my_dict_x = {"city": "南方绿色 北方红色", "age1": "年龄30以下绿色，30以上红色", "birth_jd": "冬春绿色 夏秋红色",
                     "fenmian_way": "顺产绿色，剖腹产红色",
                     "chanci": "1胎绿色2胎红色"}
        xl = my_dict_x[inf[0]]
        plt.xlabel(xl)
        filepath = 'static/pic/'
        plt.savefig(str(filepath)+'2d'+str(filename1) + '.jpg', dpi=300)
        plt.savefig('PCA.jpg', dpi=300)
        filename1 ='2d'+ str(filename1) + '.jpg'

        plt.show()
        return filename1
    else:
        filename1 = filename
        iris_t2 = pd.DataFrame(iris_t2)
        #all_anjisuan = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
        numlem = len(iris_t2)
        for i in range(numlem):
            plt.plot(iris_t2.iloc[i][0], iris_t2.iloc[i][1], '*', color='red')
        for i in range(numlem):
            print(i)
            print(all_anjisuan.iloc[i][0])
            plt.text(iris_t2.iloc[i][0], iris_t2.iloc[i][1],all_anjisuan.iloc[i][0], fontsize=5)

        plt.show()
        # my_dict_x = {"city": "南方绿色 北方红色", "age1": "年龄30以下绿色，30以上红色", "birth_jd": "冬春绿色 夏秋红色",
        #              "fenmian_way": "顺产绿色，剖腹产红色",
        #              "chanci": "1胎绿色2胎红色"}
        # xl = my_dict_x[inf[0]]
        # plt.xlabel(xl)
        filepath = 'static/pic/'
        plt.savefig(str(filepath) + '2d' + str(filename1) + '.jpg', dpi=300)
        plt.savefig('PCA.jpg',dpi=300)
        filename1 = '2d' + str(filename1) + '.jpg'
        return filename1




def charts_mds(all_naifen, all_anjisuan, naifen_mama,inf,ajs,filename,ldim,uid):
    if naifen_mama == 'mama':
        id = all_anjisuan
        all_anjisuan1 =all_anjisuan
        a = comMDS(all_anjisuan1, 2,naifen_mama)
    elif naifen_mama =='naifen':
        X = pd.DataFrame(all_naifen)
        id = X.iloc[:, 0:1]
        all_anjisuan1 = X.iloc[:, 1:]
        a = comMDS(all_anjisuan1, 2,naifen_mama)
        inf = ''
    else:
        Y = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
        id1 = Y.iloc[:, 0:1]
        Y = Y.iloc[:, 6:]
        title = list(Y.columns)
        X = pd.DataFrame(all_naifen)
        id2 = X.iloc[:, 0:1]
        id2.columns = ['mother_id']
        X = X.iloc[:, 1:]
        X.columns = title

        all_anjisuan1 = pd.concat([X, Y], axis=0)
        id = pd.concat([id1, id2], axis=0)
        a = comMDS(all_anjisuan1, 2,naifen_mama)
        inf = ''

    f = []
    f2 = plot_2d(id,a,filename,inf,naifen_mama)
    if ldim >= 3:
        a=comMDS(all_anjisuan1, 3,naifen_mama)
        f3 = plot_3d(id, a,filename, inf,naifen_mama)
        f.append(f3)

    f.append(f2)
    return f
def charts_pca(all_naifen, all_anjisuan, naifen_mama,inf,ajs,filename,ldim,uid):
    if naifen_mama == 'mama':
        id = all_anjisuan
        all_anjisuan1 = all_anjisuan
        a = compca(all_anjisuan1, 2,naifen_mama)
    elif naifen_mama == 'naifen':
        X = pd.DataFrame(all_naifen)
        id = X.iloc[:, 0:1]
        all_anjisuan1 = X.iloc[:, 1:]
        a = compca(all_anjisuan1, 2,naifen_mama)
        inf = ''
    else:
        Y = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
        id1 = Y.iloc[:, 0:1]
        Y = Y.iloc[:, 6:]
        title = list(Y.columns)
        X = pd.DataFrame(all_naifen)
        id2 = X.iloc[:, 0:1]
        id2.columns = ['mother_id']
        X = X.iloc[:, 1:]
        X.columns = title

        all_anjisuan1 = pd.concat([X, Y], axis=0)
        id = pd.concat([id1, id2], axis=0)
        a = compca(all_anjisuan1,2,naifen_mama)
        inf = ''

    f = []
    f2 = plot_2d(id, a, filename, inf, naifen_mama)
    if ldim >= 3:
        a = compca(all_anjisuan1, 3,naifen_mama)
        f3 = plot_3d(id, a, filename, inf, naifen_mama)
        f.append(f3)

    f.append(f2)
    return f
def charts_vio_box(all_naifen, all_anjisuan,x,y,charts,filename,ldim,uid):

    # X = DataFrame(all_anjisuan)
    X = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
    # x 影响因素 y 氨基酸
    g = sns.PairGrid(X,
                     x_vars=x,
                     y_vars=y, palette='GnBu_d',
                     )

    # 下三角绘多 sns.boxplot plt.scatter sns.boxplot sns.violinplot
    if charts == 'box':
        plotway = sns.boxplot
    else:
        plotway = sns.violinplot
    my_dict_x = {"city": "南方1北方2", "age1": "年龄30以下1，30以上2", "birth_jd": "冬春1夏秋2", "fenmian_way": "顺产1，剖腹产2", "chanci": "1胎2胎"}
    xlabels = []
    for i in range(len(x)):
        xlabels.append(my_dict_x[x[i]])
    my_dict_y = {"fbaa": "非必须必总氨基酸","baa": "必须必总氨基酸","lai": "赖氨酸", "su": "苏氨酸", "jie": "缬氨酸","dan": "蛋氨酸","yiliang": "异亮氨酸", "liang": "亮氨酸",
                 "benbing": "苯丙氨酸","se": "色氨酸","zu": "组氨酸","tiandong": "天冬氨酸","si": "丝氨酸",
                 "gu": "谷氨酸","gan": "甘氨酸","bing": "丙氨酸","lao": "酪氨酸", "jing": "精氨酸","fu": "脯氨酸","banguang": "半胱氨酸",
                 }
    ylabels = []
    for i in range(len(y)):
        ylabels.append(my_dict_y[y[i]])
    g.map_diag(plotway,)
    g.map_offdiag(plotway, )
    for i in range(len(x)):
        for j in range(len(y)):
            g.axes[j, i].xaxis.set_label_text(xlabels[i])
            g.axes[j, i].yaxis.set_label_text(ylabels[j])
    plt.show()
    filepath = 'static/pic/'
    g.savefig(str(filepath)+str(filename)+'.jpg', dpi=400)
    g.savefig('box.jpg', dpi=400)
    filename1 = str(filename) + '.jpg'
    #savepath(filename1, filepath, uid)
    return filename1
def charts_vio_box_naifen(all_naifen, all_anjisuan,x,yl,charts,filename,ldim,uid):


    Y = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
    Y = Y.iloc[:, 6:]
    title = list(Y.columns)

    X = pd.DataFrame(all_naifen)
    X = X.iloc[:, 1:]

    #title = pd.DataFrame(title.values.T)
    #res = pd.concat([X, Y], axis=0)
    #Y = pd.DataFrame(X, columns=title)
    X.columns = title

    col_name = X.columns.tolist()  # 将数据框的列名全部提取出来存放在列表里
    col_name.insert(0, 'num')  # 在列索引为2的位置插入一列,列名为:city，刚插入时不会有值，整列都是NaN
    X = X.reindex(columns=col_name)  # DataFrame.reindex() 对原行/列索引重新构建索引值
    X['num'] = [1]*X.shape[0]
    x = ['num']
    #X = pd.DataFrame(X.values.T)
    # x 影响因素 y 氨基酸
    g = sns.PairGrid(X,
                     x_vars=x,
                     y_vars=yl, palette='GnBu_d',
                     )

    # 下三角绘多 sns.boxplot plt.scatter sns.boxplot sns.violinplot
    if charts == 'box':
        plotway = sns.boxplot
    else:
        plotway = sns.violinplot
    # my_dict_x = {"city": "南方1北方2", "age1": "年龄30以下1，30以上2", "birth_jd": "冬春1夏秋2", "fenmian_way": "顺产1，剖腹产2", "chanci": "1胎2胎"}
    # xlabels = []
    # for i in range(len(x)):
    #     xlabels.append(my_dict_x[x[i]])
    my_dict_y = {"fbaa": "非必须必总氨基酸","baa": "必须必总氨基酸","lai": "赖氨酸", "su": "苏氨酸", "jie": "缬氨酸","dan": "蛋氨酸","yiliang": "异亮氨酸", "liang": "亮氨酸",
                 "benbing": "苯丙氨酸","se": "色氨酸","zu": "组氨酸","tiandong": "天冬氨酸","si": "丝氨酸",
                 "gu": "谷氨酸","gan": "甘氨酸","bing": "丙氨酸","lao": "酪氨酸", "jing": "精氨酸","fu": "脯氨酸","banguang": "半胱氨酸",
                 }
    ylabels = []
    for i in range(len(yl)):
        ylabels.append(my_dict_y[yl[i]])
    g.map_diag(plotway,)
    g.map_offdiag(plotway, )
    for i in range(1):
        for j in range(len(yl)):
            #g.axes[j, i].xaxis.set_label_text('')
            g.axes[j, i].yaxis.set_label_text(ylabels[j])
    plt.show()
    filepath = 'static/pic/'
    g.savefig(str(filepath)+str(filename)+'.jpg', dpi=400)
    g.savefig('box.jpg', dpi=400)
    filename1 = str(filename) + '.jpg'
    #savepath(filename1, filepath, uid)
    return filename1
def charts_vio_box_naifen_mama(all_naifen, all_anjisuan,x,yl,charts,filename,ldim,uid):


    Y = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
    Y = Y.iloc[:, 6:]
    title = list(Y.columns)

    X = pd.DataFrame(all_naifen)
    X = X.iloc[:, 1:]

    #title = pd.DataFrame(title.values.T)
    #res = pd.concat([X, Y], axis=0)
    #Y = pd.DataFrame(X, columns=title)
    X.columns = title

    #X = pd.DataFrame(X.values.T)
    # x 影响因素 y 氨基酸
    X = pd.concat([X, Y], axis=0)
    col_name = X.columns.tolist()  # 将数据框的列名全部提取出来存放在列表里
    col_name.insert(0, 'num')  # 在列索引为2的位置插入一列,列名为:city，刚插入时不会有值，整列都是NaN
    X = X.reindex(columns=col_name)  # DataFrame.reindex() 对原行/列索引重新构建索引值
    X['num'] = [1] * X.shape[0]
    x = ['num']
    g = sns.PairGrid(X,
                     x_vars=x,
                     y_vars=yl, palette='GnBu_d',
                     )

    # 下三角绘多 sns.boxplot plt.scatter sns.boxplot sns.violinplot
    if charts == 'box':
        plotway = sns.boxplot
    else:
        plotway = sns.violinplot
    # my_dict_x = {"city": "南方1北方2", "age1": "年龄30以下1，30以上2", "birth_jd": "冬春1夏秋2", "fenmian_way": "顺产1，剖腹产2", "chanci": "1胎2胎"}
    # xlabels = []
    # for i in range(len(x)):
    #     xlabels.append(my_dict_x[x[i]])
    my_dict_y = {"fbaa": "非必须必总氨基酸","baa": "必须必总氨基酸","lai": "赖氨酸", "su": "苏氨酸", "jie": "缬氨酸","dan": "蛋氨酸","yiliang": "异亮氨酸", "liang": "亮氨酸",
                 "benbing": "苯丙氨酸","se": "色氨酸","zu": "组氨酸","tiandong": "天冬氨酸","si": "丝氨酸",
                 "gu": "谷氨酸","gan": "甘氨酸","bing": "丙氨酸","lao": "酪氨酸", "jing": "精氨酸","fu": "脯氨酸","banguang": "半胱氨酸",
                 }
    ylabels = []
    for i in range(len(yl)):
        ylabels.append(my_dict_y[yl[i]])
    g.map_diag(plotway,)
    g.map_offdiag(plotway, )
    for i in range(1):
        for j in range(len(yl)):
            #g.axes[j, i].xaxis.set_label_text('')
            g.axes[j, i].yaxis.set_label_text(ylabels[j])
    plt.show()
    filepath = 'static/pic/'
    g.savefig(str(filepath)+str(filename)+'.jpg', dpi=400)
    g.savefig('box.jpg', dpi=400)
    filename1 = str(filename) + '.jpg'
    #savepath(filename1, filepath, uid)
    return filename1
def english_chinese(miruqi,ajs,inf,mama,charts,duan):
    if duan == "1":
        md = '1段'
    elif duan == "2":
        md = '2段'
    elif duan == "all":
        md = '12段'
    else:
        md = ''

    if miruqi == "3":
        mr = '早期成熟乳'
    else:
        mr = '晚期成熟乳'

    if mama == 'mds10':
        ma = '删除10%妈妈'
    elif mama == 'all':
        ma = '全部妈妈'
    elif mama == 'all-max':
        ma = '全部妈妈-最大值'
    elif mama == 'all-min':
        ma = '全部妈妈-最小值'
    elif mama == 'all-ave':
        ma = '全部妈妈-平均值'
    elif mama == 'all-mid':
        ma = '全部妈妈-中位数'
    elif mama == 'mds10-max':
        ma = '删除10%妈妈-最大值'
    elif mama == 'mds10-min':
        ma = '删除10%妈妈-最小值'
    elif mama == 'mds10-ave':
        ma = '删除10%妈妈-平均值'
    else:
        ma = '删除10%妈妈-中位数'


    if charts == 'box':
        mc = '箱型图'
    elif  charts == 'vio':
        mc = '小提琴图'
    elif  charts == 'mds':
        mc = 'MDS'
    elif charts == 'pca':
        mc = 'PCA'
    else:
        mc = ''

    my_inf = {"city": "南北方", "birth_jd": "季度",
              "age1": "年龄", "fenmian_way": "分娩方式",
              "产次": "chanci", "1": "无",
                   }
    mi = ''
    if inf != None:
        for i in range(len(inf)):
            mi = mi + str(my_inf[inf[i]])
    else:
        mi = ''

    my_dict_y = {"fbaa": "非必须必总氨基酸","baa": "必须必总氨基酸", "lai": "赖氨酸", "su": "苏氨酸", "jie": "缬氨酸", "dan": "蛋氨酸", "yiliang": "异亮氨酸",
                 "liang": "亮氨酸",
                 "benbing": "苯丙氨酸", "se": "色氨酸", "zu": "组氨酸", "tiandong": "天冬氨酸", "si": "丝氨酸",
                 "gu": "谷氨酸", "gan": "甘氨酸", "bing": "丙氨酸", "lao": "酪氨酸", "jing": "精氨酸", "fu": "脯氨酸", "banguang": "半胱氨酸",
                 }
    # ylabels = []
    # for i in range(len(ajs)-1):
    #     ylabels.append(my_dict_y[ajs[i+1]])
    ylabels = ''
    for i in range(len(ajs)):
        ylabels = ylabels + str(my_dict_y[ajs[i]])
    filename = md+'-'+mr+'-'+ma+'-'+mi+'-'+mc+'-'+str(ylabels)
    return filename
def charts(request):
    dt = '原始数据'
    if request.method == "POST":
        filename2 = request.POST.getlist('filename1')
        filename3 = request.POST.getlist('filename2')
        if len(filename2) > 0 and len(filename3) > 0:
            for f1 in filename2:
                filename = f1
                f = open('static/pic/' + filename, 'rb')
                response = FileResponse(f)
                response['Content-Type'] = 'application/octet-stream'
                filename = 'attachment; filename =' + filename
                response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
                return response

        if len(filename2) > 0 and len(filename3) == 0:
            for f1 in filename2:
                filename = f1
                f = open('static/pic/' + filename, 'rb')
                response = FileResponse(f)
                response['Content-Type'] = 'application/octet-stream'
                filename = 'attachment; filename =' + filename
                response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
                return response
        elif len(filename3) > 0 and len(filename2) == 0:
            for f1 in filename3:
                filename = f1
                f = open('static/picdata/' + filename, 'rb')
                response = FileResponse(f)
                response['Content-Type'] = 'application/octet-stream'
                filename = 'attachment; filename =' + filename
                response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
                return response

        uid = request.session.get('uid', default='1')
        # mama = request.POST.get('check_box_list_mama')
        # mama = int(mama)*0.01
        naifen_mama = request.POST.get('naifen_mama')
        miruqi = request.POST.get('check_box_list_miruqi')
        inf = request.POST.getlist("check_box_list_inf")
        ajs = request.POST.getlist("check_box_list_ajs")
        charts = request.POST.get("check_box_list_charts")
        mama = request.POST.get('delmama')
        if naifen_mama == 'pro':
            df = pd.read_csv('F:\mn - 副本\APP\migrations\mama.csv', encoding='utf-8-sig')
            flag1,flag2 = charts_pro(df,charts)
            #charts_pro(df,'violinplot')
            flag = '查询成功'
            return render(request, 'charts.html', locals())
        elif naifen_mama == 'mama':
            if not mama.isdigit():
                flag = '请输入0-10的数字'
                return render(request, 'charts.html', locals())
            mama = int(mama)
            if mama < 0 or mama > 10:
                flag = '请输入0-10的数字'
                return render(request, 'charts.html', locals())
            mama = mama * 0.01
            if len(inf) == 0:
                flag = '请选择影响因素'
                return render(request, 'charts.html', locals())


        else:
            mama = 0
        if len(ajs) == 0:
            flag = '请选择氨基酸'
            return render(request, 'charts.html', locals())
        ajs1 = copy.deepcopy(ajs)
        plot = True
        all_naifen, all_anjisuan,ma ,filename3= chartssql_box(ajs, miruqi ,mama,True,uid,plot)
        filename = english_chinese(miruqi,ajs,inf,mama,charts,'')
        ldim = len(ajs)
        if naifen_mama == 'mama':
            filename1 = charts_vio_box(all_naifen, all_anjisuan,inf,ajs1,charts,filename,ldim,uid)
        elif naifen_mama == 'naifen':
            filename1 = charts_vio_box_naifen(all_naifen, all_anjisuan, inf, ajs1, charts, filename, ldim, uid)
        else:
            filename1 = charts_vio_box_naifen_mama(all_naifen, all_anjisuan, inf, ajs1, charts, filename, ldim, uid)

        flag = '查询成功'
        flag1 = filename1
        flag2 = filename3
        return render(request, 'charts.html', locals())
    return render(request, 'charts.html', locals())

def PCA_MDS(request):
    dt='原始数据'
    if request.method == "POST":
        filename2 = request.POST.getlist('filename2')
        filename3 = request.POST.getlist('filename3')

        if len(filename2) > 0 and len(filename3) == 0:
            for f1 in filename2:
                filename = f1
                f = open('static/pic/' + filename, 'rb')
                response = FileResponse(f)
                response['Content-Type'] = 'application/octet-stream'
                filename = 'attachment; filename =' + filename
                response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
                return response
        elif len(filename3) > 0 and len(filename2) == 0:
            for f1 in filename3:
                filename = f1
                f = open('static/pic/' + filename, 'rb')
                response = FileResponse(f)
                response['Content-Type'] = 'application/octet-stream'
                filename = 'attachment; filename =' + filename
                response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
                return response



        uid = request.session.get('uid', default='1')
        naifen_mama = request.POST.get('naifen_mama')
        # mama = request.POST.get('check_box_list_mama')
        # mama = int(mama) * 0.01
        miruqi = request.POST.get('check_box_list_miruqi')
        inf1 = request.POST.get("check_box_list_inf")
        inf = []
        inf.append(inf1)
        ajs = request.POST.getlist("check_box_list_ajs")
        charts = request.POST.get("check_box_list_charts")
        mama = request.POST.get('delmama')
        if naifen_mama == 'mama':
            if not mama.isdigit():
                flag = '请输入0-10的数字'
                return render(request, 'PCA_MDS.html', locals())
            mama = int(mama)
            if mama < 0 or mama > 10:
                flag = '请输入0-10的数字'
                return render(request, 'PCA_MDS.html', locals())
            mama = mama * 0.01
            if len(inf) == 0:
                flag = '请选择影响因素'
                return render(request, 'PCA_MDS.html', locals())
        else:
            mama = 0
        if len(ajs) == 0:
            flag = '请选择氨基酸'
            return render(request, 'PCA_MDS.html', locals())
        ajs1 = copy.deepcopy(ajs)
        plot = True
        all_naifen, all_anjisuan,ma = chartssql(ajs, miruqi ,mama,True,uid,plot,'原始数据')
        if naifen_mama != 'mama':
            inf = ''
        filename = english_chinese(miruqi,ajs,inf,mama,charts,'')

        ldim = len(ajs)
        if charts == 'mds':
             if naifen_mama == 'mama':
                 if len(inf) >= 2:
                     flag = '绘制MDS图时请选择一个影响因素'
                     return render(request, 'PCA_MDS.html', locals())
             if len(ajs) == 1:
                 flag = '请选择至少两个氨基酸'
                 return render(request, 'PCA_MDS.html', locals())
             if len(ajs) == 2:
                 flag = '请选择至少三个氨基酸，才能绘制三维图'
                 return render(request, 'PCA_MDS.html', locals())
             filename1 = charts_mds(all_naifen, all_anjisuan, naifen_mama,inf,ajs,filename,ldim,uid)
        elif charts == 'pca':
             if naifen_mama == 'mama':
                 if len(inf) >= 2:
                    flag = '绘制PCA图时请选择一个影响因素'
                    return render(request, 'PCA_MDS.html', locals())
             if len(ajs) == 1:
                flag = '请选择至少两个氨基酸'
                return render(request, 'PCA_MDS.html', locals())
             if len(ajs) == 2:
                flag = '请选择至少三个氨基酸，才能绘制三维图'
             filename1 = charts_pca(all_naifen, all_anjisuan, naifen_mama,inf,ajs,filename,ldim,uid)

        flag = '查询成功'
        flag1 = filename1[0]
        flag2 = filename1[1]
        return render(request, 'PCA_MDS.html', locals())
    return render(request, 'PCA_MDS.html', locals())
def oplds(request):
    if request.method == "POST":
        iris = load_iris()
        X = iris['data']

        Y = iris['target']


        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3)

        # 然后对y进行转换（哑变量）
        y_train_labels = pd.get_dummies(y_train)

        # 建模
        n_component = 3
        model = PLSRegression(n_components=n_component)
        model.fit(x_train, y_train_labels)

        #
        x_test_trans = model.transform(x_test)
        VIPs = compute_VIP(x_test, y_test, model.x_rotations_, x_test_trans, n_component)
        plt.bar(np.arange(0, X.shape[1]), VIPs)
        Res.objects.all().delete()

        for i in range(0,4):
            Res.objects.create(naifenname=X[0][i], distancename=VIPs[i],mama1=VIPs[i])





        # 分页
        paginator = Paginator(Res, 12)
        current_page = 1
        page = paginator.page(current_page)
        # 构建page_range
        max_page_count = 11
        max_page_count_half = int(max_page_count / 2)
        # 判断页数是否大于max_page_count
        if paginator.num_pages >= max_page_count:
            # 得出start位置
            if current_page <= max_page_count_half:
                page_start = 1
                page_end = max_page_count + 1
            else:
                if current_page + max_page_count_half + 1 > paginator.num_pages:
                    page_start = paginator.num_pages - max_page_count
                    page_end = paginator.num_pages + 1
                else:
                    page_start = current_page - max_page_count_half
                    page_end = current_page + max_page_count_half + 1
            page_range = range(page_start, page_end)
        else:
            page_range = paginator.page_range
        # 分页结束
        flag = '查询成功'
        return render(request, 'oplds.html', locals())

    return render(request, 'oplds.html', locals())
# 距离
def searchdis(request):
    uid = request.session.get('uid', default='1')
    current_page = request.GET.get("page")
    ajs_def = ['lai','se','fu']
    mydim = request.session.get('mydim', default=ajs_def)
    mymr = request.session.get('mymr', default=3)
    mymama = request.session.get('mymama', default=0)
    my12 = request.session.get('my12', default='all')
    mydischance = request.session.get('mydischance', default='欧氏距离')
    mydatetype = request.session.get('mydatatype', default='相对')
    seqnum = 1
    request.session['is_login'] = True
    request.session.set_expiry(0)


    if request.method == "POST":
        ##down

        filename1 = request.POST.getlist('filename')
        mydischance = request.POST.getlist('dis')
        mydatatype = request.POST.getlist('datatype')

        if len(filename1) > 0:
            for f1 in filename1:
                filename = f1
                f = open('static/distancefile/' + filename, 'rb')
                response = FileResponse(f)
                response['Content-Type'] = 'application/octet-stream'
                filename = 'attachment; filename =' + filename
                response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
                return response
        else:
                flag='下载失败'

        ajs1 = request.POST.getlist("check_box_list_ajs")
        miruqi = request.POST.get('check_box_list_miruqi')
        mama = request.POST.get('delmama')

        if not mama.isdigit():
            flag = '请输入0-10的数字'
            return render(request, 'searchdis.html', locals())
        mama = int(mama)
        if mama < 0 or mama > 10:
            flag = '请输入0-10的数字'
            return render(request, 'searchdis.html', locals())
        mama = mama*0.01
        duan = request.POST.get('check_box_list_12')

        if len(ajs1) == 0:
            flag = '请选择氨基酸'
            return render(request, 'searchdis.html', locals())
        dim = len(ajs1)
        current_page = 1
        request.session['mydim'] = ajs1
        request.session['mymr'] = miruqi
        request.session['mymama'] = mama
        request.session['my12'] = duan
        request.session['mydischance'] = mydischance
        request.session['mydatatype'] = mydatatype
        plot = False
        all_naifen, all_anjisuan,ma = chartssql(ajs1, miruqi,mama,False,uid,plot,mydatatype[0])

        new_list = distanceCompute(all_naifen, all_anjisuan,dim)
        Res.objects.all().delete()

        for i in new_list:
            ave = np.mean(i[2:])
            Res.objects.create(naifenname=i[0], distancename=i[1], mama1=ave)

        #计算分数开始
        #score()
        #(1-num)/max*100

        ll_res1 = Res.objects.values('distancename').annotate(nax_mama=Min('mama1'))
        ll_res2 = Res.objects.values('distancename').annotate(nax_mama=Max('mama1'))
        plus = []
        for ll_res11 in ll_res1:
            for ll_res22 in ll_res2:
                if ll_res11['distancename'] == ll_res22['distancename'] and ll_res11['distancename'] != '氨基酸比值系数':
                    plus.append((ll_res22['nax_mama']-ll_res11['nax_mama'])/20)
        cnt = 0
        for ll_res in ll_res2:
            if ll_res['distancename'] != '氨基酸比值系数':
                reporter = Res.objects.filter(distancename=ll_res['distancename'])
                reporter.update(mama1=70+((1-F('mama1'))-(1-ll_res['nax_mama']))/plus[cnt])
                cnt = cnt + 1

        #d1 = Res.objects.values('distancename').annotate((1-'mama1')/Max('mama1')*100)
        #计算分数结束
        if duan == 'all':
            all_res = Res.objects.filter(Q(distancename__exact=mydischance[0])).order_by('-mama1')
        else:
            all_res = Res.objects.filter(Q(naifenname__icontains=duan)&Q(distancename__exact=mydischance[0])).order_by('-mama1')
        temp = []
        for anjisuan in all_res:
            a = anjisuan
            temp.append(a.naifenname)
            temp.append(a.distancename)
            temp.append(a.mama1)
        all_anjisuan = [temp[i:i + 3] for i in range(0, len(temp), 3)]
        all_res1 = pd.DataFrame(all_anjisuan,columns=['奶粉名称','距离名称','得分'])
        a = None
        filename1 = english_chinese(miruqi,ajs1,a,mama,'',duan)
        filename1 = '距离计算-'+filename1
        filepath = 'static/distancefile/'
        all_res1.to_csv(str(filepath)+str(filename1)+'.csv',encoding='utf-8-sig')
        filename1 = str(filename1) + '.csv'
        #(filename1, filepath,uid)
        flag1 = filename1
        # 分页
        paginator = Paginator(all_res, 12)
        current_page = 1
        page = paginator.page(current_page)
        # 构建page_range
        max_page_count = 11
        max_page_count_half = int(max_page_count / 2)
        # 判断页数是否大于max_page_count
        if paginator.num_pages >= max_page_count:
            # 得出start位置
            if current_page <= max_page_count_half:
                page_start = 1
                page_end = max_page_count + 1
            else:
                if current_page + max_page_count_half + 1 > paginator.num_pages:
                    page_start = paginator.num_pages - max_page_count
                    page_end = paginator.num_pages + 1
                else:
                    page_start = current_page - max_page_count_half
                    page_end = current_page + max_page_count_half + 1
            page_range = range(page_start, page_end)
        else:
            page_range = paginator.page_range
        # 分页结束
        flag='查询成功'
        return render(request, 'searchdis.html', locals())

    else:
        if current_page is not  None:
            if mydim[0] == 'nf_name':
                mydim = mydim[1:]

            plot = False

            # 计算分数结束

            if my12 == 'all':
                all_res = Res.objects.filter(Q(distancename__exact=mydischance[0])).order_by('-mama1')
            else:
                all_res = Res.objects.filter(
                    Q(naifenname__icontains=my12) & Q(distancename__exact=mydischance[0])).order_by('-mama1')

            #分页
            paginator = Paginator(all_res, 12)
            current_page = int(request.GET.get("page", 1))
            page = paginator.page(current_page)
            # 构建page_range
            max_page_count = 11
            max_page_count_half = int(max_page_count / 2)
            # 判断页数是否大于max_page_count
            if paginator.num_pages >= max_page_count:
                # 得出start位置
                if current_page <= max_page_count_half:
                    page_start = 1
                    page_end = max_page_count + 1
                else:
                    if current_page + max_page_count_half + 1 > paginator.num_pages:
                        page_start = paginator.num_pages - max_page_count
                        page_end = paginator.num_pages + 1
                    else:
                        page_start = current_page - max_page_count_half
                        page_end = current_page + max_page_count_half + 1
                page_range = range(page_start, page_end)
            else:
                page_range = paginator.page_range
            # 分页结束
            flag = '查询成功'
            return render(request, 'searchdis.html', locals())
        else:
            #flag = 'welcome'
            return render(request, 'searchdis.html', locals())
def disnew(request):
    uid = request.session.get('uid', default='1')
    current_page = request.GET.get("page")
    ajs_def = ['lai','se','fu']
    mydim = request.session.get('mydim', default=ajs_def)
    mymr = request.session.get('mymr', default=3)

    my12 = request.session.get('my12', default='all')
    mydischance = request.session.get('mydischance', default='欧氏距离')
    mydatetype = request.session.get('mydatatype', default='相对')
    seqnum = 1
    request.session['is_login'] = True
    request.session.set_expiry(0)


    if request.method == "POST":
        ##down

        filename1 = request.POST.getlist('filename')
        mydischance = request.POST.getlist('dis')
        mydatatype = request.POST.getlist('datatype')

        if len(filename1) > 0:
            for f1 in filename1:
                filename = f1
                f = open('static/distancefile/' + filename, 'rb')
                response = FileResponse(f)
                response['Content-Type'] = 'application/octet-stream'
                filename = 'attachment; filename =' + filename
                response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
                return response
        else:
                flag='下载失败'

        ajs1 = request.POST.getlist("check_box_list_ajs")
        miruqi = request.POST.get('check_box_list_miruqi')



        duan = request.POST.get('check_box_list_12')

        if len(ajs1) == 0:
            flag = '请选择氨基酸'
            return render(request, 'disnew.html', locals())
        dim = len(ajs1)
        current_page = 1
        request.session['mydim'] = ajs1
        request.session['mymr'] = miruqi

        request.session['my12'] = duan
        request.session['mydischance'] = mydischance
        request.session['mydatatype'] = mydatatype
        plot = False
        #all_naifen, all_anjisuan,ma = chartssql(ajs1, miruqi,mama,False,uid,plot,mydatatype[0])

        if mydatatype == "相对数据":
            all_naifen = NfXiangdui.objects.values_list(*ajs1)
            all_anjisuan = NfXiangdui.objects.values_list(*ajs1)
        else:
            all_naifen = NfYuanshi.objects.values_list(*ajs1)
            all_anjisuan = NfYuanshi.objects.values_list(*ajs1)

        # #all_naifen1, all_anjisuan1, ma = chartssql(ajs1, miruqi, 0.1, False, uid, plot, mydatatype[0])
        # all_anjisuan1 = delmamasql(ajs1, 3, [], 0.1, False, '相对数据')
        # new_list = distanceCompute(all_naifen, all_anjisuan1,dim)
        #new_list = Res.objects.all()
        #new_list = Res.objects.filter(Q(distancename__exact='欧氏距离'))

        # for i in new_list:
        #     ave = np.mean(i[2:])
        #     Res.objects.create(naifenname=i[0], distancename=i[1], mama1=ave)

        #计算分数开始
        #score()
        #(1-num)/max*100

        ll_res1 = Res.objects.values('distancename').annotate(nax_mama=Min('mama1'))
        ll_res2 = Res.objects.values('distancename').annotate(nax_mama=Max('mama1'))
        plus = []
        for ll_res11 in ll_res1:
            for ll_res22 in ll_res2:
                if ll_res11['distancename'] == ll_res22['distancename'] and ll_res11['distancename'] != '氨基酸比值系数':
                    plus.append((ll_res22['nax_mama']-ll_res11['nax_mama'])/20)
        cnt = 0
        for ll_res in ll_res2:
            if ll_res['distancename'] != '氨基酸比值系数':
                reporter = Res.objects.filter(distancename=ll_res['distancename'])
                reporter.update(mama1=70+((1-F('mama1'))-(1-ll_res['nax_mama']))/plus[cnt])
                cnt = cnt + 1

        #d1 = Res.objects.values('distancename').annotate((1-'mama1')/Max('mama1')*100)
        #计算分数结束
        if duan == 'all':
            all_res = Res.objects.filter(Q(distancename__exact=mydischance[0])).order_by('-mama1')
        else:
            all_res = Res.objects.filter(Q(naifenname__icontains=duan)&Q(distancename__exact=mydischance[0])).order_by('-mama1')
        temp = []
        for anjisuan in all_res:
            a = anjisuan
            temp.append(a.naifenname)
            temp.append(a.distancename)
            temp.append(a.mama1)
        all_anjisuan = [temp[i:i + 3] for i in range(0, len(temp), 3)]
        all_res1 = pd.DataFrame(all_anjisuan,columns=['奶粉名称','距离名称','得分'])
        a = None
        filename1 = english_chinese(miruqi,ajs1,a,mama,'',duan)
        filename1 = '距离计算-'+filename1
        filepath = 'static/distancefile/'
        all_res1.to_csv(str(filepath)+str(filename1)+'.csv',encoding='utf-8-sig')
        filename1 = str(filename1) + '.csv'
        #(filename1, filepath,uid)
        flag1 = filename1
        # 分页
        paginator = Paginator(all_res, 12)
        current_page = 1
        page = paginator.page(current_page)
        # 构建page_range
        max_page_count = 11
        max_page_count_half = int(max_page_count / 2)
        # 判断页数是否大于max_page_count
        if paginator.num_pages >= max_page_count:
            # 得出start位置
            if current_page <= max_page_count_half:
                page_start = 1
                page_end = max_page_count + 1
            else:
                if current_page + max_page_count_half + 1 > paginator.num_pages:
                    page_start = paginator.num_pages - max_page_count
                    page_end = paginator.num_pages + 1
                else:
                    page_start = current_page - max_page_count_half
                    page_end = current_page + max_page_count_half + 1
            page_range = range(page_start, page_end)
        else:
            page_range = paginator.page_range
        # 分页结束
        flag='查询成功'
        return render(request, 'disnew.html', locals())

    else:
        if current_page is not  None:
            if mydim[0] == 'nf_name':
                mydim = mydim[1:]

            plot = False

            # 计算分数结束

            if my12 == 'all':
                all_res = Res.objects.filter(Q(distancename__exact=mydischance[0])).order_by('-mama1')
            else:
                all_res = Res.objects.filter(
                    Q(naifenname__icontains=my12) & Q(distancename__exact=mydischance[0])).order_by('-mama1')

            #分页
            paginator = Paginator(all_res, 12)
            current_page = int(request.GET.get("page", 1))
            page = paginator.page(current_page)
            # 构建page_range
            max_page_count = 11
            max_page_count_half = int(max_page_count / 2)
            # 判断页数是否大于max_page_count
            if paginator.num_pages >= max_page_count:
                # 得出start位置
                if current_page <= max_page_count_half:
                    page_start = 1
                    page_end = max_page_count + 1
                else:
                    if current_page + max_page_count_half + 1 > paginator.num_pages:
                        page_start = paginator.num_pages - max_page_count
                        page_end = paginator.num_pages + 1
                    else:
                        page_start = current_page - max_page_count_half
                        page_end = current_page + max_page_count_half + 1
                page_range = range(page_start, page_end)
            else:
                page_range = paginator.page_range
            # 分页结束
            flag = '查询成功'
            return render(request, 'disnew.html', locals())
        else:
            #flag = 'welcome'
            return render(request, 'disnew.html', locals())
#妈妈
def mama(request):
            uid = request.session.get('uid', default='1')
            current_page = request.GET.get("page")
            ajs_def = ['lai', 'se', 'fu']
            mydim = request.session.get('mydim', default=ajs_def)
            mymr = request.session.get('mymr', default=3)
            mymama = request.session.get('mymama', default=0)
            myinf = request.session.get('myinf', default="city")
            myway = request.session.get('myway', default='c')
            seqnum = 1
            request.session['is_login'] = True
            request.session.set_expiry(0)

            if request.method == "POST":
                filename1 = request.POST.getlist('filename')
                mydischance = request.POST.getlist('dis')

                if len(filename1) > 0:
                    for f1 in filename1:
                        filename = f1
                        f = open('static/distancefile/' + filename, 'rb')
                        response = FileResponse(f)
                        response['Content-Type'] = 'application/octet-stream'
                        filename = 'attachment; filename =' + filename
                        response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
                        return response
                else:
                    flag = '下载失败'

                ajs1 = request.POST.getlist("check_box_list_ajs")
                miruqi = request.POST.get('check_box_list_miruqi')
                mama = request.POST.get('delmama')
                way = request.POST.get('check_box_com')

                if not mama.isdigit():
                    flag = '请输入0-10的数字'
                    return render(request, 'mama.html', locals())
                mama = int(mama)
                if mama < 0 or mama > 10:
                    flag = '请输入0-10的数字'
                    return render(request, 'mama.html', locals())
                mama = mama * 0.01

                if len(ajs1) == 0:
                    flag = '请选择氨基酸'
                    return render(request, 'mama.html', locals())
                dim = len(ajs1)
                current_page = 1
                request.session['mydim'] = ajs1
                request.session['mymr'] = miruqi
                request.session['mymama'] = mama
                request.session['mydischance'] = mydischance

                request.session['myway'] = way
                if way == 'c':
                    plot = False
                    all_naifen, all_anjisuan,ma = chartssql(ajs1, miruqi, mama, False, uid, plot)
                    new_list = distanceComputemama(all_anjisuan, all_anjisuan, dim,ma)
                    Res.objects.all().delete()

                    for i in new_list:
                        Res.objects.create(naifenname=i[2], distancename=i[0], mama1=i[1])

                    all_res = Res.objects.order_by('distancename', 'mama1')
                    all_res = Res.objects.filter(Q(distancename__exact=mydischance[0])).order_by('-mama1')

                    temp = []
                    for anjisuan in all_res:
                        a = anjisuan
                        temp.append(a.naifenname)
                        temp.append(a.distancename)
                        temp.append(a.mama1)
                    all_anjisuan = [temp[i:i + 3] for i in range(0, len(temp), 3)]
                    all_res1 = pd.DataFrame(all_anjisuan, columns=['名称', '距离名称', '得分'])
                    a = None
                    filename1 = english_chinese(miruqi, ajs1, a, mama, '','' )
                    filename1 = '全部妈妈距离计算-' + filename1
                    filepath = 'static/distancefile/'

                    all_res1.to_csv(str(filepath)+str(filename1) + '.csv', encoding='utf-8-sig')
                    filename1 = str(filename1) + '.csv'
                    #(filename1, filepath, uid)
                elif way == 'a':
                    plot = False
                    all_naifen, all_anjisuan, ma = chartssql(ajs1, miruqi, mama, False, uid, plot)
                    new_list = distanceComputemama_3(all_anjisuan, all_anjisuan, dim, ma)

                    all_res1 = pd.DataFrame(new_list, columns=ajs1,index=['平均值','最小值','中位数','最大值'])
                    a = None
                    filename1 = english_chinese(miruqi, ajs1, a, mama, '', '')
                    filename1 = '最大最小和均值妈妈距离计算-' + filename1
                    filepath = 'static/distancefile/'

                    all_res1.to_csv(str(filepath) + str(filename1) + '.csv', encoding='utf-8-sig')
                    filename1 = str(filename1) + '.csv'
                    #(filename1, filepath, uid)
                    all_res = ''
                elif way == 'pro':
                    df = pd.read_csv('F:\mn - 副本\APP\migrations\mama.csv', encoding='utf-8-sig')
                    data = distancepro(df, df)
                    filepath = 'static/distancefile/'
                    data.to_csv(str(filepath) +'pro.csv', encoding='utf-8-sig')
                    filename1 = 'pro.csv'
                    all_res = ''
                    Res.objects.all().delete()
                    for i in data:
                        i = list(data.loc[i, :])
                        #ave = int(i[-1])/126
                        ave = np.mean(i[3:])
                        Res.objects.create(naifenname=i[0], distancename=i[1], mama1=ave)
                    all_res = Res.objects.order_by('distancename', 'mama1')
                    all_res = Res.objects.filter(Q(distancename__exact=mydischance[0])).order_by('-mama1')
                else:
                    plot = False
                    inf = request.POST.get("check_box_list_inf")
                    request.session['myinf'] = inf
                    all_naifen, all_anjisuan,ma = chartssql_inf(ajs1, miruqi, mama, False, uid, inf)
                    all_anjisuan = pd.DataFrame(all_anjisuan[1:], columns=all_anjisuan[0])
                    all_anjisuan_1 = all_anjisuan[all_anjisuan[inf] == 1]
                    all_anjisuan_1 = all_anjisuan_1.iloc[:, 6:]
                    all_anjisuan_2 = all_anjisuan[all_anjisuan[inf] == 2]
                    all_anjisuan_2 = all_anjisuan_2.iloc[:, 6:]
                    ###

                    my_inf = {"city": ['南方','北方'], "age1": ['30以下','30以上2'], "birth_jd": ["冬春","夏秋"],
                                 "fenmian_way": ["顺产","剖腹产"], "chanci": ["1胎","2胎"]}


                    new_list = distanceComputemama(all_anjisuan_1, all_anjisuan_1, dim,ma)
                    Res.objects.all().delete()
                    for i in new_list:
                        Res.objects.create(naifenname=i[2], distancename=i[0], mama1=i[1])
                    all_res = Res.objects.order_by('distancename', 'mama1')
                    all_res = Res.objects.filter(Q(distancename__exact=mydischance[0])).order_by('-mama1')
                    temp = []
                    for anjisuan in all_res:
                        a = anjisuan
                        temp.append(a.naifenname)
                        temp.append(a.distancename)
                        temp.append(a.mama1)
                        temp.append(my_inf[inf][0])
                    all_anjisuan = [temp[i:i + 4] for i in range(0, len(temp), 4)]
                    all_res1 = pd.DataFrame(all_anjisuan, columns=['名称', '距离名称', '得分','影响因素'])




                    new_list = distanceComputemama(all_anjisuan_2, all_anjisuan_2, dim, ma)
                    Res.objects.all().delete()
                    for i in new_list:
                        Res.objects.create(naifenname=i[2], distancename=i[0], mama1=i[1])
                    all_res = Res.objects.order_by('distancename', 'mama1')
                    temp = []
                    for anjisuan in all_res:
                        a = anjisuan
                        temp.append(a.naifenname)
                        temp.append(a.distancename)
                        temp.append(a.mama1)
                        temp.append(my_inf[inf][1])
                    all_anjisuan = [temp[i:i + 4] for i in range(0, len(temp), 4)]
                    all_res2 = pd.DataFrame(all_anjisuan,columns=['名称', '距离名称', '得分', '影响因素'])
                    all_res1 = pd.concat([all_res2, all_res1], axis=0)
                    filename1 = english_chinese(miruqi, ajs1, '', mama, '', '')
                    filepath = 'static/distancefile/'
                    all_res1.to_csv(str(filepath) + str(filename1) + '.csv', encoding='utf-8-sig')
                    filename1 = str(filename1) + '.csv'
                    # savepath(filename1, filepath, uid)


                flag1 = filename1


                # 分页
                paginator = Paginator(all_res, 12)
                current_page = 1
                page = paginator.page(current_page)
                # 构建page_range
                max_page_count = 11
                max_page_count_half = int(max_page_count / 2)
                # 判断页数是否大于max_page_count
                if paginator.num_pages >= max_page_count:
                    # 得出start位置
                    if current_page <= max_page_count_half:
                        page_start = 1
                        page_end = max_page_count + 1
                    else:
                        if current_page + max_page_count_half + 1 > paginator.num_pages:
                            page_start = paginator.num_pages - max_page_count
                            page_end = paginator.num_pages + 1
                        else:
                            page_start = current_page - max_page_count_half
                            page_end = current_page + max_page_count_half + 1
                    page_range = range(page_start, page_end)
                else:
                    page_range = paginator.page_range
                # 分页结束
                flag = '查询成功'
                return render(request, 'mama.html', locals())

            else:
                if current_page is not None:


                    all_res = Res.objects.order_by('distancename', 'mama1')
                   # 分页
                    paginator = Paginator(all_res, 12)
                    current_page = int(request.GET.get("page", 1))
                    page = paginator.page(current_page)
                    # 构建page_range
                    max_page_count = 11
                    max_page_count_half = int(max_page_count / 2)
                    # 判断页数是否大于max_page_count
                    if paginator.num_pages >= max_page_count:
                        # 得出start位置
                        if current_page <= max_page_count_half:
                            page_start = 1
                            page_end = max_page_count + 1
                        else:
                            if current_page + max_page_count_half + 1 > paginator.num_pages:
                                page_start = paginator.num_pages - max_page_count
                                page_end = paginator.num_pages + 1
                            else:
                                page_start = current_page - max_page_count_half
                                page_end = current_page + max_page_count_half + 1
                        page_range = range(page_start, page_end)
                    else:
                        page_range = paginator.page_range
                    # 分页结束
                    flag = '查询成功'
                    return render(request, 'mama.html', locals())
                else:
                    #flag = 'welcome'
                    return render(request, 'mama.html', locals())




