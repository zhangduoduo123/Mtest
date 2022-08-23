import csv
import os
from datetime import time
import time

import pymysql
from django.contrib import messages
from django.core import paginator
from django.shortcuts import render
from django_pandas.io import read_frame

from APP.models import File, User, Yuanshi, Xiangdui, NfYuanshi, NfXiangdui,Protain
import pandas as pd
from pandas.api.types import *
# Create your views here.
from django.http import HttpResponse, request, response, JsonResponse, FileResponse, StreamingHttpResponse
import copy
import math

import pandas as pd
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.db.models import Q, Avg, Sum, Max, F, Min
from django.http import HttpResponse, request
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from django.shortcuts import render
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances
from APP.models import NfXiangdui, Xiangdui, Res

matplotlib.rc("font", family='YouYuan')


##renlele


def upload(request):
    """
    文件上传
    只允许上传xlsx|xls|csv格式的文件
    :param request:
    :return:
    """
    uname = request.session.get('uname', default="勇敢牛牛")
    uid = request.session.get('uid', default='1')
    now = time.localtime(time.time())
    insert_time = time.strftime("%m%d%H%M", now)

    if request.method == "POST":
        F = request.FILES.get('file1', None)
        f2 = request.FILES.get('file2',None)
        if F is None and f2 is None:
            return render(request, "500.html")
        if F:
            file = str(F)
            file_type = file.split('.')[-1]  # 获得文件类型
            filename = uname + '_' + insert_time + '_' + file
            # 文件类型校验
            if file_type not in ['xlsx', 'xls', 'csv']:
                return render(request, "500.html")
            # 上传本地
            d = open(os.path.join('static/upload_file', filename), 'wb+')
            for chunk in F.chunks():
                d.write(chunk)
                d.close()

            # 文件信息保存到数据库
            filepath = 'static/upload_file/' + filename
            qs = User.objects.filter(userid=uid).first()
            filename = filename.split('_')[2]
            f = File.objects.create(filename=filename, userid=qs, upload_time=insert_time, filepath=filepath)
            f.save()
            file_list = File.objects.filter(userid=uid)
            fid = File.objects.get(filepath=filepath).id
            flag = tosql(uid, uname, fid)
            if flag != 1:
                messages.error(request,flag)
            else:
                messages.success(request,'上传成功')
            return render(request, "index.html", {"file_list": file_list, "uname": uname})
        if f2:
            file = str(f2)
            file_type = file.split('.')[-1]  # 获得文件类型
            filename = uname + '_' + insert_time + '_' + file
            # 文件类型校验
            if file_type not in ['xlsx', 'xls', 'csv']:
                return render(request, "500.html")
            # 上传本地
            d = open(os.path.join('static/upload_file', filename), 'wb+')
            for chunk in f2.chunks():
                d.write(chunk)
                d.close()

            # 文件信息保存到数据库
            filepath = 'static/upload_file/' + filename
            qs = User.objects.filter(userid=uid).first()
            filename = filename.split('_')[2]
            f = File.objects.create(filename=filename, userid=qs, upload_time=insert_time, filepath=filepath)
            f.save()
            file_list = File.objects.filter(userid=uid)
            fid = File.objects.get(filepath=filepath).id
            flag = tosql2(uid, uname, fid)
            if flag != 1:
                messages.error(request,flag)
            else:
                messages.success(request,'上传成功')
            return render(request, "index.html", {"file_list": file_list, "uname": uname})
    else:
        file_list = File.objects.filter(userid=uid)
        return render(request, "index.html", {"file_list": file_list, "uname": uname})


def file_iterator(file_name):
    with open(file_name, 'rb') as f:  # 切记open打开文件的方式
        while True:
            c = f.read()
            if c:
                yield c
            else:
                break


def deldownfile(request):
    """
    删除或下载文件
    :return:
    """
    uid = request.session.get('uid', default=1)
    uname = request.session.get('uname', default='勇敢牛牛')
    if request.method == 'POST':
        fid = request.POST.get('fid')
        if 'del' in request.POST:  # 删除
            # 从本地删除
            filepath = File.objects.filter(id=fid).get().filepath
            os.remove(os.path.join(filepath))
            # 从数据库中删除
            file = File.objects.filter(id=fid)
            file.delete()
        if 'down' in request.POST:  # 下载
            filepath = File.objects.get(id=fid).filepath
            filename = filepath.split('/')[2]
            response = StreamingHttpResponse(file_iterator(filepath))
            response['Content-Type'] = 'application/octet-stream'
            filename = 'attachment; filename =' + filename
            response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
            return response



    file_list = File.objects.filter(userid=uid)
    return render(request, "index.html", {"file_list": file_list, "uname": uname})


def down(request):
    uid = request.session.get('uid', default=1)
    uname = request.session.get('uname', default='勇敢牛牛')

    if request.method == "GET":
        f = open('static/upload_file/样表.xlsx', 'rb')
        response = FileResponse(f)
        filename = '样表.xlsx'
        response['Content-Type'] = 'application/octet-stream'
        filename = 'attachment; filename =' + filename
        response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
        return response
    file_list = File.objects.filter(userid=uid)
    return render(request, "index.html", {"file_list": file_list, "uname": uname})


def saveSql(data, flag, uid):
    qs = User.objects.filter(userid=uid).first()
    # Nan 值填充
    data = data.fillna(value=-1)
    if flag == 'mr':
        # 清除原始数据表，将新数据存到原始含量表
        ys = Yuanshi.objects.filter(insertpeople=uid)
        ys.delete()

        for i in range(len(data)):
            ys = Yuanshi.objects.create(mother_id=data.at[i, 'mother_id'], city=data.at[i, 'city'],
                                        mother_name=data.at[i, 'mother_name'], nfamily=data.at[i, 'nfamily'],
                                        nchild=data.at[i, 'nchild'], minzu=data.at[i, 'minzu'],
                                        age=data.at[i, 'age'], age1=data.at[i, 'age1'], chanci=data.at[i, 'chanci'],
                                        tall=data.at[i, 'tall'], weight1=data.at[i, 'weight1'],
                                        weight2=data.at[i, 'weight2'],
                                        weight3=data.at[i, 'weight3'], fenmian_way=data.at[i, 'fenmian_way'],
                                        birthday=data.at[i, 'birthday'],
                                        birth_jd=data.at[i, 'birth_jd'], birthweight=data.at[i, 'birthweight'],
                                        birthtall=data.at[i, 'birthtall'], sex=data.at[i, 'sex'],
                                        birth_suit=data.at[i, 'birth_suit'],
                                        yangben_id=data.at[i, 'yangben_id'], pro=data.at[i, 'pro'],
                                        day=data.at[i, 'day'],
                                        miruqi=data.at[i, 'miruqi'], caiyang_time=data.at[i, 'caiyang_time'],
                                        lai=data.at[i, 'lai'],
                                        su=data.at[i, 'su'], jie=data.at[i, 'jie'], dan=data.at[i, 'dan'],
                                        yiliang=data.at[i, 'yiliang'], liang=data.at[i, 'liang'],
                                        benbing=data.at[i, 'benbing'],
                                        se=data.at[i, 'se'], zu=data.at[i, 'zu'], tiandong=data.at[i, 'tiandong'],
                                        si=data.at[i, 'si'], gu=data.at[i, 'gu'], gan=data.at[i, 'gan'],
                                        bing=data.at[i, 'bing'], lao=data.at[i, 'lao'], jing=data.at[i, 'jing'],
                                        fu=data.at[i, 'fu'], banguang=data.at[i, 'banguang'], sumaa=data.at[i, 'sumaa'],
                                        baa=data.at[i, 'baa'], fbaa=data.at[i, 'fbaa'], b2sum=data.at[i, 'b2sum'],
                                        b2fb=data.at[i, 'b2fb'], fb2sum=data.at[i, 'fb2sum'], insertpeople=qs)
            ys.save()
        # 按照必须氨基酸总和排序
        data.sort_values(by='baa', inplace=True)
        # 计算相对含量
        data['lai'] = data['lai'] / data['sumaa']
        data['su'] = data['su'] / data['sumaa']
        data['jie'] = data['jie'] / data['sumaa']
        data['dan'] = data['dan'] / data['sumaa']
        data['yiliang'] = data['yiliang'] / data['sumaa']
        data['liang'] = data['liang'] / data['sumaa']
        data['benbing'] = data['benbing'] / data['sumaa']
        data['se'] = data['se'] / data['sumaa']
        data['zu'] = data['zu'] / data['sumaa']
        data['tiandong'] = data['tiandong'] / data['sumaa']
        data['si'] = data['si'] / data['sumaa']
        data['gu'] = data['gu'] / data['sumaa']
        data['gan'] = data['gan'] / data['sumaa']
        data['bing'] = data['bing'] / data['sumaa']
        data['lao'] = data['lao'] / data['sumaa']
        data['jing'] = data['jing'] / data['sumaa']
        data['fu'] = data['fu'] / data['sumaa']
        data['banguang'] = data['banguang'] / data['sumaa']
        data['sumaa'] = data['lai'] + data['su'] + data['jie'] + data['dan'] + data['yiliang'] + data['liang'] + data[
            'benbing'] + data['se'] + data['zu'] + data['tiandong'] + data['si'] + data['gu'] + data['gan'] + data[
                            'bing'] + data['lao'] + data['jing'] + data['fu'] + data['banguang']
        data['baa'] = data['lai'] + data['su'] + data['jie'] + data['dan'] + data['yiliang'] + data['liang'] + data[
            'benbing'] + data['se'] + data['zu']
        data['fbaa'] = data['tiandong'] + data['si'] + data['gu'] + data['gan'] + data['bing'] + data['lao'] + data[
            'jing'] + data['fu'] + data['banguang']
        data['b2fb'] = data['baa'] / data['fbaa']
        data['b2sum'] = data['baa'] / data['sumaa']
        data['fb2sum'] = data['fbaa'] / data['sumaa']

        # 将相对含量存到数据库
        xd = Xiangdui.objects.filter(insertpeople=uid)
        xd.delete()
        for i in range(int(len(data) * 0.05), int(len(data) * 0.95)):
            xd = Xiangdui.objects.create(mother_id=data.at[i, 'mother_id'], city=data.at[i, 'city'],
                                         mother_name=data.at[i, 'mother_name'], nfamily=data.at[i, 'nfamily'],
                                         nchild=data.at[i, 'nchild'], minzu=data.at[i, 'minzu'],
                                         age=data.at[i, 'age'], age1=data.at[i, 'age1'], chanci=data.at[i, 'chanci'],
                                         tall=data.at[i, 'tall'], weight1=data.at[i, 'weight1'],
                                         weight2=data.at[i, 'weight2'],
                                         weight3=data.at[i, 'weight3'], fenmian_way=data.at[i, 'fenmian_way'],
                                         birthday=data.at[i, 'birthday'],
                                         birth_jd=data.at[i, 'birth_jd'], birthweight=data.at[i, 'birthweight'],
                                         birthtall=data.at[i, 'birthtall'], sex=data.at[i, 'sex'],
                                         birth_suit=data.at[i, 'birth_suit'],
                                         yangben_id=data.at[i, 'yangben_id'], pro=data.at[i, 'pro'],
                                         day=data.at[i, 'day'],
                                         miruqi=data.at[i, 'miruqi'], caiyang_time=data.at[i, 'caiyang_time'],
                                         lai=data.at[i, 'lai'],
                                         su=data.at[i, 'su'], jie=data.at[i, 'jie'], dan=data.at[i, 'dan'],
                                         yiliang=data.at[i, 'yiliang'], liang=data.at[i, 'liang'],
                                         benbing=data.at[i, 'benbing'],
                                         se=data.at[i, 'se'], zu=data.at[i, 'zu'], tiandong=data.at[i, 'tiandong'],
                                         si=data.at[i, 'si'], gu=data.at[i, 'gu'], gan=data.at[i, 'gan'],
                                         bing=data.at[i, 'bing'], lao=data.at[i, 'lao'], jing=data.at[i, 'jing'],
                                         fu=data.at[i, 'fu'], banguang=data.at[i, 'banguang'],
                                         sumaa=data.at[i, 'sumaa'],
                                         baa=data.at[i, 'baa'], fbaa=data.at[i, 'fbaa'], b2sum=data.at[i, 'b2sum'],
                                         b2fb=data.at[i, 'b2fb'], fb2sum=data.at[i, 'fb2sum'], insertpeople=qs)
            xd.save()
    else:
        # 清除原始数据表，将新数据存到原始含量表
        ys = NfYuanshi.objects.filter(insertpeople=uid)
        ys.delete()
        for i in range(len(data)):
            ys = NfYuanshi.objects.create(nf_name=data.at[i, 'nf_name'], pro=data.at[i, 'pro'],
                                          lai=data.at[i, 'lai'],
                                          su=data.at[i, 'su'], jie=data.at[i, 'jie'], dan=data.at[i, 'dan'],
                                          yiliang=data.at[i, 'yiliang'], liang=data.at[i, 'liang'],
                                          benbing=data.at[i, 'benbing'],
                                          se=data.at[i, 'se'], zu=data.at[i, 'zu'], tiandong=data.at[i, 'tiandong'],
                                          si=data.at[i, 'si'], gu=data.at[i, 'gu'], gan=data.at[i, 'gan'],
                                          bing=data.at[i, 'bing'], lao=data.at[i, 'lao'], jing=data.at[i, 'jing'],
                                          fu=data.at[i, 'fu'], banguang=data.at[i, 'banguang'],
                                          sumaa=data.at[i, 'sumaa'],
                                          baa=data.at[i, 'baa'], fbaa=data.at[i, 'fbaa'], b2sum=data.at[i, 'b2sum'],
                                          b2fb=data.at[i, 'b2fb'], fb2sum=data.at[i, 'fb2sum'], insertpeople=qs)
            ys.save()

        # 计算相对含量
        data['lai'] = data['lai'] / data['sumaa']
        data['su'] = data['su'] / data['sumaa']
        data['jie'] = data['jie'] / data['sumaa']
        data['dan'] = data['dan'] / data['sumaa']
        data['yiliang'] = data['yiliang'] / data['sumaa']
        data['liang'] = data['liang'] / data['sumaa']
        data['benbing'] = data['benbing'] / data['sumaa']
        data['se'] = data['se'] / data['sumaa']
        data['zu'] = data['zu'] / data['sumaa']
        data['tiandong'] = data['tiandong'] / data['sumaa']
        data['si'] = data['si'] / data['sumaa']
        data['gu'] = data['gu'] / data['sumaa']
        data['gan'] = data['gan'] / data['sumaa']
        data['bing'] = data['bing'] / data['sumaa']
        data['lao'] = data['lao'] / data['sumaa']
        data['jing'] = data['jing'] / data['sumaa']
        data['fu'] = data['fu'] / data['sumaa']
        data['banguang'] = data['banguang'] / data['sumaa']
        data['sumaa'] = data['lai'] + data['su'] + data['jie'] + data['dan'] + data['yiliang'] + data['liang'] + data[
            'benbing'] + data['se'] + data['zu'] + data['tiandong'] + data['si'] + data['gu'] + data['gan'] + data[
                            'bing'] + data['lao'] + data['jing'] + data['fu'] + data['banguang']
        data['baa'] = data['lai'] + data['su'] + data['jie'] + data['dan'] + data['yiliang'] + data['liang'] + data[
            'benbing'] + data['se'] + data['zu']
        data['fbaa'] = data['tiandong'] + data['si'] + data['gu'] + data['gan'] + data['bing'] + data['lao'] + data[
            'jing'] + data['fu'] + data['banguang']
        data['b2fb'] = data['baa'] / data['fbaa']
        data['b2sum'] = data['baa'] / data['sumaa']
        data['fb2sum'] = data['fbaa'] / data['sumaa']

        # 将相对含量存到数据库
        xd = NfXiangdui.objects.filter(insertpeople=uid)
        xd.delete()
        for i in range(len(data)):
            xd = NfXiangdui.objects.create(nf_name=data.at[i, 'nf_name'], lai=data.at[i, 'lai'],
                                           su=data.at[i, 'su'], jie=data.at[i, 'jie'], dan=data.at[i, 'dan'],
                                           yiliang=data.at[i, 'yiliang'], liang=data.at[i, 'liang'],
                                           benbing=data.at[i, 'benbing'], se=data.at[i, 'se'], zu=data.at[i, 'zu'],
                                           tiandong=data.at[i, 'tiandong'],
                                           si=data.at[i, 'si'], gu=data.at[i, 'gu'], gan=data.at[i, 'gan'],
                                           bing=data.at[i, 'bing'], lao=data.at[i, 'lao'], jing=data.at[i, 'jing'],
                                           fu=data.at[i, 'fu'], banguang=data.at[i, 'banguang'],
                                           baa=data.at[i, 'baa'], fbaa=data.at[i, 'fbaa'], b2fb=data.at[i, 'b2fb'],
                                           insertpeople=qs)
            xd.save()

def tosql2(uid,uname,fid):
    """
        检查当前文件格式
            符合格式定义：当前使用数据库的文件状态改为0，本文件状态改为1，并上传至数据库
            不符合格式定义：本文件状态为-1
        """
    qs = User.objects.filter(userid=uid).first()
    file = File.objects.get(id=fid)
    filepath = File.objects.filter(id=fid).get().filepath
    filetype = filepath.split('.')[1]
    # 读取数据
    if filetype == 'csv':
        df = pd.read_csv(filepath, dtype=object)
    elif filetype == 'xlsx':
        df = pd.read_excel(filepath, engine='openpyxl', dtype=object)
    else:
        df = pd.read_excel(filepath,  engine='openpyxl', dtype=object)


    # 检查数据列数
    temp = 1
    if df.shape[1] != 26:
        file.active = -1
        file.save()
        temp = 0
        # messages.error(request, "蛋白质数据列数不满足要求！请检查")

        return "蛋白质数据列数不满足要求！请检查"
    # 检查列名
    order = ['母亲编号', '城市', '母亲姓名', 'NFAMILY', 'NCHILD', '民族', '年龄', '年龄分类',
             '产次', '身高', '孕前体重', '产前体重', '现体重', '分娩方式', '出生日期', '出生季度', '出生体重',
             '出生身长', '性别', '出生情况', '样本编码', '采样次数','泌乳期', '采样天数','α-乳白蛋白',
             'β-酪蛋白']
    order = sorted(order)
    data_col = [column for column in df]
    data_col = sorted(data_col)

    if data_col != order:
        file.active = -1
        file.save()
        temp = 0
        # messages.error(request, "检查数据列名")
        return "检查数据列名"
    # 根据样本编号去重
    df.drop_duplicates(subset=['样本编码'], keep='first', inplace=True)
    # 进行字段排序

    df = df[order]
    # 修改列名 和数据库字段保持一致
    df.rename(columns={'母亲编号': 'mother_id', '城市': 'city', '母亲姓名': 'mother_name', 'NFAMILY': 'nfamily',
                       'NCHILD': 'nchild', '民族': 'minzu', '年龄': 'age', '年龄分类': 'age1',
                       '产次': 'chanci', '身高': 'tall', '孕前体重': 'weight1', '产前体重': 'weight2', '现体重': 'weight3',
                       '分娩方式': 'fenmian_way', '出生日期': 'birthday', '出生季度': 'birth_jd', '出生体重': 'birthweight',
                       '出生身长': 'birthtall', '性别': 'sex', '出生情况': 'birth_suit', '样本编码': 'yangben_id',
                       '采样天数': 'day', '泌乳期': 'miruqi', '采样次数': 'caiyang_time','α-乳白蛋白':'α_rubai',
                       'β-酪蛋白':'β_lao',}, inplace=True)
    df = df.dropna(subset=["age1","city","fenmian_way","chanci","birth_jd"])
    df = df.fillna(value=1)
    df.to_csv('new.csv',encoding='utf-8-sig')
    # 检查蛋白质数值类型
    float_col = ['α_rubai', 'β_lao']
    try:
        df[float_col] = df[float_col].apply(pd.to_numeric)
    except Exception as e:
        temp = 0
        # messages.error(request, str(e))
        return str(e)
    # 检查影响因素数值类型
    int_col = ['city', 'minzu', 'age1',  'chanci',  'fenmian_way', 'birth_jd']
    try:
        df[int_col] = df[int_col].apply(pd.to_numeric)
    except Exception as e:
        temp = 0
        # messages.error(request,str(e))
        return str(e)
    if temp:
        # 将本文件的数据保存到数据库中
        f = File.objects.filter(userid=qs)
        if len(f) > 0:
            f = f.first()
            f.save()
        file.active = 1
        file.save()
        data = Protain.objects.filter(insert_people=uid)
        data.delete()
        data = df
        for i in range(len(data)):
            ys = Protain.objects.create(insert_people=qs, mother_id=data.at[i, 'mother_id'], city=data.at[i, 'city'],
                                        mother_name=data.at[i, 'mother_name'],nfamily=data.at[i, 'nfamily'],nchild=data.at[i, 'nchild'],
                                        minzu=data.at[i, 'minzu'],age=data.at[i, 'age'], age1=data.at[i, 'age1'],
                                        chanci=data.at[i, 'chanci'],tall=data.at[i, 'tall'], weight1=data.at[i, 'weight1'],
                                        weight2=data.at[i, 'weight2'],weight3=data.at[i, 'weight3'], fenmian_way=data.at[i, 'fenmian_way'],
                                        birthday=data.at[i, 'birthday'],birth_jd=data.at[i, 'birth_jd'], birthweight=data.at[i, 'birthweight'],
                                        birthtall=data.at[i, 'birthtall'],sex=data.at[i, 'sex'],birth_suit=data.at[i, 'birth_suit'],
                                        miruqi=data.at[i, 'miruqi'],sample_id=data.at[i, 'yangben_id'], times=data.at[i, 'caiyang_time'],
                                        caiyang_day=data.at[i, 'day'],α_rubai=data.at[i, 'α_rubai'],β_lao=data.at[i, 'β_lao']
                                        )
            ys.save()
        # messages.success(request,'上传成功')
        return 1
def tosql(uid, uname, fid):
    """
    检查当前文件格式
        符合格式定义：当前使用数据库的文件状态改为0，本文件状态改为1，并上传至数据库
        不符合格式定义：本文件状态为-1
    """
    qs = User.objects.filter(userid=uid).first()
    file = File.objects.get(id=fid)
    filepath = File.objects.filter(id=fid).get().filepath
    filetype = filepath.split('.')[1]
    # 读取数据
    if filetype == 'csv':
        df = pd.read_csv(filepath, dtype=object)
    elif filetype == 'xlsx':
        df = pd.read_excel(filepath, sheet_name='母乳', engine='openpyxl', dtype=object)
        df_nf = pd.read_excel(filepath, sheet_name='奶粉', engine='openpyxl', dtype=object)
    else:
        df = pd.read_excel(filepath, sheet_name='母乳', engine='openpyxl', dtype=object)
        df_nf = pd.read_excel(filepath, sheet_name='奶粉', engine='openpyxl', dtype=object)
    # 检查数据列数
    temp = 1
    if df.shape[1] != 49 or df_nf.shape[1] != 26:
        file.active = -1
        file.save()
        temp = 0
        # messages.error(request, "数据列数不满足要求！请检查")
        return "数据列数不满足要求！请检查"
    # 检查列名
    order = ['母亲编号', '城市', '母亲姓名', 'NFAMILY', 'NCHILD', '民族', '年龄', '年龄分类',
             '产次', '身高', '孕前体重', '产前体重', '现体重', '分娩方式', '出生日期', '出生季度', '出生体重',
             '出生身长', '性别', '出生情况', '样本编号', '蛋白质', '采样天数', '泌乳期', '采样次数',
             '赖氨酸', '苏氨酸', '缬氨酸', '蛋氨酸', '异亮氨酸', '亮氨酸', '苯丙氨酸', '色氨酸', '组氨酸',
             '天冬氨酸', '丝氨酸', '谷氨酸', '甘氨酸', '丙氨酸', '酪氨酸', '精氨酸', '脯氨酸', '半胱氨酸', '总氨基酸', '必须氨基酸', '非必须氨基酸',
             '必须氨基酸/总氨基酸', '必须氨基酸/非必须氨基酸', '非必须氨基酸/总氨基酸']
    order_nf = ['奶粉名称', '总蛋白', '赖氨酸', '苏氨酸', '缬氨酸', '蛋氨酸', '异亮氨酸', '亮氨酸', '苯丙氨酸', '色氨酸', '组氨酸',
                '天冬氨酸', '丝氨酸', '谷氨酸', '甘氨酸', '丙氨酸', '酪氨酸', '精氨酸', '脯氨酸', '半胱氨酸', '总氨基酸', '必须氨基酸', '非必须氨基酸',
                '必须氨基酸/总氨基酸', '必须氨基酸/非必须氨基酸', '非必须氨基酸/总氨基酸']
    order = sorted(order)
    order_nf = sorted(order_nf)
    data_col = [column for column in df]
    nf_col = [column for column in df_nf]
    data_col = sorted(data_col)
    nf_col = sorted(nf_col)

    if data_col != order or nf_col != order_nf:
        file.active = -1
        file.save()
        temp = 0
        # messages.error(request, "检查数据列名")
        return  "检查数据列名"
    # 根据样本编号去重
    df.drop_duplicates(subset=['样本编号'], keep='first', inplace=True)
    # 进行字段排序

    df = df[order]
    df_nf = df_nf[order_nf]
    # 修改列名 和数据库字段保持一致
    df.rename(columns={'母亲编号': 'mother_id', '城市': 'city', '母亲姓名': 'mother_name', 'NFAMILY': 'nfamily',
                       'NCHILD': 'nchild', '民族': 'minzu', '年龄': 'age', '年龄分类': 'age1',
                       '产次': 'chanci', '身高': 'tall', '孕前体重': 'weight1', '产前体重': 'weight2', '现体重': 'weight3',
                       '分娩方式': 'fenmian_way', '出生日期': 'birthday', '出生季度': 'birth_jd', '出生体重': 'birthweight',
                       '出生身长': 'birthtall', '性别': 'sex', '出生情况': 'birth_suit', '样本编号': 'yangben_id',
                       '蛋白质': 'pro', '采样天数': 'day', '泌乳期': 'miruqi', '采样次数': 'caiyang_time',
                       '赖氨酸': 'lai', '苏氨酸': 'su', '缬氨酸': 'jie', '蛋氨酸': 'dan', '异亮氨酸': 'yiliang',
                       '亮氨酸': 'liang', '苯丙氨酸': 'benbing', '色氨酸': 'se', '组氨酸': 'zu',
                       '天冬氨酸': 'tiandong', '丝氨酸': 'si', '谷氨酸': 'gu', '甘氨酸': 'gan', '丙氨酸': 'bing',
                       '酪氨酸': 'lao', '精氨酸': 'jing', '脯氨酸': 'fu', '半胱氨酸': 'banguang', '总氨基酸': 'sumaa', '必须氨基酸': 'baa',
                       '非必须氨基酸': 'fbaa',
                       '必须氨基酸/总氨基酸': 'b2sum', '必须氨基酸/非必须氨基酸': 'b2fb', '非必须氨基酸/总氨基酸': 'fb2sum', }, inplace=True)
    df_nf.rename(columns={'奶粉名称': 'nf_name', '总蛋白': 'pro',
                          '赖氨酸': 'lai', '苏氨酸': 'su', '缬氨酸': 'jie', '蛋氨酸': 'dan', '异亮氨酸': 'yiliang',
                          '亮氨酸': 'liang', '苯丙氨酸': 'benbing', '色氨酸': 'se', '组氨酸': 'zu',
                          '天冬氨酸': 'tiandong', '丝氨酸': 'si', '谷氨酸': 'gu', '甘氨酸': 'gan', '丙氨酸': 'bing',
                          '酪氨酸': 'lao', '精氨酸': 'jing', '脯氨酸': 'fu', '半胱氨酸': 'banguang', '总氨基酸': 'sumaa',
                          '必须氨基酸': 'baa', '非必须氨基酸': 'fbaa',
                          '必须氨基酸/总氨基酸': 'b2sum', '必须氨基酸/非必须氨基酸': 'b2fb', '非必须氨基酸/总氨基酸': 'fb2sum', }, inplace=True)

    # 检查氨基酸数值类型
    float_col = ['lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong', 'si', 'gu',
                 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang']
    try:
        df[float_col] = df[float_col].apply(pd.to_numeric)
        df_nf[float_col] = df_nf[float_col].apply(pd.to_numeric)
    except Exception as e:
        temp = 0
        # messages.error(request, str(e))
        return str(e)

    # 检查影响因素数值类型
    int_col = ['city', 'nfamily', 'nchild', 'minzu', 'age1', 'age', 'chanci', 'tall', 'fenmian_way', 'birth_jd',
               'birth_suit']
    try:
        df[int_col] = df[int_col].apply(pd.to_numeric)
    except Exception as e:
        temp = 0
        # messages.error(request, str(e))
        return str(e)

    if temp:
        # 将本文件的数据保存到数据库中
        f = File.objects.filter(userid=qs)
        if len(f) > 0:
            f = f.first()
            f.save()
        file.active = 1
        file.save()
        saveSql(df, 'mr', uid)
        saveSql(df_nf, 'nf', uid)
        # messages.success(request, '上传成功！')
        return 1

def yuanshi(request):

    uname = request.session.get('uname', default='勇敢牛牛')
    uid = request.session.get('uid', default=1)
    qs = User.objects.filter(userid=uid).first()
    ys_cr = Yuanshi.objects.filter(insertpeople=qs,miruqi=1)
    ys_gd = Yuanshi.objects.filter(insertpeople=qs, miruqi=2)
    ys_zq = Yuanshi.objects.filter(insertpeople=qs, miruqi=3)
    ys_wq = Yuanshi.objects.filter(insertpeople=qs, miruqi=4)
    ys_cr = read_frame(ys_cr)
    ys_gd = read_frame(ys_gd)
    ys_zq = read_frame(ys_zq)
    ys_wq = read_frame(ys_wq)
    data1 = ys_cr.describe().loc[:, ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                 'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                 'b2fb', 'b2sum', 'fb2sum')]
    data1 = data1.apply(lambda x: round(x, 2))

    data2 = ys_gd.describe().loc[:, ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                 'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                 'b2fb', 'b2sum', 'fb2sum')]
    data2 = data2.apply(lambda x: round(x, 2))
    data3 = ys_zq.describe().loc[:, ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                 'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                 'b2fb', 'b2sum', 'fb2sum')]
    data3 = data3.apply(lambda x: round(x, 2))
    data4 = ys_wq.describe().loc[:, ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                 'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                 'b2fb', 'b2sum', 'fb2sum')]
    data4 = data4.apply(lambda x: round(x, 2))
    if 'downdata' in request.GET:
        miruqi = int(request.GET.get('miruqi'))
        if miruqi == 1:
            data = data1
            name = '初乳描述性统计'
        elif miruqi == 2:
            data = data2
            name = '过渡乳描述性统计'
        elif miruqi == 3:
            data = data3
            name = '早期成熟乳描述性统计'
        else:
            data = data4
            name = '晚期成熟乳描述性统计'
        filename = uname + '-' + name + '.csv'
        data.to_csv('static/newfile/'+filename)
        f = open('static/newfile/'+filename, 'rb')
        response = FileResponse(f)
        response['Content-Type'] = 'application/octet-stream'
        filename = 'attachment; filename =' + filename
        response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
        return response


    return render(request, "table_basic.html", {'temp': 'mr', 'YS1': data1, 'YS2': data2,'YS3': data3, 'YS4': data4,'uname': uname})


def xiangdui(request):
    uname = request.session.get('uname', default='勇敢牛牛')
    uid = request.session.get('uid', default=1)
    qs = User.objects.filter(userid=uid).first()
    xd_cr = Xiangdui.objects.filter(insertpeople=qs, miruqi=1)
    xd_gd = Xiangdui.objects.filter(insertpeople=qs, miruqi=2)
    xd_zq = Xiangdui.objects.filter(insertpeople=qs, miruqi=3)
    xd_wq = Xiangdui.objects.filter(insertpeople=qs, miruqi=4)
    xd_cr = read_frame(xd_cr)
    xd_gd = read_frame(xd_gd)
    xd_zq = read_frame(xd_zq)
    xd_wq = read_frame(xd_wq)
    data1 = xd_cr.describe().loc[:, ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                     'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                     'b2fb')]
    data1 = data1.apply(lambda x: round(x, 2))

    data2 = xd_gd.describe().loc[:, ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                     'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                     'b2fb')]
    data2 = data2.apply(lambda x: round(x, 2))
    data3 = xd_zq.describe().loc[:, ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                     'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                     'b2fb')]
    data3 = data3.apply(lambda x: round(x, 2))
    data4 = xd_wq.describe().loc[:, ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                     'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                     'b2fb')]
    data4 = data4.apply(lambda x: round(x, 2))
    if 'downdata' in request.GET:
        miruqi = int(request.GET.get('miruqi'))
        if miruqi == 1:
            data = data1
            name = '初乳描述性统计'
        elif miruqi == 2:
            data = data2
            name = '过渡乳描述性统计'
        elif miruqi == 3:
            data = data3
            name = '早期成熟乳描述性统计'
        else:
            data = data4
            name = '晚期成熟乳描述性统计'
        filename = uname + '-相对数据-' + name + '.csv'
        data.to_csv('static/newfile/' + filename)
        f = open('static/newfile/' + filename, 'rb')
        response = FileResponse(f)
        response['Content-Type'] = 'application/octet-stream'
        filename = 'attachment; filename =' + filename
        response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
        return response

    return render(request, "table_complete.html", {'temp': 'mr', 'XD1': data1, 'XD2': data2,'XD3': data3, 'XD4': data4,'uname': uname})


def nfYuanshi(request):
    if request.method == "GET":
        uname = request.session.get('uname', default='勇敢牛牛')
        uid = request.session.get('uid', default=1)
        qs = User.objects.filter(userid=uid).first()
        ys = NfYuanshi.objects.filter(insertpeople=qs)
        ys = read_frame(ys)
        data = ys.describe().loc[:, ('pro','lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                                     'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'sumaa', 'baa', 'fbaa',
                                     'b2fb', 'b2sum', 'fb2sum')]
        data = data.apply(lambda x: round(x, 2))
        if 'downdata' in request.GET:
            filename = uname + '-' + '配方粉原始数据描述性统计.csv'
            data.to_csv('static/newfile/' + filename)
            f = open('static/newfile/' + filename, 'rb')
            response = FileResponse(f)
            response['Content-Type'] = 'application/octet-stream'
            filename = 'attachment; filename =' + filename
            response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
            return response

    return render(request, "table_basic.html", {'temp': 'nf', 'YS': data, 'uname': uname})


def nfXiangdui(request):
    if request.method == "GET":
        uname = request.session.get('uname', default='勇敢牛牛')
        uid = request.session.get('uid', default=1)
        qs = User.objects.filter(userid=uid).first()
        xd = NfXiangdui.objects.filter(insertpeople=qs)
        xd = read_frame(xd)
        data = xd.describe().loc[:,
               ('lai', 'su', 'jie', 'dan', 'yiliang', 'liang', 'benbing', 'se', 'zu', 'tiandong',
                'si', 'gu', 'gan', 'bing', 'lao', 'jing', 'fu', 'banguang', 'baa', 'fbaa',
                'b2fb')]
        data = data.apply(lambda x: round(x, 2))
        if 'downdata' in request.GET:
            filename = uname + '-' + '配方粉相对数据描述性统计.csv'
            data.to_csv('static/newfile/' + filename)
            f = open('static/newfile/' + filename, 'rb')
            response = FileResponse(f)
            response['Content-Type'] = 'application/octet-stream'
            filename = 'attachment; filename =' + filename
            response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
            return response
    return render(request, "table_complete.html", {'temp': 'nf', 'XD': data, 'uname': uname})


def login(request):
    if request.method == "POST":
        uname = request.POST.get("uname")
        pwd = request.POST.get("pwd")
        user = User.objects.filter(username=uname)
        if len(user) == 0:
            messages.error(request, '用户名不存在')
            return render(request, "login.html")
        else:
            password = User.objects.get(username=uname).passward
            if password == pwd:
                uid = User.objects.get(username=uname).userid
                uname = User.objects.get(username=uname).username
                request.session['uid'] = uid
                request.session['uname'] = uname
                file_list = File.objects.filter(userid=uid)
                return render(request, "index.html", {"file_list": file_list, "uname": uname})
            else:
                messages.error(request, '密码错误')
                return render(request, "login.html")
    else:
        return render(request, "login.html")


def volin(request):
    return render(request, "chart_line.html")


def register(request):
    if request.method == "GET":
        return render(request, "register.html")
    else:
        uname = request.POST.get("uname")
        email = request.POST.get("email")
        pwd = request.POST.get("pwd")
        pwd1 = request.POST.get("pwd1")
        user = User.objects.filter(username=uname)
        if len(user) > 0:
            messages.error(request, '用户名已经存在')
        user = User.objects.filter(email=email)
        if len(user) > 0:
            messages.error(request, '邮箱已经被注册')
        if pwd != pwd1:
            messages.error(request, '两次输入密码不符')
        user = User.objects.create(username=uname, email=email, passward=pwd)
        user.save()
        id = User.objects.get(username=uname).userid
        file_list = File.objects.filter(userid=id)
        request.session['uid'] = id
        request.session['uname'] = uname
        return render(request, "index.html", {"file_list": file_list, "uname": uname})

def F_cal1(mr,nf):
    '''
    所有母乳比奶粉  矩阵的范数
    '''
    nf_name = []
    res = []
    for i in range(len(nf)):
        nf_name.append(nf['nf_name'][i])
    mr = mr.values
    nf = nf.values
    for i in range(len(nf)):
        bizhi = []
        for j in range(len(mr)):
            temp = []
            for k in range(1,len(nf[i])):
                temp.append(mr[j][k] / nf[i][k])
            bizhi.append(temp)

        res.append(np.linalg.norm(np.array(bizhi)))
    for i in range(len(res)):
        res[i] = round(res[i],4)
    dic_res = dict(map(lambda x, y: [x, y], nf_name, res))
    dic_res = sorted(dic_res.items(), key=lambda item: item[1])
    return dic_res


def F_cal2(mr,nf):
    '''
    每个奶粉比母乳的范数，范数的平均值
    '''
    nf_name = []
    res = []
    for i in range(len(nf)):
        nf_name.append(nf['nf_name'][i])
    mr = mr.values
    nf = nf.values
    for i in range(len(nf)):
        bizhi = []
        for j in range(len(mr)):
            temp = []
            for k in range(1,len(nf[i])):
                temp.append(mr[j][k] / nf[i][k])
            # 比值的范数
            bizhi.append(np.linalg.norm(np.array(temp)) )
        res.append(np.mean(bizhi))
    for i in range(len(res)):
        res[i] = round(res[i],4)
    dic_res = dict(map(lambda x, y: [x, y], nf_name, res))
    dic_res = sorted(dic_res.items(), key=lambda item: item[1])
    return dic_res





def fanshu(request):
    uid = request.session.get('uid', default='1')
    uname = request.session.get('uname', default='勇敢牛牛')
    page = request.GET.get('page')
    engine = pymysql.connect(host="10.2.172.49", user="root", passwd="123456", database="zhangduoduo_www")
    if request.method == 'POST':
        ajs1 = request.POST.getlist("check_box_list_ajs")
        request.session['ajs1'] = ajs1
        miruqi = request.POST.get('check_box_list_miruqi')
        request.session['miruqi'] = miruqi
        nfdc = request.POST.get('check_box_list_12')
        request.session['nfdc'] = nfdc
        datatype = request.POST.get('check_box_list_datatype')
        request.session['datatype'] = datatype
        if miruqi == '5':
            miruqi = '*'
        # 数据库查询语句
        sql_mr = 'select mother_id, '
        sql_nf = 'select nf_name, '
        for item in ajs1:
            sql_mr += item +','
            sql_nf += item +','
        sql_mr = sql_mr[:-1] + ' from '+ datatype + ' where miruqi=' + miruqi + ' and insertpeople='+uid

        if datatype=='yuanshi':
            data_1 = pd.read_sql(sql_mr,engine)
            if nfdc != 'all':
                sql_nf = sql_nf[:-1] + ' from nf_yuanshi where insertpeople ='+uid+ ' and nf_name like \'%'+nfdc+'%\''
            else:
                sql_nf = sql_nf[:-1] + ' from nf_yuanshi where insertpeople ='+uid
            data_2 = pd.read_sql(sql_nf,engine)
        else:
            data_1 = pd.read_sql(sql_mr,engine)
            if nfdc != 'all':
                sql_nf = sql_nf[:-1] + ' from nf_xiangdui where insertpeople ='+uid+ ' and nf_name like \'%'+nfdc+'%\''
            else:
                sql_nf = sql_nf[:-1] + ' from nf_xiangdui where insertpeople ='+uid
            data_2 = pd.read_sql(sql_nf,engine)

        # 删除重复妈妈
        data_1 = data_1.drop_duplicates(subset='mother_id')

        # 所有母乳比奶粉  矩阵的范数
        dic_res1 = F_cal1(data_1,data_2)
        # 每个母乳比奶粉 比值的范数，再范数平均
        dic_res2 = F_cal2(data_1,data_2)
        request.session['dic_res1'] = dic_res1
        request.session['dic_res2'] = dic_res2

        paginator1 = Paginator(dic_res1, 10)  # 每页显示10条
        paginator2 = Paginator(dic_res2,10)
        page = 1
        try:
            number1 = paginator1.page(page)
            number2 = paginator2.page(page)
        except PageNotAnInteger:
            number1 = paginator1.page(1)
            number2 = paginator2.page(1)
        except EmptyPage:
            number1 = paginator1.page(paginator1.num_pages)
            number2 = paginator2.page(paginator2.num_pages)
        if 'down' in request.POST:
            filename = '母乳比奶粉范数分析-' + uid + '.csv'
            f = open('static/newfile/' + filename, 'w', encoding='utf-8', newline="")
            csv_write = csv.writer(f)
            csv_write.writerow(['氨基酸选择'])
            csv_write.writerow(ajs1)
            csv_write.writerow(['泌乳期', miruqi])
            csv_write.writerow(['奶粉选择', nfdc])
            csv_write.writerow(['数据类型', datatype])
            csv_write.writerow([''])
            csv_write.writerow(
                ['奶粉名称', 'F范数(矩阵)', ' ', '奶粉名称', 'F范数(平均)'])
            for i in range(len(dic_res1)):
                csv_write.writerow([dic_res1[i][0], dic_res1[i][1], ' ', dic_res2[i][0], dic_res2[i][1]])
            f.close()
            f = open('static/newfile/' + filename, 'rb')
            response = FileResponse(f)
            response['Content-Type'] = 'application/octet-stream'
            filename = 'attachment; filename =' + filename
            response['Content-Disposition'] = filename.encode('utf-8', 'ISO-8859-1')
            return response
        return render(request, "fanshu.html", {"page":number1,"page2":number2})
    else:
        if page is not None:
            dic_res1 = request.session.get('dic_res1')
            dic_res2 = request.session.get('dic_res2')
            paginator1 = Paginator(dic_res1, 10)  # 每页显示10条
            paginator2 = Paginator(dic_res2, 10)
            page = request.GET.get('page')
            try:
                number1 = paginator1.page(page)
                number2 = paginator2.page(page)
            except PageNotAnInteger:
                number1 = paginator1.page(1)
                number2 = paginator2.page(1)
            except EmptyPage:
                number1 = paginator1.page(paginator1.num_pages)
                number2 = paginator2.page(paginator2.num_pages)
            return render(request, "fanshu.html", {"page":number1,"page2":number2})
        else:
            return render(request,"fanshu.html")

# write by zdd

# from APP.models import File, User, Yuanshi,Xiangdui,NfYuanshi,NfXiangdui,Tempyaunshi,Tempxiangdui
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
matplotlib.rc("font",family='YouYuan')
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

    ma,delmamaid = fundelmamaid(ajs,mr,mama)
    all_anjisuan = delmamasql(ajs,mr,delmamaid,mama,plot)
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
        all_naifen, all_anjisuan,ma = chartssql(ajs, miruqi ,mama,True,uid,plot)
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
            all_anjisuan = Tempxiangdui.objects.values_list(*ajs1)
        else:
            all_naifen = NfYuanshi.objects.values_list(*ajs1)
            all_anjisuan = Tempyaunshi.objects.values_list(*ajs1)
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





