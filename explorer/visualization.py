# Copyright (c) 2018-2019 Katori lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: サーチツール

import os
import subprocess
import sys
import numpy as np
import pandas as pd
import itertools
import datetime
import time
from pprint import pprint
#import matplotlib as mpl
#mpl.use('Agg')# リモート・サーバ上で図を出力するための設定
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from . import common

### analyze
def analyze(df0,key,y):
    #df0= pd.read_csv(file_csv,sep=',',names=columns)
    df = df0.groupby(key).aggregate(['mean','std','min','max'])
    x1 = df0.groupby(key)[key].mean().values
    y1mean = df[y,'mean'].values
    y1std  = df[y,'std'].values
    y1min  = df[y,'min'].values
    y1max  = df[y,'max'].values
    return x1,y1mean,y1std,y1min,y1max

def analyze2d(df0,key,X1,X2,Y):
    #df0= pd.read_csv(file_csv,sep=',',names=columns)
    df = df0.groupby(key).aggregate(['mean','std','min','max'])
    #x1 = df0.groupby(key)[x1].mean().values
    #x2 = df0.groupby(key)[x2].mean().values
    y1mean = df[Y,'mean'].values
    y1std  = df[Y,'std'].values
    y1min  = df[Y,'min'].values
    y1max  = df[Y,'max'].values
    x1 = df[X1,'mean'].values
    x2 = df[X2,'mean'].values
    return x1,x2,y1mean,y1std,y1min,y1max

### plot
def vline(X1,color="r",linestyle="dotted"):
    """
    common.configのX1属性の値をx軸の値とする縦線を引く
    """
    x1 = getattr(common.config,X1)
    plt.axvline(x=x1,color=color,linestyle=linestyle)

def savefig(fig=None):
    plt_output(fig)
    
def plt_output(fig=None):
    """
    画像を保存する。plt.savefig()でファイルを保存し、それをレポートに書き込む。
    fig: 出力するpngファイルの名前（デフォルト値：None,自動で名前付け）
    """
    #print("fig:",fig)
    #print("dir_path:",common.dir_path)
    if fig == None:
        fig=os.path.splitext(common.last_csv)[0] + '.png'

    fig_path = os.path.join(common.dir_path, fig)
    #print(fig_path,"   ",fig)
    plt.savefig(fig_path)
    common.report_figure(fig)

def plot_stat_summary(df, X, Y, label=None, color=None, width=1, linestyle='-', alpha_fill=.2, linestyle_minmax=':'):
    """
    エラーバーをつけてグラフを描画、平均、標準偏差、最大値、最小値をプロットする。
    変数(X)について同じ値を持つデータを収集し、変数Yについての基本的な統計量をもとめる。
    df: dataframe
    X: keyになる変数（column名）、グラフの横軸になる
    Y: 統計をとる対象の変数、グラフの縦軸になる
    linestyle: プロットの線のスタイル
    alpha_fill: 塗りつぶしの透明度
    linestyle_minmax: 最小値と最大値のプロットの線のスタイル
    """
    x, ymean, ystd, ymin, ymax = analyze(df, X, Y)
    plt.plot(x, ymean, color=color, linestyle=linestyle, linewidth=width, label=label)
    plt.fill_between(x, ymean-ystd, ymean+ystd, color=color, alpha=alpha_fill)
    plt.plot(x, ymin, color=color, linestyle=linestyle_minmax, linewidth=1)
    plt.plot(x, ymax, color=color, linestyle=linestyle_minmax, linewidth=1)


def plot1d(X1,Y1,fig=None,csv=None):
    """
    1次元プロットを作成する。
    X1: x軸の列名(columnsの中に含まれる名前)
    Y1: y軸の列名(columnsの中に含まれる名前)
    fig: 出力するpngファイルの名前（デフォルト値：None,自動で名前付け）
    csv: 入力するCSVファイルの名前（デフォルト値：None,直前に保存したファイルの名前）
    """
    if fig==None:
        #fig=common.name_file(common.prefix+"_scan1d_%s.png" % (X1))
        fig=os.path.splitext(common.last_csv)[0] + '.png'

    df = common.load_dataframe(csv)
    plt.figure()
    plt.plot(df[X1],df[Y1],'o')
    plt.ylabel(Y1)
    plt.xlabel(X1)
    #plt.show()
    plt_output(fig)

def plot1ds(X1,Y1,fig=None,csv=None):
    """
    1次元プロットを作成する。異なるseedのデータについては平均を標準偏差をとる
    X1: x軸の列名(columnsの中に含まれる名前)
    Y1: y軸の列名(columnsの中に含まれる名前)
    fig: 出力するpngファイルの名前（デフォルト値：None,自動で名前付け）
    csv: 入力するCSVファイルの名前（デフォルト値：None,直前に保存したファイルの名前）
    """
    if fig==None:
        #fig=common.name_file(common.prefix+"_scan1ds_%s.png" % (X1))
        fig=os.path.splitext(common.last_csv)[0] + '.png'

    df = common.load_dataframe(csv)
    x,ymean,ystd,ymin,ymax = analyze(df,X1,Y1)
    plt.figure()
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel(Y1)
    plt.xlabel(X1)
    plt_output(fig)

def plot2d(X1,X2,Y1,fig=None,csv=None):
    if fig==None:
        #fig=common.name_file(common.prefix+"_scan2d_%s_%s.png" % (X1,X2))
        fig=os.path.splitext(common.last_csv)[0] + '.png'

    df = common.load_dataframe(csv)
    df = df.sort_values('id',ascending=True)
    x1 = df[X1].values
    x2 = df[X2].values
    y1 = df[Y1].values
    nx1 = len(set(x1))
    nx2 = len(set(x2))

    plt.figure()
    for cx2 in set(x2):
        df2 = df[(df[X2]==cx2)]
        x1 = df2[X1].values
        y1 = df2[Y1].values
        plt.plot(x1,y1)

    plt.xlabel(X1)
    plt.ylabel(Y1)
    #plt.show()
    plt_output(fig)

def plot2ds(X1,X2,Y1,fig=None,csv=None):
    if fig==None:
        #fig=common.name_file(common.prefix+"_scan2ds_%s_%s.png" % (X1,X2))
        fig=os.path.splitext(common.last_csv)[0] + '.png'

    df = common.load_dataframe(csv)
    df = df.sort_values('id',ascending=True)
    x1 = df[X1].values
    x2 = df[X2].values
    y1 = df[Y1].values
    nx1 = len(set(x1))
    nx2 = len(set(x2))

    plt.figure()
    for cx2 in set(x2):
        df2 = df[(df[X2]==cx2)]
        x,ymean,ystd,ymin,ymax = analyze(df2,X1,Y1)
        plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)

    plt.xlabel(X1)
    plt.ylabel(Y1)
    #plt.show()
    plt_output(fig)

def plot2d_pcolor(X1,X2,Y1,fig=None,csv=None):
    if fig==None:
        #fig=common.name_file(common.prefix+"_scan2d_%s_%s_pcolor.png" % (X1,X2))
        fig=os.path.splitext(common.last_csv)[0] + '_pcolor.png'

    df = common.load_dataframe(csv)
    df = df.sort_values('id',ascending=True)
    x1 = df[X1].values # XXX 正しい値を返さない。？？？
    print(df[X1])
    print("x1:\n",x1)
    print(set(x1))
    x2 = df[X2].values
    y1 = df[Y1].values
    nx1 = len(set(x1)) # XXX このsetがNG?
    nx2 = len(set(x2))
    x1 = x1.reshape(nx1,nx2)
    x2 = x2.reshape(nx1,nx2)
    y1 = y1.reshape(nx1,nx2)

    #print(nx1,nx2)
    #print(x2)
    plt.figure()
    plt.pcolor(x1,x2,y1)
    plt.colorbar()
    plt.xlabel(X1)
    plt.ylabel(X2)
    plt.title(Y1)
    #plt.show()
    plt_output(fig)

def plot2ds_pcolor(X1,X2,Y1,fig=None,csv=None):
    if fig==None:
        #fig=common.name_file(common.prefix+"_scan2ds_%s_%s_pcolor.png" % (X1,X2))
        fig=os.path.splitext(common.last_csv)[0] + '_pcolor.png'

    df = common.load_dataframe(csv)
    df = df.sort_values('id',ascending=True)
    x1,x2,ymean,ystd,ymin,ymax = analyze2d(df,"id",X1,X2,Y1)
    nx1=len(set(x1))
    nx2=len(set(x2))
    x1=x1.reshape(nx1,nx2)
    x2=x2.reshape(nx1,nx2)
    ymean=ymean.reshape(nx1,nx2)

    plt.figure()
    plt.pcolor(x1,x2,ymean)
    plt.colorbar()
    plt.xlabel(X1)
    plt.ylabel(X2)
    plt.title(Y1)
    #plt.show()
    plt_output(fig)
