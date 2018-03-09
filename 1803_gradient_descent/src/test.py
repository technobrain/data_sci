# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# http://aima.cs.berkeley.edu/data/iris.csv
names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
iris = pd.read_csv('./data/iris.data', header=None, names=names)

np.random.seed(1)
# 描画領域のサイズ
FIGSIZE = (5, 3.5)

# 説明変数は 2つ
columns = ['Petal.Width', 'Petal.Length']

def plot_x_by_y(x, y, colors, ax=None):
    if ax is None:
        # 描画領域を作成
        fig = plt.figure(figsize=FIGSIZE)

        # 描画領域に Axes を追加、マージン調整
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(bottom=0.15)

    x1 = x.columns[0]
    x2 = x.columns[1]
    for (species, group), c in zip(x.groupby(y), colors):
        ax = group.plot(kind='scatter', x=x1, y=x2,
                        color=c, ax=ax, figsize=FIGSIZE)
    return ax

data = iris

x = data[columns]
y = data['Species']

plot_x_by_y(x, y, colors=['red', 'blue', 'green'])
plt.show()

# ラベルを 0, 1の列 * 3 に変換
y = pd.DataFrame({'setosa': (y == 'setosa').astype(int),
                  'versicolor': (y == 'versicolor').astype(int),
                  'virginica': (y == 'virginica').astype(int)
                  })

# index と同じ長さの配列を作成し、ランダムにシャッフル
indexer = np.arange(x.shape[0])
np.random.shuffle(indexer)

# x, y を シャッフルされた index の順序に並び替え
x = x.iloc[indexer, ]
y = y.iloc[indexer, ]

def p_y_given_x(x, w, b):

    def softmax(x):
        return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T

    return softmax(np.dot(x, w.T) + b)

def grad(x, y, w, b):
    error = p_y_given_x(x, w, b) - y
    w_grad = np.zeros_like(w)
    b_grad = np.zeros_like(b)

    for j in range(w.shape[0]):
        w_grad[j] = (error.iloc[:, j] * x.T).mean(axis=1)
        b_grad[j] = error.iloc[:, j].mean()

    return w_grad, b_grad, np.mean(np.abs(error), axis=0)


def msgd2(x, y, w, b, eta=0.1, num=700, batch_size=10):
    for i in range(num):
        for index in range(0, x.shape[0], batch_size):
            # index 番目のバッチを切り出して処理
            _x = x[index:index + batch_size]
            _y = y[index:index + batch_size]
            w_grad, b_grad, error = grad(_x, _y, w, b)
            w -= eta * w_grad
            b -= eta * b_grad
            e = np.sum(np.mean(np.abs(y - p_y_given_x(x, w, b))))
        if i % 5 == 0:
            yield i, w, b, e

def plot_logreg2(x, y, orig_x, orig_y, fitter, title):
    # 描画領域を作成
    fig = plt.figure(figsize=FIGSIZE)

    # 描画領域に Axes を追加、マージン調整
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=0.15)

    # 描画オブジェクト保存用
    objs = []

    # 回帰直線描画用の x 座標
    bx = np.arange(x.iloc[:, 0].min(), x.iloc[:, 0].max(), 0.1)

    # w, b の初期値を作成
    w = np.zeros((y.shape[1], x.shape[1]))
    b = np.zeros(y.shape[1])
    # 勾配法の関数からジェネレータを生成
    gen = fitter(x, y, w, b)
    # ジェネレータを実行し、勾配法 1ステップごとの結果を得る
    for i, w, b, e in gen:
        # 回帰直線の y 座標を計算
        import pdb; pdb.set_trace()
        by0 = -(w[0, 0] - w[1, 0]) / (w[0, 1] - w[1, 1]) * bx - (b[0] - b[1]) / (w[0, 1] - w[1, 1])
        by1 = -(w[1, 0] - w[2, 0]) / (w[1, 1] - w[2, 1]) * bx - (b[1] - b[2]) / (w[1, 1] - w[2, 1])
        # 回帰直線を生成
        l0 = ax.plot(bx, by0, color='gray', linestyle='dashed')
        l1 = ax.plot(bx, by1, color='gray', linestyle='dashed')
        # 描画するテキストを生成

        wt = """Iteration = {0} times
      |{1[0][0]:.2f}, {1[0][1]:.2f}|
w = |{1[1][0]:.2f}, {1[1][1]:.2f}|
      |{1[2][0]:.2f}, {1[2][1]:.2f}|
b = [{2[0]:.2f}, {2[1]:.2f}, {2[2]:.2f}]
error = {3:.3}""".format(i, w, b, e)

        # axes 上の相対座標 (0.1, 0.9) に text の上部を合わせて描画
        t = ax.text(0.1, 0.9, wt, va='top', transform=ax.transAxes)
        # 描画した line, text をアニメーション用の配列に入れる
        objs.append(tuple(l0) + tuple(l1) + (t, ))

    # データ, 表題を描画
    ax = plot_x_by_y(orig_x, orig_y, colors=['red', 'blue', 'green'], ax=ax)
    ax.set_title(title)
    ax.set_ylim(-0.5, 10)
    # アニメーション開始
    ani = animation.ArtistAnimation(fig, objs, interval=1, repeat=False)
    plt.show()
    # ファイル保存する場合は plt.show をコメントアウトして以下を使う
    # ani.save(title + '.gif', fps=15)

plot_logreg2(x, y, data[columns], data['Species'], msgd2, 'Minibatch SGD')
