import matplotlib

import matplotlib.pyplot as plt
import numpy
from numpy import polyval, math, sin


def visualPoly(*WsandLabels,X=None,T=None,title="Data Graph",savePath="None",isShow=False):
    """
    :param WsandLabels: 按次数由低到高排列，先W，全部W输入后接入按顺序接入labels
    :return:
    """
    t=len(WsandLabels)//2
    assert t*2==len(WsandLabels)
    cmap = plt.get_cmap('viridis')
    colors = cmap(numpy.linspace(0, 1, t))
    x = numpy.arange(0, 1, 0.001)
    for i,color in zip(range(t),colors):
        W=WsandLabels[i]
        label=WsandLabels[i+t]
        s = list(W)
        s.reverse()
        y = polyval(s, x)
        plt.plot(x, y, "--",c=color, linewidth=1,label=label)
    plt.plot(x, sin(2*math.pi*x), "r--", linewidth=1, label="sin2πx")
    if len(X)!=0 and len(T)!=0:
        plt.plot(X,T,"b*",linewidth=10,label="Sample")
    plt.title(title)  # 添加图形标题

    plt.legend()  # 展示图例
    if savePath !="None":
        plt.savefig(savePath+title+".png")
    if isShow:
        plt.show()
    plt.clf()


def visualResultAndSampleAndTarget(W,X,T):
    x = numpy.arange(0, 1, 0.001)
    s = list(W)
    s.reverse()
    y = polyval(s, x)
    plt.plot(x, y, 'r--', linewidth=1,label="W")
    plt.plot(X, T, 'g*', linewidth=10,label="Sample")
    plt.plot(x, [math.sin(2 * math.pi * x) for x in x], 'b--', linewidth=1,label="sin2πx")
    plt.legend()
    plt.show()

if __name__ == '__main__':
  pass