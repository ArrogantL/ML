import numpy as np
from numpy import argsort


def test1():
    a = np.arange(12).reshape(3, 4)
    print(a.dtype.name)
    b = np.array([0, 1, 2], dtype=complex)
    c = np.zeros((3, 4))
    d = np.ones((3, 4))
    e = np.arange(1, 15, 0.7)
    # arange并不能准确的控制间隔，如果想要均匀分布的点使用line-space函数
    f = np.linspace(1, 15, 16)


def test2():
    print(np.arange(120).reshape(2,3,4,5))
def test3():
    a=np.arange(5)
    print(a-np.arange(0,10,2))
    print(a**2)
    print(10*np.sin(a))
    print(a<3)
def test4():
    a=np.array([(1,0),(2,3)])
    b=np.array([(2,2),(2,2)])
    print(a*b)
    # np中二维array矩阵乘法
    print(a@b)
    print(np.dot(a,b))
def test5():
    a=np.random.random_sample((3,4))
    print(a)
    print(a.sum(),a.mean(),a.max(),a.min())
    # 通过设定axis为0 1 None 能够获得最小行、列、单数据
    print(a.max(axis=None),a.max(axis=0))
    # 递增数组，按行、列、单元素
    print(a.cumsum(axis=0))
def test6():
    # 通用函数如sin exp cos 均在 ufunc中，可以在array上产生单元素处理
    # 的array
    a=np.linspace(0,15,3)
    print(np.sqrt(a))
    b=np.arange(3)
    print(np.add(a,b))
def test7():
    # 索引、切片、迭代
    # numpy中一维array可以相当与list一样
    a=np.arange(10)
    a[:6:2] = -1000
    print(a)
    # 多维array，每个轴一个索引，用逗号隔开
    b=np.fromfunction(lambda x,y:10*x+y,(5,4),dtype=int)
    print(b)
    # 对单元素的索引事实上是两个轴索引的集成
    print(b[2,3])
    print(b[:5,1])

    # 当某个轴的索引省略，默认为全选。
    # 当切片是一维时，自动降为一维数组,且没有方向之分(列自动变成行),对于非单行列的均保持方向
    print(b[-1])
    print(b[1, :4])
    print(b[1:4,1:3])
    # 两个连续的范围会取交集
    print(b[0:3][:,3:4])
    # 迭代
    for e in b.flat:
        print(e)
        e=0
    print(b)

def test8():
    # 改变数组形状
    a=np.arange(12).reshape(3,4)
    print(a)
    # reshape，不改变数组本身，返回新的array
    # resize改变数组本身
    a.resize(2,6)
    print(a)
    # 将某个维度设为-1会自动平衡
    a.reshape(4, -1)
    # 看到未改变！shape并不改变array
    print(a)

    a=a.reshape(4,-1)
    print(a)

def sortsort():
    # 对数组进行排序，选取特定的内容
    # 第一种是对一个多维array，以某个维度排序某个维度
    # 第二种是两个单独的一维array，以一个对另外一个排序
    vals, vects = np.arange(5),np.arange(5,0)
    # 返回以numpy数组（对list不支持）值为小到大排序的索引array
    indexs = argsort(vals)
    indexs = indexs[1:5]
    topdvects = vects[:, indexs]
    pass

if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    test8()