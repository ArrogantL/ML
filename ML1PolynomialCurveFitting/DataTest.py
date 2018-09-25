import math

from AnalyticalSolution import analyticalSolve
from ConjugateGradient import conjugateGradient
from DataGenerator import generateData
from GradientDescent import RSS, gradientDescent
from Visualization import visualPoly


def solutionTest(func,ns, dataNum, lnLambdas=[],SavePath=""):
    list=[]
    for i in ns:
        listi=[]
        list.append(listi)
        for lx in dataNum:
            listx=[]
            listi.append(listx)
            X, T = generateData(lx)
            testX,testT=generateData(lx//2)
            if len(lnLambdas)==0:
                W = func(i, X, T)
                print("%d;%d;None;%r" % (i, lx, RSS(testT, testX, W) / len(testX)))
                listx.append(W)
            else:
                for lnLambda in lnLambdas:

                    W = analyticalSolve(i, X, T,lnLambda)
                    listx.append(W)
                    print("%d;%d;%.0f;%r" % (i, lx, lnLambda, RSS(testT, testX, W) / len(testX)))
    for i in range(len(ns)):
        for j in range(len(dataNum)):
            rW = []
            labels=[]
            for k in range(len(lnLambdas)):

                rW.append(list[i][j][k])
                labels.append("λ e^%.0f"%(lnLambdas[k]))
            visualPoly(*rW, *labels, title="%s poly%d datanum%d"%(func.__name__,ns[i],dataNum[j]), savePath=SavePath)

if __name__ == '__main__':
    solutionTest(gradientDescent,[5,7,9,15],[5,10,20],[-99999999,-18,0],SavePath="DataGif/GradientDescent/")
