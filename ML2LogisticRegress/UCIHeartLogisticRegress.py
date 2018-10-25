import pandas as pd

from LogisticRegress import *


def processUCIheart():
    data = pd.read_csv("data/motified_data.csv")
    Y = data["11"].values.tolist()
    data = data.drop("11", axis=1)
    XX = data.values.tolist()
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.25, random_state=0)  # 随机选择25%作为测试集，剩余作为训练集
    tmp=pd.DataFrame(X_train)
    tmp['y']=y_train
    print("punishment\taccuracy_in_Train\taccuracy_in_Test")

    for punishment in range(20):
        punishment=punishment/10
        W, loss = newtonLogisticRegress(X_train, y_train, punishment=punishment)
        print("%.1f\t%f\t%f" %(punishment,accuracy(W, X_train, y_train),accuracy(W, X_test, y_test)) )



def preprocessData():
    data = pd.read_csv("data/UCI_heart.csv")
    data = data.replace('?', np.nan)
    for i in range(11):
        i += 1
        age_mode = data[str(i)].dropna().mode()[0]
        data[str(i)] = data[str(i)].fillna(age_mode)

    data.to_csv("data/motified_data.csv")


if __name__ == '__main__':
    processUCIheart()
