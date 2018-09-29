'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    # dataSet=[[1,2],[2,2],[3,3]] dataSetSize= 4
    # shape：查看数组或矩阵的维数
    dataSetSize = dataSet.shape[0]
    # tile([0,0],(2,1))在列方向重复1次，行2次，
    #array([0,0],
    #      [0,0])
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    # numpy的sum函数，axis=1行方向相加，axis=0列方向相加
    # sum([[1,2],[0,2]],axis=0)=array([1,4])
    # sum([[1,2],[0,2]],axis=1)=array([3,2])
    sqDistances = sqDiffMat.sum(axis=1)
    # **0.5求根
    distances = sqDistances**0.5
    # argsort返回的是数组值从小到大的索引值。
    # argsort([2,1.1,1,0]).argsort()=[3,2,1,0] 索引为3最小，然后依次
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        print('---------')
        print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        print(classCount[voteIlabel])
        print('222222222222222')
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','B','C','D']
    return group, labels

group,labels=createDataSet()
classify0([0,0],group,labels,3)

def test():
    labels=['A','B','C','D']
    sortedDistIndicies=array([2,3,1,0])
    classCount={}
    k=3
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    print('111111')
    print(classCount)
    print('22222')
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print('33333333')
    print(sortedClassCount)
    print('44444444444')
    return sortedClassCount[0][0]