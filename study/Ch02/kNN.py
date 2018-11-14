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

# --------------------DEMO1------------------------------------------------------------
def classify0(inX, dataSet, labels, k):
    # dataSet=[[1,2],[2,2],[3,3]] dataSetSize= 4
    # shape：查看数组或矩阵的维数
    print(dataSet.shape)
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
        # classCount.get('B',0)从对象中获取'B'属性的value，如果没有获取到，给默认值0
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        print(classCount[voteIlabel])


    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# # 执行DEMO1
group,labels=createDataSet()
classify0([0,0],group,labels,3)


# --------------------DEMO2------------------------------------------------------------
# 读取文件数据
def file2matrix(filename):  # 从文件中读入训练数据，并存储为矩阵
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)        # 获取 n=样本的行数
    returnMat = zeros((numberOfLines,3))    # 创建一个2维矩阵用于存放训练样本数据，一共有n行，每一行存放3个数据
    classLabelVector = []                   # 创建一个1维数组用于存放训练样本标签。
    index = 0
    for line in arrayOLines :
        line = line.strip()  # 把回车符号给去掉
        listFromLine = line.split('\t')  # 把每一行数据用\t分割
        returnMat[index, :] = listFromLine[0:3]  # 把分割好的数据放至数据集，其中index是该样本数据的下标，就是放到第几行
        labels = {'didntLike':1,'smallDoses':2,'largeDoses':3}  # 新增
        classLabelVector.append(labels[listFromLine[-1]])  # 去掉了int
        # 把该样本对应的标签放至标签集，顺序与样本集对应。 python语言中可以使用-1表示列表中的最后一列元素

        index += 1
    return returnMat,classLabelVector

# 读取文件内容
datingDataMat,datingLabels=file2matrix('datingTestSet.txt')

# 数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges,minVals

normMat ,ranges,minVals = autoNorm(datingDataMat)

# 测试代码
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if(classifierResult!=datingLabels[i]):errorCount += 1.0
    print ("the total error rate is: %f"%(errorCount/float(numTestVecs)))

# datingClassTest()

# # 绘图
# import matplotlib
# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax =fig.add_subplot(111)
# ax.scatter(normMat[:,1],normMat[:,0],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()