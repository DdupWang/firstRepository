# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
import numpy as np


def loadAlltrain(trainAllDataPath):
    resutl={}
    with open(file=trainAllDataPath,mode='r',encoding='utf-8') as f:
        for str in f.readlines():
            strArr=str.split("\\|")
            eci=strArr[0]
            inputStr=strArr[1]
            outputStr=strArr[2]
            inputdim=int(inputStr.split(":")[0])+1
            outputdim=2

            inputone = np.zeros(shape=(inputdim), dtype=float)
            outputone = np.zeros(shape=(outputdim), dtype=float)

            for inputx in inputStr.split(";"):
                if (len(inputx) > 0):
                    inputValue = inputx.split(":");
                    inputone[int(inputValue[0])] = float(inputValue[1])
            outputone[0] = float(outputStr.split(";")[0])
            outputone[1] = float(outputStr.split(";")[1])

            if (eci in resutl):
                resutl.get(eci)[1].append(inputone)
                resutl.get(eci)[2].append(outputone)
            else:
                resutl[eci]=[inputdim,[inputone],[outputone]]
    return resutl


def load_train(train_path,inputdim,outputdim):
    input=[]
    output=[]
    with open(file=train_path,mode='r',encoding='utf-8') as f:
        for str in f.readlines():
            inputone = np.zeros(shape=(inputdim), dtype=float)
            outputone = np.zeros(shape=(outputdim), dtype=float)

            inputList=str.split("\\|")[0].split(";");
            outputStr=str.split("\\|")[1]
            for inputStr in inputList:
                if(len(inputStr)>0):
                    inputValue=inputStr.split(":");
                    inputone[int(inputValue[0])]=float(inputValue[1])
            outputone[0]=float(outputStr.split(";")[0])
            outputone[1] = float(outputStr.split(";")[1])
            input.append(inputone)
            output.append(outputone)

    return input,output

class DataSet(object):
    def __init__(self, input,output):
        self._num_examples = len(input)
        self._index_in_epoch = 0

        # self.input=input
        # self.output=output
        self.input,self.output=shuffle(input,output)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if(self._index_in_epoch>self._num_examples):
            self._index_in_epoch=0
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
        end=self._index_in_epoch

        return self.input[start:end],self.output[start:end]

if __name__=="__main__":
    # input, output=load_train("C:\\Users\\Administrator\\PycharmProjects\\myPostitionModel\\file\\trainData.txt",8,2)
    # print(input)
    # print(np.array(output)[1:5])
    resutl=loadAlltrain("C:\\Users\\Administrator\\PycharmProjects\\myPostitionModel\\file\\trainAllData.txt")
    for i in resutl:
        eci=i

        inputdim=resutl[eci][0]
        print(eci,",",inputdim)


