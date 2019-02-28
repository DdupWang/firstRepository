# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import DataProcess

def getAllResult(eci,input,output):
    path = "C:\\Users\\Administrator\\PycharmProjects\\myPostitionModel\\modeldata"
    filemodel = str(eci) + ".ckpt.meta"
    modeldata = str(eci) + ".ckpt"
    modelpath=os.path.join(path,filemodel)

    print(modelpath)
    if(os.path.exists(modelpath)):

        g = tf.Graph()
        with tf.Session(graph=g) as sess:
            saver = tf.train.import_meta_graph(modelpath)
            saver.restore(sess, os.path.join(path,modeldata))
            y_pre = g.get_tensor_by_name("y_pre:0")
            x = g.get_tensor_by_name("x:0")
            y = g.get_tensor_by_name("y:0")

            result = sess.run(y_pre, feed_dict={x: input, y: output})
            # print(result)
            # print(result[0])
            position=([str(
                pow(pow(result[i][0] - output[i][0], 2) + pow(result[i][1] - output[i][1], 2), 0.5)) + ";" + ','.join(
                str(i) for i in result[i]) + ";" + ','.join(str(i) for i in output[i]) for i in range(len(result))])

            lf = list(filter(lambda x: float(x.split(";")[0]) < 100, position))
            # print(lf)
            print(float(len(lf))/float(len(position)))
            positionPath="C:\\Users\\Administrator\\PycharmProjects\\myPostitionModel\\result\\result.txt"
            for i in position:
                with open(positionPath,"a+") as f:
                    f.write(str(eci)+"\t"+i+"\n")


def preCNN(eci,input,output):
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        path = "C:\\Users\\Administrator\\PycharmProjects\\myPostitionModel\\modeldata"
        filemodel = str(eci) + ".ckpt.meta"
        modeldata = os.path.join(path,str(eci) + ".ckpt")
        modelpath = os.path.join(path ,filemodel)

        saver = tf.train.import_meta_graph(modelpath)
        saver.restore(sess, modeldata)
        y_pre = g.get_tensor_by_name("y_pre:0")
        x = g.get_tensor_by_name("x:0")
        y = g.get_tensor_by_name("y:0")
        result = sess.run(y_pre, feed_dict={x: input, y: output})
        # print(result)
        # print(result[0])
        print([str(
            pow(pow(result[i][0] - output[i][0], 2) + pow(result[i][1] - output[i][1], 2), 0.5)) + ";" + ','.join(
            str(i) for i in result[i]) + ";" + ','.join(str(i) for i in output[i]) for i in range(len(result))])

if __name__=="__main__":
    alltestData = DataProcess.loadAlltrain(
        "C:\\Users\\Administrator\\PycharmProjects\\myPostitionModel\\file\\testAllData.txt")

    for i in alltestData:
        if(str(i) != "139059338" or str(i) != "140403851"):
            try:
                input=alltestData[i][1]
                output=alltestData[i][2]
                getAllResult(i,input,output)
            except:
                print(i," 有问题")