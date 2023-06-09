# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：Plot.py 
@Author     ：ZAY
@Time       ：2023/5/29 19:08
@Annotation : "画图 "
"""
import numpy as np
import matplotlib.pyplot as plt

def plotloss(x, y1, y2=None, picd_path=None,label=None):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if y2 == None:
        plt.plot(x, y1, label = label)
        plt.legend()
    else:
        line1, = plt.plot(x, y1, color = 'r', label='Train_loss')
        line2, = plt.plot(x, y2, color = 'b', label='Val_loss')
        plt.legend(handles=[line1, line2])  # 设置折线名称
    plt.savefig(picd_path)
    plt.show()

def plotShow(id_to_species,test_face_dataloader,label_batch = None,path = None):

    if label_batch == None:
        print("labels_batch is None!")
        return

    imgs_batch, labels_batch= next(iter(test_face_dataloader))

    plt.figure(figsize = (12, 8))
    plt.title('人脸朝向实验图像及预测结果')
    for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
        img = img.permute(1, 2, 0).numpy() # (H,W,C)
        plt.subplot(2, 3, i + 1) # subplot(numRows, numCols, plotNum) numRows 行 numCols 列
        # plt.title("预测结果:"+id_to_species.get(label.item()))
        plt.title("预测结果:" + label_batch[i])
        plt.imshow(img)
    plt.savefig(path)
    plt.show()  # 展示图片

def plotROC(fpr,tpr,roc_auc,path = None):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color = 'darkorange',
             lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc = "lower right")
    plt.savefig(path)
    plt.show()
