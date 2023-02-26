import cv2 as cv
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import clustering
import meanshiftpp as ms
import regionGrowing as rg
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from os import listdir
from os.path import isfile, join
# import clustering as cs


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():
    '''wantImgGray = False
    mypath = '../images/'
    im_dirs = [f for f in listdir(mypath)]
    for im_dir in im_dirs:
        im_dir = join(mypath, im_dir)
        images = [f for f in listdir(im_dir) if isfile(join(im_dir, f))]
        h = 0
        if '15' in im_dir:
            h = 15
        elif '20' in im_dir:
            h = 20
        elif '25' in im_dir:
            h = 25
        elif '17' in im_dir:
            h = 17
        for image_name in images:
            print('H value: %d' % h)
            image_path = join(im_dir, image_name)
            img = cv.imread(image_path, cv.IMREAD_COLOR)
            color_string = 'color'
            if wantImgGray:
                color_string = 'gray'
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = np.stack((img,) * 3, axis=-1)
            img = cv.resize(img, (320, 240))
            mspp_img = ms.meanshiftpp(img, h)
            cv.imshow('Filtered', mspp_img[0])
            region_growed = clustering.segmentation_ms(mspp_img[0], h)

            size_string = '320x240'
            filename = 'resultnt/TRYSLICINGHSV' + color_string + size_string + image_name
            status = cv.imwrite(filename, mspp_img[0])
            print('Image written: %s - Status: %s' % (image_name, status))
            with open('resultnt/record.txt', 'a') as f:
                exec_time = mspp_img[1]
                result = 'H value: ' + str(h) + ' ' + image_name + ': ' + str(exec_time) + '\n'
                f.write(result)
                f.close()'''
    '''img = cv.imread('../DATASET/2092.jpg', cv.IMREAD_COLOR)
    # if img.shape[0] > 256 or img.shape[1] > 256:
    img = cv.resize(img, (481, 321))
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h = 20
    mspp_img = ms.meanshiftpp(img, h)
    # labeled = clustering.segmentation_ms(mspp_img[0], 40)
    labeledImg = rg.main(mspp_img[0], h*3)
    # print(labeledImg[1])
    filename = '../DATASET/2092.mat'
    mat = io.loadmat(filename)
    data_type = type(mat['groundTruth'])
    print(data_type)
    print(mat['groundTruth'][0][0][0][0][0].shape)
    print(mat['groundTruth'][0][0][0][0][0])
    print(mat.keys())
    segmentationMap = mat['groundTruth'][0][0][0][0][0]
    true_labels = np.array(mat['groundTruth'][0][0][0][0][0])
    # true_labels = true_labels[1:][1:]
    true_labels = true_labels.ravel()
    image_2d = np.array(labeledImg[0]).reshape(-1, 1)
    labels = np.array(labeledImg[1]).ravel()
    print(true_labels.shape)
    print(labels.shape)
    ARIscore = adjusted_rand_score(true_labels, labels)
    AMIscore = adjusted_mutual_info_score(true_labels, labels)
    # Visualizzazione della segmentation map
    plt.imshow(segmentationMap)
    # plt.axis('off')
    plt.show()
    print('SCORE: ', ARIscore)
    print('SCORE: ', AMIscore)
    cv.imshow('Filtered', mspp_img[0])
    cv.imshow('AfterMsPP', labeledImg[0])
    cv.waitKey(0)'''
    ARItotalScore = 0
    AMItotalScore = 0
    datasetPath = '../DATASET/'
    resultPath = '../DATASET/results/'
    imgSavePath = join(resultPath, 'predictedTEMP/')
    gtSavePath = join(resultPath, 'true_GTTEMP/')
    groundTruthPath = join(datasetPath, 'groundTruth/train/')
    imagesPath = join(datasetPath, 'images/train/')
    imgDirFiles = [f for f in listdir(imagesPath) if isfile(join(imagesPath, f))]
    gtDirFiles = [f for f in listdir(groundTruthPath) if isfile(join(groundTruthPath, f))]
    totImgs = len(imgDirFiles)
    for i in range(totImgs):
        # currImgName, currGTName in imgDirFiles, gtDirFiles:
        currImgName = imgDirFiles[i]
        currGTName = gtDirFiles[i]
        imgPath = join(imagesPath, currImgName)
        gtPath = join(groundTruthPath, currGTName)
        img = cv.imread(imgPath, cv.IMREAD_COLOR)
        imgGT = io.loadmat(gtPath)
        imgGT = imgGT['groundTruth'][0][0][0][0][0]
        imgGT = cv.resize(imgGT, (0, 0), fx=0.5, fy=0.5)
        # SAVING THE IMAGE GROUND TRUTH
        print(currGTName)
        currGTName = currGTName[:-4]
        print(currGTName)
        plt.imsave(gtSavePath + str(currGTName) + '.png', imgGT)
        # plt.imshow(imgGT)
        # plt.axis('off')
        # plt.savefig(gtSavePath + str(currImgName) + '.png')
        plt.close()
        gtShape = imgGT.shape
        img = cv.resize(img, (gtShape[1], gtShape[0]))
        true_labels = imgGT.ravel()
        h = 10
        clusteredImg = ms.meanshiftpp(img, h)
        labeledImg = rg.labeling(clusteredImg[0], h * 3)
        pred_labels = np.array(labeledImg[1]).ravel()
        tot_clusters = labeledImg[2]
        print(true_labels.shape)
        print(pred_labels.shape)
        ARIscore = adjusted_rand_score(true_labels, pred_labels)
        AMIscore = adjusted_mutual_info_score(true_labels, pred_labels)
        ARItotalScore += ARIscore
        AMItotalScore += AMIscore
        print('ARI SCORE: ', ARIscore)
        print('AMI SCORE: ', AMIscore)

        # SAVING IMG AND INFO OF CLUSTERING
        status = cv.imwrite(imgSavePath + str(currImgName), clusteredImg[0])
        print('Image written: %s - Status: %s' % (currImgName, status))
        with open(imgSavePath + 'record.txt', 'a') as f:
            exec_time = clusteredImg[1]
            result = 'H value: ' + str(h) + ' ' + currImgName + ':\n' + str(exec_time) + '\nTotC: ' + str(tot_clusters) + \
                     '\nARI: ' + str(ARIscore) + '\nAMI: ' + str(AMIscore) + '\n\n'
            f.write(result)
            f.close()
    with open(imgSavePath + 'record.txt', 'a') as f:
        ARIaverageScore = ARItotalScore / totImgs
        AMIaverageScore = AMItotalScore / totImgs
        result = 'ARI AverageScore: ' + str(ARIaverageScore) + '\nAMI AverageScore: ' + str(AMIaverageScore) + '\n\n'
        f.write(result)
        f.close()
        '''# Visualizzazione della segmentation map
        plt.imshow(imgGT)
        # plt.axis('off')
        plt.show()
        cv.imshow('Filtered', clusteredImg[0])
        cv.imshow('AfterMsPP', labeledImg[0])
        cv.waitKey(0)'''
        '''print('ImgPath: ' + str(imgPath) + ' gtPath: ' + str(gtPath))
        print('imgShape: ' + str(img.shape) + ' gtShape: ' + str(gtShape) + '\n\n')'''


if __name__ == '__main__':
    main()
