import cv2 as cv
import numpy as np
import meanshiftpp as ms
from os import listdir
from os.path import isfile, join
# import clustering as cs


def main():
    wantImgGray = False
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
            # img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            mspp_img = ms.meanshiftpp(img, h)
            size_string = '320x240'
            filename = 'resultnt/TRYSLICINGHSV' + color_string + size_string + image_name
            status = cv.imwrite(filename, mspp_img[0])
            print('Image written: %s - Status: %s' % (image_name, status))
            with open('resultnt/record.txt', 'a') as f:
                exec_time = mspp_img[1]
                result = 'H value: ' + str(h) + ' ' + image_name + ': ' + str(exec_time) + '\n'
                f.write(result)
                f.close()
    '''img = cv.imread('../images/dataset15/tiger.jpg', cv.IMREAD_COLOR)
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = cv.resize(img, (320, 240))
    h = 17
    mspp_img = ms.meanshiftpp(img, h)
    cv.imshow('AfterMsPP', mspp_img[0])
    cv.waitKey(0)'''


if __name__ == '__main__':
    main()
