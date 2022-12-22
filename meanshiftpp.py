import cv2 as cv
import numpy as np
import math
import time


def meanshiftpp(src, h):
    print("Start: %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    img = np.copy(src)
    rows = img.shape[0]
    cols = img.shape[1]
    d = 3
    curr_img = np.copy(img)
    prev_img = np.copy(curr_img)
    images = []
    step = 0
    max_iterations = 5
    while True:
        counter = np.zeros((256, 256, 256))
        values = np.empty((256, 256, 256), object)

        for i in range(rows):
            for j in range(cols):
                y = int(math.floor(prev_img[i][j][0] / h))
                x = int(math.floor(prev_img[i][j][1] / h))
                z = int(math.floor(prev_img[i][j][2] / h))
                counter[y][x][z] += 1
                if not values[y][x][z]:
                    values[y][x][z] = [0] * d
                values[y][x][z][0] += prev_img[i][j][0]
                values[y][x][z][1] += prev_img[i][j][1]
                values[y][x][z][2] += prev_img[i][j][2]

        curr_img = np.zeros((rows, cols, 3), np.uint8)
        shift = 0
        for i in range(rows):
            for j in range(cols):
                count_sum = 0
                val_sum = [0] * d
                r = 1
                for m in range(-r, r + 1):
                    for n in range(-r, r + 1):
                        for o in range(-r, r + 1):
                            i_cell = int(math.floor(prev_img[i][j][0] / h)) + m
                            j_cell = int(math.floor(prev_img[i][j][1] / h)) + n
                            k_cell = int(math.floor(prev_img[i][j][2] / h)) + o
                            if 0 <= i_cell < 256 and 0 <= j_cell < 256 and 0 <= k_cell < 256:
                                if values[i_cell][j_cell][k_cell] is not None:
                                    count_sum += counter[i_cell][j_cell][k_cell]
                                    val_sum[0] += values[i_cell][j_cell][k_cell][0]
                                    val_sum[1] += values[i_cell][j_cell][k_cell][1]
                                    val_sum[2] += values[i_cell][j_cell][k_cell][2]
                for k in range(d):
                    curr_img[i][j][k] = val_sum[k] / count_sum
                    shift += abs(int(curr_img[i][j][k]) - int(prev_img[i][j][k]))
        prev_img = np.copy(curr_img)
        images.append(curr_img)
        step += 1
        # print('Difference-iter: %d-%d' % (shift, iter))
        if shift < 1:
            break
    exec_time = time.time() - start_time
    print("\tSecondi: %s" % exec_time)
    print("Finish: %s" % (time.asctime(time.localtime(time.time()))))
    return [np.copy(curr_img), exec_time]
