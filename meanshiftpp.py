import cv2 as cv
import numpy as np
import math
import time


def checkValueExistence(storage, indexes, h):
    
    if storage[indexes[0]][indexes[1]][indexes[2]] is None:
        return False
    return storage[indexes[0]][indexes[1]][indexes[2]]
    

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
        values = np.zeros((256, 256, 256, 3))
        storage = np.empty((256, 256, 256), object)
        for i in range(rows):
            for j in range(cols):
                y = int(math.floor(prev_img[i][j][0] / h))
                x = int(math.floor(prev_img[i][j][1] / h))
                z = int(math.floor(prev_img[i][j][2] / h))
                counter[y][x][z] += int(1)
                '''if not values[y][x][z]:
                    values[y][x][z] = [0] * d'''
                values[y][x][z][0] += prev_img[i][j][0]
                values[y][x][z][1] += prev_img[i][j][1]
                values[y][x][z][2] += prev_img[i][j][2]

        curr_img = np.zeros((rows, cols, 3), np.uint8)
        shift = 0
        for i in range(rows):
            for j in range(cols):
                count_sum = 0
                val_sum = [0] * d
                # calculate
                blue_index = int(math.floor(prev_img[i][j][0] / h))
                green_index = int(math.floor(prev_img[i][j][1] / h))
                red_index = int(math.floor(prev_img[i][j][2] / h))
                if storage[blue_index][green_index][red_index] is None:
                    # blue_indexes for slicing
                    blue_start_block = blue_index - 1
                    blue_end_block = blue_index + 1
                    if blue_start_block < 0:
                        blue_start_block += 1
                    elif blue_end_block + 1 > 256:
                        blue_end_block -= 1
                    # green_indexes for slicing
                    green_start_block = green_index - 1
                    green_end_block = green_index + 1
                    if green_start_block < 0:
                        green_start_block += 1
                    elif green_end_block + 1 > 256:
                        green_end_block -= 1
                    # red_indexes for slicing
                    red_start_block = red_index - 1
                    red_end_block = red_index + 1
                    if red_start_block < 0:
                        red_start_block += 1
                    elif red_end_block + 1 > 256:
                        red_end_block -= 1
                    storage[blue_index][green_index][red_index] = [0] * d
                    # SLICING RADIUS = 1
                    # bgr_total = np.array(values[blue_start_block:blue_end_block+1, green_start_block:green_end_block+1, red_start_block:red_end_block+1])
                    bgr_total = values[blue_start_block:blue_end_block+1, green_start_block:green_end_block+1, red_start_block:red_end_block+1]
                    count_sum = np.array(counter[blue_start_block:blue_end_block+1, green_start_block:green_end_block+1, red_start_block:red_end_block+1])
                    bgr_totalCopy = [element for matrix in bgr_total for row in matrix for element in row]
                    '''if element is not None'''
                    bgr_total = np.copy(bgr_totalCopy)
                    val_sum = np.sum(bgr_total[:], axis=0)
                    count_sum = np.sum(count_sum)
                    for k in range(d):
                        curr_img[i][j][k] = val_sum[k] / count_sum
                        storage[blue_index][green_index][red_index][k] = curr_img[i][j][k]
                        shift += abs(int(curr_img[i][j][k]) - int(prev_img[i][j][k]))
                else:
                    for k in range(d):
                        curr_img[i][j][k] = storage[blue_index][green_index][red_index][k]
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
