import math

import cv2 as cv
import numpy as np

pointShift = ((-1, -1),
              (-1, 0),
              (-1, 1),
              (0, -1),
              (0, 1),
              (1, -1),
              (1, 0),
              (1, -1))


def colordistance(color_a, color_b):
    blue = int(color_a[0]) - int(color_b[0])
    green = int(color_a[1]) - int(color_b[1])
    red = int(color_a[2]) - int(color_b[2])
    distance = math.sqrt((blue ** 2) + (green ** 2) + (red ** 2))
    return distance


def segmentation_ms(src, color_band):
    img = np.copy(src)
    labeledImg = np.zeros(img.shape)
    rows = img.shape[0]
    cols = img.shape[1]
    clusters = []
    member_mode_count = np.zeros((rows*cols))
    max_len_queue = 0
    labels = np.full((rows, cols), -1)
    label = -1
    for i in range(rows):
        for j in range(cols):
            if labels[i][j] < 0:
                label += 1
                labels[i][j] = label
                center_color = np.copy(img[i][j])
                b, g, r = np.copy(center_color)
                clusters.append(np.copy(center_color))
                neighbour_pts = [(i, j)]
                n = 0
                while len(neighbour_pts) > 0:
                    if len(neighbour_pts) > max_len_queue:
                        max_len_queue = len(neighbour_pts)
                    point = neighbour_pts.pop()
                    for k in range(8):
                        y = point[0] + pointShift[k][0]
                        x = point[1] + pointShift[k][1]
                        if 0 <= y < rows and 0 <= x < cols and labels[y][x] < 0:
                            neighbour_color = np.copy(img[y][x])
                            color_dist = colordistance(center_color, neighbour_color)
                            if color_dist < color_band:
                                labels[y][x] = label
                                neighbour_pts.append((y, x))
                                member_mode_count[label] += 1
                                b += int(neighbour_color[0])
                                g += int(neighbour_color[1])
                                r += int(neighbour_color[2])
                member_mode_count[label] += 1
                b /= member_mode_count[label]
                g /= member_mode_count[label]
                r /= member_mode_count[label]
                clusters[label][0] = b
                clusters[label][1] = g
                clusters[label][2] = r
                '''if j % 50 == 0 and i == 0:
                    print(j)'''
    cluster_size = label + 1
    for y in range(rows):
        for x in range(cols):
            label = labels[y][x]
            labeledImg[y][x] = int(label) + 1
            color = clusters[label]
            img[y][x][0] = color[0]
            img[y][x][1] = color[1]
            img[y][x][2] = color[2]
    print("Numero cluster: ", cluster_size)
    '''cv.imshow('Segmented', img)
    cv.waitKey(0)'''
    return [img, labeledImg]
