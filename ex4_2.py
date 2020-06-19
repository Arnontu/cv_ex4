import os
import cv2
import ctypes
from collections import deque
from scipy.stats import linregress
from scipy import spatial
import numpy as np

## get Screen Size
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
def fit_Image(oriimg):
    W, H = screensize
    W = int(0.9*W)
    H = int(0.9 * H)
    height, width, depth = oriimg.shape
    scaleWidth = float(W) / float(width)
    scaleHeight = float(H) / float(height)
    if scaleHeight > scaleWidth:
        imgScale = scaleWidth
    else:
        imgScale = scaleHeight
    newX, newY = oriimg.shape[1] * imgScale, oriimg.shape[0] * imgScale
    newimg = cv2.resize(oriimg, (int(newX), int(newY)))
    return newimg


def parse_file(file):
    objects = []
    is_obj = False
    arr = []
    j = 0
    with open(file, encoding="utf-8") as fp:
        line = fp.readline()
        cnt = 1
        while line:
            line_str = line.strip()
            if line_str.startswith("Objects:"):
                print(j)
                j += 1
                is_obj = True
            elif line_str.startswith("FPS"):
                if arr:
                    objects.append(arr)
                arr = []
                is_obj = False
            elif is_obj and line_str != "":
                split = list(filter(None, line_str.split(" ")))
                try:
                    left_x, top_y, width, hight = int(split[3]), int(split[5]), int(split[7]), int(split[9][:-1])
                    center = (int(left_x + (width / 2)), int(top_y + (hight / 2)))
                    arr.append(center)
                    # arr.append([left_x, top_y, width, hight])
                    # print("Line {}: {}".format(cnt, line.strip()))
                    # print("Line: {},{},{},{}".format(left_x, top_y, width, hight))

                except:
                    print(f"Nope:\t{split}")
                    arr.append((-1, -1))
            line = fp.readline()
            cnt += 1
        return objects




objects = parse_file("results_7.txt")
final_result = []
pointer = 5
change_car_fail = 3
change_car = change_car_fail

for i, lst in enumerate(objects):
    if change_car == change_car_fail:
        change_car = 0
        prev = 0
        point = lst[pointer]
        index = pointer
        print(point)
        final_result.append((-2, -2))
        final_result.append(point)
        continue
    # each row j is the distance between point object[i-1][j] to all the points objects[i]
    dis = spatial.distance.cdist(objects[prev], objects[i])
    min_dis = np.min(dis, axis=1)
    a = np.median(min_dis)
    b = np.median(min_dis)
    if min_dis[index] > 2 * a:
        change_car += 1
        # skip
        final_result.append((-1, -1))
    else:
        change_car=0
        prev = i
        new_index = np.argmin(dis, axis=1)[index]
        point = lst[new_index]
        final_result.append(point)
        print(point)
        index = new_index


cap = cv2.VideoCapture('results_7.avi')
all_points = np.array(final_result)
all_point_adj = np.zeros(all_points.shape)
boundries = np.where(all_points[:, 0] == -2)[0]
for i in range(len(boundries) - 1):
    seg = all_points[boundries[i] + 1:boundries[i + 1]]
    seg = seg[seg != -1].reshape(-1, 2)
    y = (seg.flatten())[1::2]
    x = (seg.flatten())[::2]
    res = linregress(x,y)[:2]
    all_point_adj[boundries[i]:boundries[i + 1]] = res

if boundries[-1] < len(all_points):
    seg = all_points[boundries[-1]+1:]
    seg = seg[seg != -1].reshape(-1, 2)
    y = (seg.flatten())[1::2]
    x = (seg.flatten())[::2]
    res = linregress(x,y)[:2]
    all_point_adj[boundries[-1]:] = res
d = deque(final_result[::-1])
d_arrow = deque(all_point_adj[::-1])


while (cap.isOpened()):
    ret, frame = cap.read()
    try:
        slope, interceptfloat = d_arrow.pop()
        (x1, y1) = (0,int(interceptfloat))
        (x2, y2) = (frame.shape[1],int(frame.shape[1]*slope+interceptfloat))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, d.pop(), 20, (255, 0, 255), thickness=20)
    except Exception as e:
        pass
    try:
        frame = cv2.resize(frame, (600, 600))
        frame = fit_Image(frame)
        cv2.imshow('frame', frame)
    except:
        print("Finished")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
