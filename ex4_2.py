import os


def parse_file(file):
    objects = []
    is_obj = False
    arr = []
    j=0
    with open(file, encoding="utf-8") as fp:
        line = fp.readline()
        cnt = 1
        while line:
            line_str = line.strip()
            if line_str.startswith("Objects:"):
                print(j)
                j+=1
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
                    center = (int(left_x + (width / 2)),int(top_y + (hight / 2)))
                    arr.append(center)
                    # arr.append([left_x, top_y, width, hight])
                    # print("Line {}: {}".format(cnt, line.strip()))
                    # print("Line: {},{},{},{}".format(left_x, top_y, width, hight))

                except:
                    print(f"Nope:\t{split}")
                    arr.append((-1,-1))
            line = fp.readline()
            cnt += 1
        return objects


from scipy import spatial
import numpy as np

objects = parse_file("result.txt")
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
        final_result.append((-2,-2))
        final_result.append(point)
        continue
    # each row j is the distance between point object[i-1][j] to all the points objects[i]
    dis = spatial.distance.cdist(objects[prev], objects[i])
    min_dis = np.min(dis, axis=1)
    a = np.median(min_dis)
    b = np.median(min_dis)
    if min_dis[index] > 2*a:
        change_car+=1
        # skip
        final_result.append((-1,-1))
    else:
        prev = i
        new_index = np.argmin(dis,axis=1)[index]
        point = lst[new_index]
        final_result.append(point)
        print(point)
        index = new_index

    # np.argwhere(objects[i - 1] == point)

    # print("-")
    # print(lst)
import cv2
cap = cv2.VideoCapture('results.avi')
from collections import deque
from scipy.stats import linregress
y = (np.array(final_result)[:3].flatten())[1::2]
x = (np.array(final_result)[:3].flatten())[::2]
slope = linregress(x,y)[0]
d = deque(final_result[::-1])

while(cap.isOpened()):
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    try:
        tup = d.pop()
        if tup == (-1,-1) or (-2,-2):
            print("FAIL")
        else:
            cv2.circle(gray,d.pop(), 20, (255,0,255), thickness=20)
    except:
        pass
    try:
        gray = cv2.resize(gray, (600, 600))
        cv2.imshow('frame', gray)
    except:
        print("Error - Exit")
        break


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()