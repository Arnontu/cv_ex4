import os
import cv2
import ctypes
from collections import deque
from scipy.stats import linregress
from scipy import spatial
import numpy as np

## get Screen Size
# user32 = ctypes.windll.user32
# screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
SCREEN_SIZE = (1536, 864)
INPUT_VIDEO = 'results_7.avi'
OUTPUT_VIDEO = "final_result.avi"
BOUNDING_BOX_TXT = "results_7.txt"
CAR_POINTER = 3
CHANGE_COUNTER_MAX = 3
CHANGE_COUNTER = CHANGE_COUNTER_MAX
DISPLAY = False
RANDOM = False


def fit_Image(oriimg):
    W, H = SCREEN_SIZE
    W = int(0.9 * W)
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
                # print(j)
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
                    width = int(line_str[line_str.index("width:") + 6:].strip().split(" ")[0])
                    height = int(line_str[line_str.index("height:") + 7:].strip().split(" ")[0][:-1])
                    left_x = int(line_str[line_str.index("left_x:") + 7:].strip().split(" ")[0])
                    top_y=int(line_str[line_str.index("top_y:") + 7:].strip().split(" ")[0])
                    # left_x, top_y, width, hight = int(split[3]), int(split[5]), int(split[7]), int(split[9][:-1])

                    center = (int(left_x + (width / 2)), int(top_y + (height / 2)))
                    arr.append(center)


                except Exception as e:
                    print(str(e))

                    arr.append((-1, -1))
            line = fp.readline()
            cnt += 1
        return objects


objects = parse_file(BOUNDING_BOX_TXT)
final_result = []

for i, lst in enumerate(objects):
    if CHANGE_COUNTER == CHANGE_COUNTER_MAX:
        CHANGE_COUNTER = 0
        prev = 0
        # if CAR_POINTER>=len(lst):
        if RANDOM or CAR_POINTER >= len(lst):
            CAR_POINTER = np.random.choice(len(lst) - 1)
        point = lst[CAR_POINTER]
        index = CAR_POINTER
        final_result.append((-2, -2))
        final_result.append(point)
        continue
    dis = spatial.distance.cdist(objects[prev], objects[i])
    min_dis = np.min(dis, axis=1)
    # a = np.median(min_dis)
    # b = np.median(min_dis)
    if min_dis[index] > 2 * np.median(min_dis):
        CHANGE_COUNTER += 1
        # skip
        final_result.append((-1, -1))
    else:
        CHANGE_COUNTER = 0
        prev = i
        new_index = np.argmin(dis, axis=1)[index]
        point = lst[new_index]
        final_result.append(point)
        index = new_index

cap = cv2.VideoCapture(INPUT_VIDEO)
all_points = np.array(final_result)
all_point_adj = np.zeros(all_points.shape)
boundaries = np.where(all_points[:, 0] == -2)[0]
for i in range(len(boundaries) - 1):
    seg = all_points[boundaries[i] + 1:boundaries[i + 1]]
    seg = seg[seg != -1].reshape(-1, 2)
    y = (seg.flatten())[1::2]
    x = (seg.flatten())[::2]
    res = linregress(x, y)[:2]
    all_point_adj[boundaries[i]:boundaries[i + 1]] = res

if boundaries[-1] < len(all_points):
    seg = all_points[boundaries[-1] + 1:]
    seg = seg[seg != -1].reshape(-1, 2)
    y = (seg.flatten())[1::2]
    x = (seg.flatten())[::2]
    res = linregress(x, y)[:2]
    all_point_adj[boundaries[-1]:] = res
d = deque(final_result[::-1])
main_counter = 0
d_arrow = deque(all_point_adj[::-1])
writer = None
while (cap.isOpened()):
    ret, frame = cap.read()
    try:
        # Option A
        # slope, interceptfloat = d_arrow.pop()
        # (x1, y1) = (0, int(interceptfloat))
        # (x2, y2) = (frame.shape[1], int(frame.shape[1] * slope + interceptfloat))
        # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # cv2.circle(frame, d.pop(), 15, (0, 0, 255), thickness=30)

        # Option B
        cv2.circle(frame, final_result[main_counter], 15, (0, 0, 255), thickness=30)
        side_size = 2
        if (side_size - 1) < main_counter < len(final_result) - side_size + 1:
            seg = np.array(final_result[main_counter - side_size:main_counter + side_size + 1])
            if (-1, -1) not in seg and (-2, -2) not in seg:
                y = (seg.flatten())[1::2]
                x = (seg.flatten())[::2]
                slope, interceptfloat = linregress(x, y)[:2]
                (x1, y1) = (0, int(interceptfloat))
                (x2, y2) = (frame.shape[1], int(frame.shape[1] * slope + interceptfloat))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        main_counter += 1


    except Exception as e:
        pass
    try:
        frame = cv2.resize(frame, (600, 600))
        frame = fit_Image(frame)
        if DISPLAY:
            cv2.imshow('frame', frame)
        if writer is None:
            (h, w) = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps,
                                     (h, w))
        writer.write(frame)
    except Exception as e:
        print(str(e))
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
