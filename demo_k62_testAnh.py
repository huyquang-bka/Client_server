import os
import cv2
import numpy as np
from skimage import transform
from keras.models import model_from_json
import socket
import zipfile
from datetime import datetime

# C9_day PKSpace coordinates (other images coordinates could be offset by a few pixels)
# First row 130x130
# Should write each x-y pair on the same line instead of multiple lines
# Initialize, skip arr[0] for counting convenience in while loop

x_min = [];
y_min = [];
x_max = [];
y_max = []
x_min.append(0);
y_min.append(0);
x_max.append(0);
y_max.append(0)  # Redundant
# First row 130x130
x_min.append(18);
y_min.append(732)  # 1
x_max.append(x_min[1] + 130);
y_max.append(y_min[1] + 130)
x_min.append(216);
y_min.append(750)  # 2
x_max.append(x_min[2] + 130);
y_max.append(y_min[2] + 130)
x_min.append(346);
y_min.append(750)  # 3
x_max.append(x_min[3] + 130);
y_max.append(y_min[3] + 130)
x_min.append(490);
y_min.append(756)  # 4
x_max.append(x_min[4] + 130);
y_max.append(y_min[4] + 130)
x_min.append(636);
y_min.append(768)  # 5
x_max.append(x_min[5] + 130);
y_max.append(y_min[5] + 130)
x_min.append(784);
y_min.append(762)  # 6
x_max.append(x_min[6] + 130);
y_max.append(y_min[6] + 130)
x_min.append(934);
y_min.append(762)  # 7
x_max.append(x_min[7] + 130);
y_max.append(y_min[7] + 130)
x_min.append(1234);
y_min.append(762)  # 8
x_max.append(x_min[8] + 130);
y_max.append(y_min[8] + 130)
x_min.append(1372);
y_min.append(742)  # 9
x_max.append(x_min[9] + 130);
y_max.append(y_min[9] + 130)
x_min.append(1584);
y_min.append(718)  # 10
x_max.append(x_min[10] + 130);
y_max.append(y_min[10] + 130)
x_min.append(1684);
y_min.append(700)  # 11
x_max.append(x_min[11] + 130);
y_max.append(y_min[11] + 130)
# Second row 90x90
x_min.append(432);
y_min.append(512)  # 12
x_max.append(x_min[12] + 90);
y_max.append(y_min[12] + 90)
x_min.append(528);
y_min.append(512)  # 13
x_max.append(x_min[13] + 90);
y_max.append(y_min[13] + 90)
x_min.append(624);
y_min.append(512)  # 14
x_max.append(x_min[14] + 90);
y_max.append(y_min[14] + 90)
x_min.append(724);
y_min.append(512)  # 15
x_max.append(x_min[15] + 90);
y_max.append(y_min[15] + 90)
x_min.append(824);
y_min.append(506)  # 16
x_max.append(x_min[16] + 90);
y_max.append(y_min[16] + 90)
x_min.append(923);
y_min.append(508)  # 17
x_max.append(x_min[17] + 90);
y_max.append(y_min[17] + 90)
# Third row 75x75
x_min.append(417);
y_min.append(455)  # 18
x_max.append(x_min[18] + 75);
y_max.append(y_min[18] + 75)
x_min.append(502);
y_min.append(455)  # 19
x_max.append(x_min[19] + 75);
y_max.append(y_min[19] + 75)
x_min.append(590);
y_min.append(455)  # 20
x_max.append(x_min[20] + 75);
y_max.append(y_min[20] + 75)
x_min.append(680);
y_min.append(453)  # 21
x_max.append(x_min[21] + 75);
y_max.append(y_min[21] + 75)
x_min.append(765);
y_min.append(453)  # 22
x_max.append(x_min[22] + 75);
y_max.append(y_min[22] + 75)
x_min.append(853);
y_min.append(453)  # 23
x_max.append(x_min[23] + 75);
y_max.append(y_min[23] + 75)
x_min.append(946);
y_min.append(451)  # 24
x_max.append(x_min[24] + 75);
y_max.append(y_min[24] + 75)
# Fouth 70x70
x_min.append(757);
y_min.append(363)  # 25
x_max.append(x_min[25] + 70);
y_max.append(y_min[25] + 70)
x_min.append(829);
y_min.append(352)  # 26
x_max.append(x_min[26] + 70);
y_max.append(y_min[26] + 70)
x_min.append(958);
y_min.append(371)  # 27
x_max.append(x_min[27] + 60);
y_max.append(y_min[27] + 60)

# Sensor
x_min.append(164);
y_min.append(509)
x_min.append(1304);
y_min.append(765)

x_max.append(x_min[25] + 90);
y_max.append(y_min[25] + 90)
x_max.append(x_min[26] + 90);
y_max.append(y_min[26] + 90)

# Load Model
json_file = open('Model/modelC9f.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
Loaded_Model = model_from_json(loaded_model_json)
# Load weights into new model
Loaded_Model.load_weights("Model/weightC9f.h5")
print("Loaded")

####Variable
slot = 27
folder_write_base = 'Live/Base'
folder_write_tmp = 'Live/Img'
folder_write_busy = 'Live/Busy'
folder_write_free = 'Live/Free'

####Host and port to connect
host = '192.168.1.248'  ######ip server
port = 5523  ######port tự chọn

# Đọc File anh
folderTest = 'anhTest'
######Counting Frame
count = 0

for imageTest in os.listdir(folderTest):
    img = cv2.imread(f"{folderTest}/{imageTest}")
    imgCopy = img.copy()
    cv2.imwrite('C9_in.jpg', img)

    ####Lấy ngày giờ
    now = datetime.now()
    cur_time = now.strftime("%d.%m.%Y_%H.%M.%S")

    ###Folder lưu ảnh base
    cv2.imwrite(os.path.join(folder_write_base, 'C9_' + cur_time + '.jpg'), img)

    ####Mở file txt để ghi slot
    f1 = open('Frame.txt', 'w+')
    f2 = open(f'Live/Txt/C9_{cur_time}.txt', 'w+')

    ####Chia slot
    for i in range(1, slot + 1):
        images = []
        img_cut = imgCopy[y_min[i]:y_max[i], x_min[i]:x_max[i]]
        images.append(transform.resize(img_cut, (150, 150, 3)))
        images = np.array(images)
        prediction = Loaded_Model.predict(images)[0][0]

        if (prediction > 0.8):
            cv2.imwrite(f"{folder_write_busy}/C9_{cur_time} slot {i}.jpg", img_cut)
            cv2.rectangle(img, (x_min[i], y_min[i]), (x_max[i], y_max[i]), (0, 0, 255), 3)
            f1.write(f"{i} {str(round(prediction * 100, 2))} C9_{cur_time} slot {i}.jpg \n")
            f2.write(f"%{i} 1 \n")

        else:
            cv2.imwrite(f"{folder_write_free}/C9_{cur_time} slot {i}.jpg", img_cut)
            cv2.rectangle(img, (x_min[i], y_min[i]), (x_max[i], y_max[i]), (0, 255, 0), 3)
            f1.write(f"{i} {str(round(1 - prediction * 100, 2))} C9_{cur_time} slot {i}.jpg \n")
            f2.write(f"%{i} 0 \n")

    f1.close()
    f2.close()
    cv2.imwrite(f"{folder_write_tmp}/C9_{cur_time}.jpg", img)
    cv2.imwrite('C9_process.jpg', img)

    ####Day server
    s = socket.socket()
    if count == 0:
        print('[+] Client socket is created.')

    ####Try to connect
    try:
        connect = True
        s.connect((host, port))
        count += 1
        if count == 1:
            print(f'[+] Socket is connected to {host}')
    except:
        connect = False
        print('Connection fail')

    ####Nén ảnh và file txt
    zip_name = 'main.zip'
    with zipfile.ZipFile(zip_name, 'w') as file:
        file.write('C9_in.jpg')
        file.write("C9_process.jpg")
        file.write('Frame.txt')

    f = open(zip_name, 'rb')
    l = f.read()

    if (connect == True):
        if count == 1:
            print('connected')
        s.send(zip_name.encode())
        s.sendall(l)
    else:
        pass
    print(f"Frame: {count}")
    f.close()
    s.close()

####Thời gian trễ giữa 2 ảnh (seconds)
# time.sleep(10)

print("Finish!")
