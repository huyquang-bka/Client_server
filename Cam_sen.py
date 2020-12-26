import cv2
import os
import socket
import zipfile


count = 0
while True:
    count += 1
    # Receive data from Pi #####
    port = 5523

    ss = socket.socket()
    print('[+] Server socket is created.')

    ss.bind(('', port))
    print('[+] Socket is binded to {}'.format(port))

    ss.listen(5)
    print('[+] Waiting for connection...')

    con, addr = ss.accept()
    print('[+] Got connection from {}'.format(addr[0]))

    filename = con.recv(1024).decode()

    f = open(filename, 'wb')
    l = con.recv(1024)
    while (l):
        f.write(l)
        l = con.recv(1024)
    f.close()
    print('[+] Received file ' + filename)

    with zipfile.ZipFile(filename, 'r') as file:
        print('[+] Extracting files...')
        file.extractall()
        print('[+] Done')
    imgProcess = cv2.imread("C9_process.jpg")
    imgProcess = cv2.resize(imgProcess, dsize=None, fy=0.3, fx=0.3)
    cv2.imshow("Result", imgProcess)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

    os.remove(filename)
    con.close()
    ss.close()


