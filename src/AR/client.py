import socket
import json
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

K = np.array([[246.38967313,   0.        , 320.        ],
       [  0.        , 246.38967313, 180.        ],
       [  0.        ,   0.        ,   1.        ]])

def project_to_camera(world_coords, K, R):
    projection_matrix = K @ R
    projected = (projection_matrix @ world_coords).T
    projected = projected / (projected[:, -1].reshape(-1, 1)) 
    return projected[:, :-1]

url = 0
cap = cv2.VideoCapture(url)
#steering_wheel = cv2.imread('steering_wheel_image.jpg',0)
res = []
def main():
    first = True
    host = '0.0.0.0'
    port = 5001
    put_text = False

    s = socket.socket()
    s.bind((host, port))

    s.listen()

    print("Server Started.")
    sock, addr = s.accept()
    print("client connected ip:<" + str(addr) + ">")

    sock.send("Connection successful".encode('utf-8'))
    start = time.time()

    while True:
        data = sock.recv(1024).decode('utf-8')
        try:
            if data:
                if time.time() - start < 2:
                    continue
                orientation = json.loads(data)
                heading = orientation['camera_heading']
                yaw = heading
                if first:
                    reference = yaw
                    first = False
                else:
                    yaw = yaw - reference
                r = R.from_euler('y', -yaw, degrees=True).as_matrix()
                world_coord = np.array([[0, 0., 4]])
                uv = project_to_camera(world_coord.T , K, r).astype('int32')
                #M = cv2.getRotationMatrix2D((240/2,240/2),-yaw,1)
                #dst = cv2.warpAffine(steering_wheel,M,(240,240))
                #cv2.imshow("steering wheel", dst) 
        except Exception as e:
            pass

        ret, frame = cap.read()
        #cv2.circle(frame, (uv[0][0], uv[0][1]), 10,(255, 255, 0),-1)
        if put_text:
            cv2.rectangle(frame, (uv[0][0]-5, uv[0][1]), (uv[0][0] + 175, uv[0][1] + 70), (255, 255, 255), -1)
            cv2.putText(frame, 'Good Coffee', (uv[0][0], uv[0][1] + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.imshow("Capturing", frame)
        res.append(frame)

        if cv2.waitKey(10) & 0xFF == ord('q') or ret is False:
            put_text = True

if __name__ == '__main__':
    main()
