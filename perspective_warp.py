import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

which_corner = {
            0: (0, 1),
            4: (1, 3),
            30: (3, 0),
            34: (2, 2),
        }

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        ids = [id[0] for id in ids]

        do_perspective_transform = all(idx in ids for idx in [0, 4, 30, 34])

        if do_perspective_transform:
            defining_corners = [None] * 4

            # we only want to keep the corners with ids 0, 4, 30, 34. These are the corners of the board
            for i in range(len(ids)):
                if ids[i] in [0, 4, 30, 34]:
                    corner_idx = which_corner[ids[i]][0]
                    x = int(corners[i][0][corner_idx][0])
                    y = int(corners[i][0][corner_idx][1])

                    defining_corners[which_corner[ids[i]][1]] = (x, y)

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    # draw text for corner id name
                    cv2.putText(frame, str(ids[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # perform perspective transformation and show it in a new window
            if len(defining_corners) == 4:
                real_board_width = 26
                real_board_height = 18.5
                ratio = real_board_width / real_board_height

                width_pixels = 500
                height_pixels = int(width_pixels * ratio) # to maintain aspect ratio of 26cm x 18.5cm
                
                src_points = np.array(defining_corners, dtype=np.float32)
                dst_points = np.array([[0,0],[height_pixels,0],[0, width_pixels],[height_pixels, width_pixels]], dtype=np.float32)
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                warped = cv2.warpPerspective(frame, matrix, (height_pixels, width_pixels))
                cv2.imshow('Warped Perspective', warped)
    else:
        cv2.putText(frame, "No ArUco marker detected", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Live ArUco Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
