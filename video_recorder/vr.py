import time

import cv2
import numpy as np
from mss import mss

w = 1366
h = 768

sct = mss()

dimensions = {'top': 0, 'left': 0, 'width': w, 'height': h}

fps = 50

# (*'mp4v') (*'XVID') ? h264 x264 ?
out = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while True:
    count = 0
    current_time = time.time()
    while time.time() - current_time < 1 and count < 60:
        image = np.array(sct.grab(dimensions))
        out.write(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
        count += 1
    print(count)
    cv2.imshow("Press q", 0)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

out.release()
cv2.destroyAllWindows()
