

import argparse
import cv2
from timeit import default_timer as timer
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    args = parser.parse_args()
 
    if args.video == "0":
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(args.video)
    if not vid.isOpened():
        raise ImportError("Couldn't open video file or webcam.")
 
    # Compute aspect ratio of video
    sfps = vid.set(cv2.CAP_PROP_FPS, 60) 
    vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vidar = vidw / vidh
    print(vidw)
    print(vidh)

    print(f"fps={vid.get(cv2.CAP_PROP_FPS)}")
    print(f"w={vid.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"h={vid.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
 
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
 
    frame_count = 1
    while True:
        ret, frame = vid.read()
        if ret == False:
            print("Done!")
            return
 
        # Resized
        im_size = (300, 300)
        resized = cv2.resize(frame, im_size)
 
        # =================================
        # Image Preprocessing
        # =================================
 
        # =================================
        # Main Processing
        result = resized.copy() # dummy
        # result = frame.copy() # no resize
        # =================================
 
        # Calculate FPS
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS:" + str(curr_fps)
            curr_fps = 0
 
        # Draw FPS in top right corner
        cv2.rectangle(result, (250, 0), (300, 17), (0, 0, 0), -1)
        cv2.putText(result, fps, (255, 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
 
        # Draw Frame Number
        cv2.rectangle(result, (0, 0), (50, 17), (0, 0, 0), -1)
        cv2.putText(result, str(frame_count), (0, 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
 
        # Output Result
        cv2.imshow("Result", result)
 
        # Stop Processing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
        frame_count += 1
 
if __name__ == '__main__':
    main()
