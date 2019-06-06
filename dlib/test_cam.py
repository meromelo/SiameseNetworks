
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
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
    vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vidar = vidw / vidh
    print(vidw)
    print(vidh)
 
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
