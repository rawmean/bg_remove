import numpy as np
import cv2

# Open Video
cap = cv2.VideoCapture('./data/video2.mp4')

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
print(frameIds)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Display median frame
cv2.imshow('background', medianFrame)
cv2.waitKey(0)

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('./data/bg_removed.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
while(ret):
  ret, frame = cap.read()
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  dframe = cv2.absdiff(frame_gray, grayMedianFrame)
  th, dframe = cv2.threshold(dframe, 20, 1, cv2.THRESH_BINARY)
  dframe_3ch = cv2.cvtColor(dframe, cv2.COLOR_GRAY2RGB)
  # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  diff_3ch = cv2.absdiff(frame, medianFrame)
  th, diff_3ch = cv2.threshold(diff_3ch, 25, 1, cv2.THRESH_BINARY)

  diff = np.max(diff_3ch, axis=2)
  diff_3ch = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)

  # masked = frame * dframe_3ch
  masked = frame * diff_3ch
  cv2.imshow('mask', dframe*255)
  cv2.imshow('masked', masked)
  out.write(masked)
  cv2.waitKey(100)

# Release video object
cap.release()
out.release()

# Destroy all windows
cv2.destroyAllWindows()
