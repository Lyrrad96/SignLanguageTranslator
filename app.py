# import numpy as np
# from keras.models import model_from_json
import operator
import cv2
import sys, os

import streamlit as st
import av
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import threading
# # Loading the model
# json_file = open("vgg-aug_model.json", "r")
# model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(model_json)
# # load weights into new model
# loaded_model.load_weights("VGG16-Aug.h5")
# print("Loaded model from disk")

# cap = cv2.VideoCapture(0)

# # Category dictionary
# categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
#                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
#                 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space',}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
#     result = loaded_model.predict(test_image.reshape(1, 64, 64, 3))
#     prediction = {'A': result[0][0], 'B': result[0][1], 
#                   'C': result[0][2], 'D': result[0][3],
#                   'E': result[0][4], 'F': result[0][5],
#                   'G': result[0][6], 'H': result[0][7],
#                   'I': result[0][8], 'J': result[0][9],
#                   'K': result[0][10], 'L': result[0][11],
#                   'M': result[0][12], 'N': result[0][13],
#                   'O': result[0][14], 'P': result[0][15],
#                   'Q': result[0][16], 'R': result[0][17],
#                   'S': result[0][18], 'T': result[0][19],
#                   'U': result[0][20], 'V': result[0][21],
#                   'W': result[0][22], 'X': result[0][23],
#                   'Y': result[0][24], 'Z': result[0][25],
#                   'del': result[0][26], 'nothing': result[0][27],
#                   'space': result[0][28],}
#     #print(prediction)
#     # Sorting based on top prediction
#     prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
#     # Displaying the predictions
#     cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)    
#     cv2.imshow("Frame", frame)
    
#     interrupt = cv2.waitKey(10)
#     if interrupt & 0xFF == 27: # esc key
#         break
        
 
cap.release()
cv2.destroyAllWindows()

#Writes the following code to a file and stores it in the file system on the left

# from google.colab.patches import cv2_imshow
# ##

# from IPython.display import display, Javascript
# from google.colab.output import eval_js
# from base64 import b64decode

# def take_photo(filename='photo.jpg', quality=0.8):
#   js = Javascript('''
#     async function takePhoto(quality) {
#       const div = document.createElement('div');
#       const capture = document.createElement('button');
#       capture.textContent = 'Capture';
#       div.appendChild(capture);

#       const video = document.createElement('video');
#       video.style.display = 'block';
#       const stream = await navigator.mediaDevices.getUserMedia({video: true});

#       document.body.appendChild(div);
#       div.appendChild(video);
#       video.srcObject = stream;
#       await video.play();

#       // Resize the output to fit the video element.
#       google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

#       // Wait for Capture to be clicked.
#       await new Promise((resolve) => capture.onclick = resolve);

#       const canvas = document.createElement('canvas');
#       canvas.width = video.videoWidth;
#       canvas.height = video.videoHeight;
#       canvas.getContext('2d').drawImage(video, 0, 0);
#       stream.getVideoTracks()[0].stop();
#       div.remove();
#       return canvas.toDataURL('image/jpeg', quality);
#     }
#     ''')
#   display(js)
#   data = eval_js('takePhoto({})'.format(quality))
#   binary = b64decode(data.split(',')[1])
#   with open(filename, 'wb') as f:
#     f.write(binary)
#   return filename

# try:
#   filename = take_photo()
#   print('Saved to {}'.format(filename))
  
#   # Show the image which was just taken.
#   display(Image(filename))
# except Exception as err:
#   # Errors will be thrown if the user does not have a webcam or if they do not
#   # grant the page permission to access it.
#   print(str(err))
# ##

#Tab Heading and Icon
st.set_page_config(page_title="Sign Language Classifier", page_icon="👍")

#Sidebar options
app_mode = st.sidebar.selectbox('Choose the model to use',['Resnet50', 'VGG16', 'InceptionV3'])

st.sidebar.markdown('---')

st.title('Sign Language Translator')
st.title("Webcam Live Feed")

# #Video Stream 
# class VideoProcessor(VideoProcessorBase):
#   def __init__(self):
#     self.style = 'color'
#   def recv(self, frame):
#     img = frame.to_ndarray(format="bgr24")
       
#     # image processing code here
#     return av.VideoFrame.from_ndarray(img, format="bgr24")
# webrtc_streamer(key="vpf", video_processor_factory=VideoProcessor)

st.markdown (
  """
  <style>
  [data-testid="stSidebar"][aria-expanded="true"] > div:first_child{
    width: 350px
  }
  [data-testid="stSidebar"][aria-expanded="false"] > div:first_child{
    width: 350px
    margin_left: -350px
  }
  </style>

  """,
  unsafe_allow_html=True,
)

#st.sidebar.title('Title')
#st.sidebar.subheader('head')

@st.cache()
def image_resize(image, width=None, height=None, inter =cv2.INTER_AREA):
  dim = None
  (h, w) = image.shpe[:2]

  if width is None and height is None:
    return image

  if width is None:
    r = width/float(w)
    dim = (int(w * r), height)

  else: 
    r = width/float(w)
    dim = (width, int(h + r))

    #resize
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# ####HERE
# #start button
# use_webcam = st.sidebar.button('Use Webcam')
# #empty frame
# stframe = st.empty()
# video_file_buffer = st.sidebar.file_uploader("Upload", type = ['mp4'])

# # tffile = tempfile.NamedTemporaryFile(delete = false)

# # if not video_file_buffer:
# #   if use_webcam:
# vid = cv2.VideoCapture(0)
# #   else:
# #     vid = cv2.VideoCapture(DEMO)
# #     tffile.name = DEMO
# # else:
# #   tffile.write(video_file_buffer.read())
# #   vid = cv2.VideoCapture(tffile.name)
# width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps_input = int(vid.get(cv2.CAP_PROP_FPS))
# print("WAWHOPSSSAP")

# i = 0
# while vid.isOpened():
#   i += 1
#   ret, frame = vid.read()
#   print(vid.read())
#   print("WAWHOPSSSAP")
#   print("res", ret, frame)
#   if not ret:
#     continue

# frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
# #results = face_mesh.process(frame)
# frame.flags.writeable = True
# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# frame = cv2.resize(frame, (0, 0), fx = 0.8, fy = 0.8)
# frame = image_resize(image = frame, width = 640)
# stframe.image_(frame, channels = "BGR", use_column_width = True)

# # while(True):
# #     ret, frame = vid.read()
# #     cv2_imshow(vid)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# vid.release()
# # cv2.destroyAllWindows()

# import the opencv library


# # define a video capture object
# vid = cv2.VideoCapture(0)

# while(True):
	
# 	# Capture the video frame
# 	# by frame
# 	ret, frame = vid.read()

# 	# Display the resulting frame
# 	cv2.imshow('frame', frame)
	
# 	# the 'q' button is set as the
# 	# quitting button you may use any
# 	# desired button of your choice
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()


if app_mode == 'Resnet50':
 None
elif app_mode == 'VGG16':
  None
elif app_mode == 'InceptionV3':
  "👍"
