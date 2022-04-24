import streamlit as st
import av
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import threading

#Tab Heading and Icon
st.set_page_config(page_title="Sign Language Classifier", page_icon="🖕")

#Sidebar options
app_mode = st.sidebar.selectbox('Choose the model to use',['Resnet50', 'VGG16', 'InceptionV3'])


st.title('Sign Language Translator')
st.title("Webcam Live Feed")

#Video Stream 
class VideoProcessor(VideoProcessorBase):
  def __init__(self):
    self.style = 'color'
  def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")
       
    # image processing code here
    return av.VideoFrame.from_ndarray(img, format="bgr24")
webrtc_streamer(key="vpf", video_processor_factory=VideoProcessor)

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

if app_mode == 'Resnet50':
 None
elif app_mode == 'VGG16':
  None
elif app_mode == 'InceptionV3':
  "👍"
