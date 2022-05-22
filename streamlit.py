import argparse
import os
import sys
import queue
import threading
from pathlib import Path
from PIL import Image

import av
import pafy
import streamlit as st
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from aiortc.contrib.media import MediaPlayer

from detect import run, StopDetectSignal, detect_from_queue

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
PROJECT = Path(os.path.abspath("/working"))
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def app_detection_image(opt):
    def get_subdirs(b='.'):
        '''
            Returns all sub-directories in a specific Path
        '''
        result = []
        for d in os.listdir(b):
            bd = os.path.join(b, d)
            if os.path.isdir(bd):
                result.append(bd)
        return result

    def get_detection_folder():
        '''
            Returns the latest folder in a runs\detect
        '''
        return max(get_subdirs(os.path.join(opt.project)), key=os.path.getmtime)

    uploaded_file = st.sidebar.file_uploader(
        "Upload", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is None:
        return
    with st.spinner(text='Loading...'):
        st.sidebar.image(uploaded_file)
        opt.source = os.path.join(opt.source, uploaded_file.name)
        picture = Image.open(uploaded_file)
        picture.save(opt.source)

    if st.button('Detect'):
        with st.spinner(text='Preparing Images'):
            run(**dict(opt._get_kwargs()))
            for img in os.listdir(get_detection_folder()):
                st.image(str(Path(f'{get_detection_folder()}') / img))

            st.balloons()


def app_detection_youtube(opt):
    youtube_url = st.sidebar.text_input("youtube_url")

    if not youtube_url:
        return

    st.sidebar.video(youtube_url)
    st.write(youtube_url)
    opt.source = youtube_url
    opt.nosave = True  # only detect
    opt.line_thickness = 1

    def create_player():
        url = youtube_url
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        return MediaPlayer(best.url)

    class Yolov5VideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._opt = opt
            self._thread = threading.Thread(target=self._worker_thread)
            self._in_queue = queue.Queue()
            self._out_queue = queue.Queue()
            self._thread.start()

        def _worker_thread(self):
            detect_from_queue(in_queue=self._in_queue, out_queue=self._out_queue, **dict(self._opt._get_kwargs()))

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self._in_queue.put(img)
            img = self._out_queue.get()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def on_ended(self):
            self._queue.put(StopDetectSignal())
            self._thread.join(3000)

    webrtc_ctx = webrtc_streamer(
        key="object-detection-youtube",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        player_factory=create_player,
        video_processor_factory=Yolov5VideoProcessor,
    )


def app_detection_webcam(opt):
    opt.nosave = True  # only detect
    opt.line_thickness = 1

    class Yolov5VideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self._opt = opt
            self._thread = threading.Thread(target=self._worker_thread)
            self._in_queue = queue.Queue()
            self._out_queue = queue.Queue()
            self._thread.start()

        def _worker_thread(self):
            detect_from_queue(in_queue=self._in_queue, out_queue=self._out_queue, **dict(self._opt._get_kwargs()))

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self._in_queue.put(img)
            img = self._out_queue.get()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def on_ended(self):
            self._queue.put(StopDetectSignal())
            self._thread.join(3000)

    webrtc_ctx = webrtc_streamer(
        key="object-detection-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=Yolov5VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


if __name__ == '__main__':

    st.title('YOLOv5 Streamlit App')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=PROJECT / 'yolov5.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    source = ("Object Detection", "Object Detection (Youtube)", "Object Detection (Webcam https only)")
    source_index = st.sidebar.selectbox("Input", range(
        len(source)), format_func=lambda x: source[x])

    st.subheader(source[source_index])
    if source_index == 0:
        app_detection_image(opt)
    elif source_index == 1:
        app_detection_youtube(opt)
    elif source_index == 2:
        app_detection_webcam(opt)
