

# 여기에 GenAI가 생성한 코드를 붙여넣으세요

# 주제 A 코드를 mosaic_app.py로 저장

mosaic_code = """

# 여기에 GenAI가 생성한 코드를 붙여넣으세요

"""

import streamlit as st

import cv2

import numpy as np

from openvino.runtime import Core

import io



# 🌟 [1단계] OpenVINO 얼굴 검출 클래스 정의

class OpenVINODetector:

    def __init__(self, model_path, confidence=0.5):

        ie = Core()

        model = ie.read_model(model_path)

        self.compiled = ie.compile_model(model, "CPU")

        self.input_layer = self.compiled.input(0)

        self.output_layer = self.compiled.output(0)

        self.confidence = confidence

        self.input_h = self.input_layer.shape[2]

        self.input_w = self.input_layer.shape[3]



    def detect(self, frame):

        h, w = frame.shape[:2]

        # 전처리

        img = cv2.resize(frame, (self.input_w, self.input_h))

        img = img.transpose(2, 0, 1)

        input_data = np.expand_dims(img, 0).astype(np.float32)

       

        # 추론

        results = self.compiled([input_data])[self.output_layer]

       

        detections = []

        for det in results[0][0]:

            conf = det[2]

            if conf > self.confidence:

                x1 = max(0, int(det[3] * w))

                y1 = max(0, int(det[4] * h))

                x2 = min(w, int(det[5] * w))

                y2 = min(h, int(det[6] * h))

                detections.append((x1, y1, x2, y2, float(conf)))

        return detections



# 🌟 [2단계] 모자이크 처리 함수

def apply_mosaic(img, x1, y1, x2, y2, block_size):

    roi = img[y1:y2, x1:x2]

    if roi.size == 0: return img

   

    # ROI 축소 후 다시 확대 (INTER_NEAREST로 격자 효과 생성)

    h, w = roi.shape[:2]

    small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_NEAREST)

    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

   

    img[y1:y2, x1:x2] = mosaic

    return img



# 🌟 [3단계] Streamlit UI 구성

st.set_page_config(page_title="얼굴 자동 모자이크 앱", layout="wide")

st.title("🛡️ 얼굴 자동 모자이크 앱")

st.write("사진을 업로드하면 인텔 AI가 자동으로 얼굴을 찾아 프라이버시를 보호합니다.")



# 사이드바 설정

st.sidebar.header("⚙️ 설정")

block_size = st.sidebar.slider("모자이크 강도 (높을수록 뭉개짐)", 5, 50, 15)

conf_threshold = st.sidebar.slider("신뢰도 임계값 (AI의 확신 정도)", 0.3, 0.9, 0.5)



# 모델 로드 (캐싱을 통해 성능 최적화)

@st.cache_resource

def load_model(threshold):

    model_path = "face-detection-adas-0001.xml"

    return OpenVINODetector(model_path, confidence=threshold)



face_detector = load_model(conf_threshold)



# 파일 업로더

uploaded_file = st.file_uploader("이미지를 선택하세요 (JPG, PNG)", type=['jpg', 'png', 'jpeg'])



if uploaded_file is not None:

    # 🌟 이미지 읽기 (np.frombuffer + cv2.imdecode)

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

   

    # 원본 복사 및 처리

    result_bgr = img_bgr.copy()

   

    # 얼굴 검출 실행

    detections = face_detector.detect(img_bgr)

   

    # 모자이크 적용

    for (x1, y1, x2, y2, _) in detections:

        result_bgr = apply_mosaic(result_bgr, x1, y1, x2, y2, block_size)

   

    # 결과 변환

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

   

    # 결과 리포트

    st.divider()

    st.metric("검출된 얼굴 수", f"{len(detections)}명")

   

    # 화면 표시 (st.columns(2) 나란히 표시)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("원본 이미지")

        st.image(img_rgb, use_container_width=True)

    with col2:

        st.subheader("모자이크 결과")

        st.image(result_rgb, use_container_width=True)

   

    # 🌟 결과 이미지 다운로드 버튼

    is_success, buffer = cv2.imencode(".png", result_bgr)

    if is_success:

        st.download_button(

            label="🖼️ 결과 이미지 다운로드",

            data=buffer.tobytes(),

            file_name="mosaic_result.png",

            mime="image/png"

        )

else:

    st.info("왼쪽에서 이미지를 업로드해 주세요.")

