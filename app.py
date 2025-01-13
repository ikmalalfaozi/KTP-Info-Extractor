import capybara
import streamlit as st
import os
import json
from PIL import Image
import cv2
import tempfile
from docaligner import ModelType
from src import KTPDetector, CornerDetector, ImageWarper, ImageOrientation
from src.extract_info import KTPInfoExtractor

# Streamlit app
st.title("KTP Information Extractor Demo")
st.write("Unggah gambar KTP, dan sistem akan mengekstrak informasi dari gambar tersebut.")

# File uploader
uploaded_file = st.file_uploader("Unggah gambar KTP", type=["jpg", "jpeg", "png"])

# Model paths and configurations
segment_model_path = "models/YOLO11n-seg-ktp.pt"
cls_model_path = "models/YOLO11n-ktp-oc.pt"
extractor_model_name = "ikmalalfaozi/donut-base-finetuned-ktp-v1"

corner_model_config = {
    'model_type': ModelType.heatmap,  # Options: 'heatmap', 'point'
    'model_cfg': "fastvit_t8",
    'backend': capybara.Backend.cpu  # Options: 'cpu', 'cuda'
}

# Initialize models
@st.cache_resource
def initialize_models():
    ktp_detector = KTPDetector(segment_model_path)
    corner_detector = CornerDetector(
        model_type=corner_model_config['model_type'],
        model_cfg=corner_model_config['model_cfg'],
        backend=corner_model_config['backend']
    )
    ktp_orientation = ImageOrientation(cls_model_path)
    extractor = KTPInfoExtractor(model_name=extractor_model_name)
    return ktp_detector, corner_detector, ktp_orientation, extractor

ktp_detector, corner_detector, ktp_orientation, extractor = initialize_models()

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_image_path = temp_file.name

    # Display uploaded image
    st.image(Image.open(uploaded_file), caption="Gambar yang diunggah", use_container_width=True)
    st.write("Memproses gambar...")

    try:
        img = cv2.imread(temp_image_path)

        # Step 1: Detect KTP regions
        st.subheader("Tahap 1: Deteksi KTP")
        ktp_regions = ktp_detector.detect_and_segment(temp_image_path, conf=0.9)
        if not ktp_regions:
            st.warning("Tidak ada KTP yang terdeteksi dalam gambar.")
        else:
            st.write(f"Jumlah KTP yang terdeteksi: {len(ktp_regions)}")

            padding = 10
            results = []
            for i, ktp_region in enumerate(ktp_regions):
                bbox = ktp_region['bbox']
                mask = ktp_region['mask']

                # Display detected region
                x1, y1, x2, y2 = bbox
                st.write(f"KTP {i + 1}: Bounding Box: {bbox}")
                roi = img[y1:y2, x1:x2]
                st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption=f"KTP {i + 1} - ROI", use_container_width=True)

                # Step 2: Detect corners and warp the image
                st.subheader(f"Tahap 2: Corner Detection dan Warping untuk KTP {i + 1}")
                x_min = max(0, x1 - padding)
                y_min = max(0, y1 - padding)
                x_max = min(img.shape[1], x2 + padding)
                y_max = min(img.shape[0], y2 + padding)
                roi = img[y_min:y_max, x_min:x_max]
                mask_roi = mask[y_min:y_max, x_min:x_max] * 255

                corners = corner_detector.detect_corners(roi, mask_roi)
                warped_image = ImageWarper.warp_image(roi, corners, output_size=(856, 540))
                st.image(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB), caption=f"KTP {i + 1} - Warped Image", use_container_width=True)

                # Step 3: Detect and correct orientation
                st.subheader(f"Tahap 3: Deteksi dan Koreksi Orientasi untuk KTP {i + 1}")
                predicted_orientation = ktp_orientation.detect_orientation(warped_image)
                corrected_image = ktp_orientation.correct_orientation(warped_image, predicted_orientation)
                st.image(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB), caption=f"KTP {i + 1} - Corrected Orientation", use_container_width=True)

                # Step 4: Extract information
                st.subheader(f"Tahap 4: Ekstraksi Informasi untuk KTP {i + 1}")
                result = extractor.predict(Image.fromarray(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)))
                st.json(result)
                results.append(result)

            # Save results as JSON
            json_output = json.dumps(results, indent=4)
            st.download_button(
                label="Unduh Hasil JSON",
                data=json_output,
                file_name="ktp_results.json",
                mime="application/json"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
