import os

# Reduce TensorFlow console messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt

from deepface import DeepFace

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FaceSense: Emotion-Aware Greeting System",
    page_icon="🙂",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🙂 FaceSense")

st.caption(
    "Emotion-Aware Greeting System — upload a short video "
    "and get a greeting matched to your expression."
)


# ============================================================
# EMOTION GREETINGS
# ============================================================

GREETINGS = {
    "happy": "Hey there! Your smile is contagious — great to see you!",
    "sad": "Hi... I can see things feel heavy right now. I'm here with you.",
    "angry": "Hello. Let's take a breath together — I'm here to help, no rush.",
    "surprise": "Whoa, welcome! Something exciting going on?",
    "fear": "Hi, it's okay — you're safe here. Let's take it one step at a time.",
    "disgust": "Hello there — let's see how I can turn things around for you.",
    "neutral": "Hi! Good to have you here."
}

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


# ============================================================
# FRAME PREPROCESSING
# ============================================================

def enhance_lighting(frame):
    """
    Improve image brightness using YCrCb color space.
    """

    ycrcb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2YCrCb
    )

    y, cr, cb = cv2.split(ycrcb)

    y_eq = cv2.equalizeHist(y)

    ycrcb_eq = cv2.merge(
        [y_eq, cr, cb]
    )

    enhanced = cv2.cvtColor(
        ycrcb_eq,
        cv2.COLOR_YCrCb2BGR
    )

    return enhanced


def denoise(frame):
    """
    Remove image noise.
    """

    return cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        h=10,
        hColor=10
    )


def resize_frame(frame, width=640):
    """
    Resize frame while maintaining aspect ratio.
    """

    h, w = frame.shape[:2]

    if w == 0:
        return frame

    scale = width / w

    new_height = int(h * scale)

    return cv2.resize(
        frame,
        (width, new_height)
    )


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_frames(
    video_path,
    sample_rate=3,
    max_frames=15
):
    """
    Extract selected frames from a video.
    """

    cap = cv2.VideoCapture(video_path)

    frames = []

    count = 0

    while len(frames) < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        if count % sample_rate == 0:
            frames.append(frame)

        count += 1

    cap.release()

    return frames


# ============================================================
# DEEPFACE ANALYSIS
# ============================================================

def analyze_video_per_frame(
    video_path,
    detector_backend="opencv",
    max_frames=15
):
    """
    Analyze emotion for each sampled video frame.
    """

    frames = extract_frames(
        video_path,
        max_frames=max_frames
    )

    all_scores = []

    errors = []

    for i, frame in enumerate(frames):

        try:

            # Improve lighting
            processed = enhance_lighting(frame)

            # Resize frame
            processed = resize_frame(processed)

            # DeepFace emotion analysis
            result = DeepFace.analyze(
                processed,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=detector_backend
            )

            # DeepFace can return either a list or dictionary
            if isinstance(result, list):
                emotion_result = result[0]
            else:
                emotion_result = result

            all_scores.append(
                emotion_result["emotion"]
            )

        except Exception as e:

            errors.append(
                f"Frame {i}: "
                f"{type(e).__name__}: {e}"
            )

    return (
        all_scores,
        len(frames),
        errors
    )


# ============================================================
# CACHED ANALYSIS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def run_analysis_cached(
    video_bytes,
    detector_backend,
    max_frames
):
    """
    Cache analysis so the same video does not need
    to be analyzed again unnecessarily.
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as tmp:

        tmp.write(video_bytes)

        video_path = tmp.name

    try:

        return analyze_video_per_frame(
            video_path,
            detector_backend,
            max_frames
        )

    finally:

        if os.path.exists(video_path):
            os.unlink(video_path)


# ============================================================
# AVERAGE EMOTION SCORES
# ============================================================

def average_scores(all_emotion_scores):

    if not all_emotion_scores:
        return None, {}

    # Use frames with reasonable confidence
    confident_scores = [
        frame_scores
        for frame_scores in all_emotion_scores
        if max(frame_scores.values()) > 30
    ]

    # If no confident frames exist,
    # use all frames
    scores_to_average = (
        confident_scores
        if confident_scores
        else all_emotion_scores
    )

    emotion_keys = scores_to_average[0].keys()

    averaged = {
        emotion: float(
            np.mean(
                [
                    scores[emotion]
                    for scores in scores_to_average
                ]
            )
        )
        for emotion in emotion_keys
    }

    overall = max(
        averaged,
        key=averaged.get
    )

    return overall, averaged


# ============================================================
# USER INTERFACE
# ============================================================

st.divider()

col_upload, col_backend = st.columns(
    [2, 1]
)


# ------------------------------------------------------------
# VIDEO UPLOAD
# ------------------------------------------------------------

with col_upload:

    uploaded_file = st.file_uploader(
        "Upload a short video",
        type=[
            "mp4",
            "mov",
            "avi"
        ],
        help="Upload a short video containing a clear face."
    )


# ------------------------------------------------------------
# DETECTOR BACKEND
# ------------------------------------------------------------

with col_backend:

    detector_backend = st.selectbox(
        "Detector backend",
        [
            "opencv",
            "mtcnn",
            "retinaface"
        ],
        index=0,
        help=(
            "OpenCV is the fastest option. "
            "MTCNN and RetinaFace can be more robust "
            "but may be slower."
        )
    )


# ------------------------------------------------------------
# NUMBER OF FRAMES
# ------------------------------------------------------------

max_frames = st.slider(
    "Frames to analyze",
    min_value=3,
    max_value=15,
    value=6,
    step=1,
    help=(
        "Fewer frames give faster results. "
        "5–8 frames are usually enough."
    )
)


# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

true_label = st.selectbox(
    "Optional: emotion you actually performed",
    ["-- skip --"] + EMOTIONS,
    index=0
)


# ============================================================
# VIDEO ANALYSIS
# ============================================================

if uploaded_file is not None:

    video_bytes = uploaded_file.getvalue()

    # Show video
    st.video(video_bytes)

    st.divider()

    # --------------------------------------------------------
    # ANALYZE VIDEO
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing facial expression..."
    ):

        (
            all_scores,
            num_frames_extracted,
            errors
        ) = run_analysis_cached(
            video_bytes,
            detector_backend,
            max_frames
        )

    # ========================================================
    # NO RESULT
    # ========================================================

    if not all_scores:

        st.error(
            "No face detected in the video."
        )

        st.info(
            "Try a clearer, well-lit, front-facing video "
            "with the face clearly visible."
        )

        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        with st.expander(
            "🔧 Debug information",
            expanded=True
        ):

            st.write(
                f"Frames extracted: "
                f"{num_frames_extracted}"
            )

            if num_frames_extracted == 0:

                st.warning(
                    "OpenCV could not read any frames "
                    "from the video."
                )

                st.write(
                    "Try re-exporting your video as "
                    "an H.264 MP4 file."
                )

            elif errors:

                st.write(
                    "Errors encountered during "
                    "DeepFace analysis:"
                )

                for err in errors:

                    st.text(err)

            else:

                st.write(
                    "The video frames were read successfully, "
                    "but no usable face/emotion result was obtained."
                )


    # ========================================================
    # SUCCESSFUL ANALYSIS
    # ========================================================

    else:

        # ----------------------------------------------------
        # OVERALL EMOTION
        # ----------------------------------------------------

        overall_emotion, scores = average_scores(
            all_scores
        )

        # Dominant emotion for every frame
        dominant_emotions_per_frame = [
            max(
                frame_scores,
                key=frame_scores.get
            )
            for frame_scores in all_scores
        ]


        # ====================================================
        # GREETING
        # ====================================================

        st.success(
            f"Detected emotion: **{overall_emotion}**"
        )

        st.subheader("👋 Personalized Greeting")

        st.write(
            GREETINGS.get(
                overall_emotion,
                "Hello! Welcome."
            )
        )


        # ====================================================
        # CONFIDENCE SCORES
        # ====================================================

        with st.expander(
            "📊 See confidence scores"
        ):

            sorted_scores = sorted(
                scores.items(),
                key=lambda x: -x[1]
            )

            for emotion, score in sorted_scores:

                st.write(
                    f"**{emotion.capitalize()}**: "
                    f"{score:.2f}%"
                )


        # ====================================================
        # FRAME INFORMATION
        # ====================================================

        st.subheader(
            "🎞️ Frame Analysis"
        )

        st.write(
            f"Frames extracted: "
            f"**{num_frames_extracted}**"
        )

        st.write(
            f"Frames successfully analyzed: "
            f"**{len(all_scores)}**"
        )


        # ====================================================
        # PER-FRAME EMOTION TRACE
        # ====================================================

        st.subheader(
            "📈 Per-frame emotion trace"
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        # Convert emotions to numbers
        emotion_to_number = {
            emotion: index
            for index, emotion
            in enumerate(EMOTIONS)
        }

        y_values = [
            emotion_to_number[
                emotion
            ]
            for emotion
            in dominant_emotions_per_frame
        ]

        x_values = range(
            len(dominant_emotions_per_frame)
        )

        ax.plot(
            x_values,
            y_values,
            marker="o"
        )

        ax.set_title(
            "Dominant Emotion Across Sampled Frames"
        )

        ax.set_xlabel(
            "Sampled Frame Index"
        )

        ax.set_ylabel(
            "Detected Emotion"
        )

        ax.set_yticks(
            range(len(EMOTIONS))
        )

        ax.set_yticklabels(
            EMOTIONS
        )

        ax.set_xticks(
            list(x_values)
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # PER-FRAME RESULTS TABLE
        # ====================================================

        st.subheader(
            "🎯 Per-frame predictions"
        )

        for index, emotion in enumerate(
            dominant_emotions_per_frame
        ):

            st.write(
                f"Frame {index + 1}: "
                f"**{emotion}**"
            )


        # ====================================================
        # OPTIONAL ACCURACY EVALUATION
        # ====================================================

        if true_label != "-- skip --":

            st.divider()

            st.subheader(
                "📊 Overall prediction vs ground truth"
            )

            # ------------------------------------------------
            # Overall comparison
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Predicted",
                overall_emotion
            )

            c2.metric(
                "Ground truth",
                true_label
            )

            c3.metric(
                "Match",
                (
                    "✅ Yes"
                    if overall_emotion == true_label
                    else "❌ No"
                )
            )


            # ------------------------------------------------
            # Prepare labels
            # ------------------------------------------------

            predicted_labels = (
                dominant_emotions_per_frame
            )

            true_labels = [
                true_label
            ] * len(predicted_labels)


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            accuracy = accuracy_score(
                true_labels,
                predicted_labels
            )

            f1 = f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0
            )

            precision = precision_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0
            )


            # ------------------------------------------------
            # Display metrics
            # ------------------------------------------------

            st.subheader(
                "📌 Classification Metrics"
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Accuracy",
                f"{accuracy:.2f}"
            )

            m2.metric(
                "F1 Score",
                f"{f1:.2f}"
            )

            m3.metric(
                "Recall",
                f"{recall:.2f}"
            )

            m4.metric(
                "Precision",
                f"{precision:.2f}"
            )


            # ------------------------------------------------
            # Classification report
            # ------------------------------------------------

            with st.expander(
                "📄 Full classification report"
            ):

                report = classification_report(
                    true_labels,
                    predicted_labels,
                    zero_division=0
                )

                st.text(report)


# ============================================================
# NO VIDEO UPLOADED
# ============================================================

else:

    st.info(
        "👆 Upload a video above to get started."
    )

    st.markdown(
        """
        ### How to use FaceSense

        1. Upload a short video.
        2. Select the face detector.
        3. Choose the number of frames.
        4. FaceSense analyzes the facial expressions.
        5. The dominant emotion is identified.
        6. A personalized greeting is displayed.
        7. Optionally select the actual emotion to calculate
           Accuracy, Precision, Recall and F1 Score.
        """
    )


# # import streamlit as st
# # import cv2
# # import numpy as np
# # import tempfile
# # import os
# # import matplotlib.pyplot as plt
# # from deepface import DeepFace
# # from sklearn.metrics import (
# #     accuracy_score, f1_score, recall_score, precision_score, classification_report
# # )

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# import matplotlib.pyplot as plt
# from deepface import DeepFace

# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     recall_score,
#     precision_score,
#     classification_report
# )


# st.set_page_config(page_title="FaceSense: Emotion-Aware Greeting System", page_icon="🙂")

# st.title("🙂 FaceSense")
# st.caption("Emotion-Aware Greeting System — upload a short video and get a greeting matched to your expression.")

# GREETINGS = {
#     "happy":    "Hey there! Your smile is contagious — great to see you!",
#     "sad":      "Hi... I can see things feel heavy right now. I'm here with you.",
#     "angry":    "Hello. Let's take a breath together — I'm here to help, no rush.",
#     "surprise": "Whoa, welcome! Something exciting going on?",
#     "fear":     "Hi, it's okay — you're safe here. Let's take it one step at a time.",
#     "disgust":  "Hello there — let's see how I can turn things around for you.",
#     "neutral":  "Hi! Good to have you here."
# }

# EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# # ---------- Frame preprocessing helpers ----------

# def enhance_lighting(frame):
#     ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#     y, cr, cb = cv2.split(ycrcb)
#     y_eq = cv2.equalizeHist(y)
#     ycrcb_eq = cv2.merge([y_eq, cr, cb])
#     return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


# def denoise(frame):
#     return cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10)


# def resize_frame(frame, width=640):
#     h, w = frame.shape[:2]
#     scale = width / w
#     return cv2.resize(frame, (width, int(h * scale)))


# def extract_frames(video_path, sample_rate=3, max_frames=15):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     count = 0
#     while len(frames) < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % sample_rate == 0:
#             frames.append(frame)
#         count += 1
#     cap.release()
#     return frames


# # ---------- Core analysis ----------

# def analyze_video_per_frame(video_path, detector_backend="mtcnn", max_frames=15):
#     frames = extract_frames(video_path, max_frames=max_frames)
#     all_scores = []
#     errors = []
#     for i, frame in enumerate(frames):
#         try:
#             processed = enhance_lighting(resize_frame(frame))
#             result = DeepFace.analyze(
#                 processed,
#                 actions=['emotion'],
#                 enforce_detection=False,
#                 detector_backend=detector_backend
#             )
#             all_scores.append(result[0]['emotion'])
#         except Exception as e:
#             errors.append(f"Frame {i}: {type(e).__name__}: {e}")
#     return all_scores, len(frames), errors


# @st.cache_data(show_spinner=False)
# def run_analysis_cached(video_bytes, detector_backend, max_frames):
#     """Cached so re-running the same video/settings is instant, and unrelated
#     widget changes (like the ground-truth dropdown) don't re-trigger DeepFace."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(video_bytes)
#         video_path = tmp.name
#     try:
#         return analyze_video_per_frame(video_path, detector_backend, max_frames)
#     finally:
#         os.unlink(video_path)


# def average_scores(all_emotion_scores):
#     confident_scores = [fs for fs in all_emotion_scores if max(fs.values()) > 30]
#     scores_to_average = confident_scores if confident_scores else all_emotion_scores
#     emotion_keys = scores_to_average[0].keys()
#     averaged = {
#         emotion: float(np.mean([fs[emotion] for fs in scores_to_average]))
#         for emotion in emotion_keys
#     }
#     overall = max(averaged, key=averaged.get)
#     return overall, averaged


# @st.cache_resource
# def warm_up_model():
#     dummy = np.zeros((100, 100, 3), dtype=np.uint8)
#     try:
#         DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')
#     except Exception:
#         pass
#     return True


# warm_up_model()

# # ---------- Single-page flow ----------

# col_upload, col_backend = st.columns([2, 1])
# with col_upload:
#     uploaded_file = st.file_uploader("Upload a short video (mp4, mov)", type=["mp4", "mov", "avi"])
# with col_backend:
#     detector_backend = st.selectbox(
#         "Detector backend", ["mtcnn", "retinaface", "opencv"], index=0,
#         help="Speed: opencv (fastest, needs haarcascade file) > mtcnn > retinaface (slowest, most robust)."
#     )

# max_frames = st.slider(
#     "Frames to analyze", min_value=3, max_value=15, value=6, step=1,
#     help="Fewer frames = faster results. 5-8 is usually enough for a stable average."
# )

# true_label = st.selectbox(
#     "Optional: emotion you actually performed (to check accuracy)",
#     ["-- skip --"] + EMOTIONS,
#     index=0
# )

# if uploaded_file is not None:
#     video_bytes = uploaded_file.getvalue()
#     st.video(uploaded_file)

#     with st.spinner("Analyzing expression..."):
#         all_scores, num_frames_extracted, errors = run_analysis_cached(
#             video_bytes, detector_backend, max_frames
#         )

#     if not all_scores:
#         st.error("No face detected in the video. Try a clearer, well-lit, front-facing clip.")
#         with st.expander("Debug info", expanded=True):
#             st.write(f"Frames extracted from video: {num_frames_extracted}")
#             if num_frames_extracted == 0:
#                 st.write(
#                     "0 frames were read from the file — OpenCV likely can't decode this "
#                     "video's codec on this machine. Try re-exporting as H.264 mp4."
#                 )
#             elif errors:
#                 st.write("DeepFace raised an error on every extracted frame, or found no face "
#                           "with confidence in any of them:")
#                 for err in errors:
#                     st.text(err)
#                 st.write(
#                     "If there are no errors listed above but this still shows 'no face detected', "
#                     "the detector genuinely didn't find a face in any sampled frame — try a video "
#                     "with a single, well-lit, front-facing, unobstructed face filling more of the frame."
#                 )
#     else:
#         overall_emotion, scores = average_scores(all_scores)
#         dominant_emotions_per_frame = [max(fs, key=fs.get) for fs in all_scores]

#         # --- Greeting ---
#         st.success(f"Detected emotion: **{overall_emotion}**")
#         st.write(GREETINGS.get(overall_emotion, "Hello! Welcome."))

#         with st.expander("See confidence scores"):
#             for emotion, score in sorted(scores.items(), key=lambda x: -x[1]):
#                 st.write(f"{emotion}: {score:.2f}%")

#         # --- Chart: always shown, no ground truth needed ---
#         st.subheader("Per-frame emotion trace")
#         fig, ax = plt.subplots(figsize=(10, 4))
#         ax.plot(dominant_emotions_per_frame, marker='o')
#         ax.set_title("Dominant emotion across sampled frames")
#         ax.set_xlabel("Sampled frame index")
#         ax.set_ylabel("Detected emotion")
#         plt.setp(ax.get_xticklabels(), rotation=45)
#         fig.tight_layout()
#         st.pyplot(fig)

#         # --- Accuracy metrics: only if user picked a ground-truth label ---
#         if true_label != "-- skip --":
#             st.subheader("Overall prediction vs. ground truth")
#             c1, c2, c3 = st.columns(3)
#             c1.metric("Predicted (averaged)", overall_emotion)
#             c2.metric("Ground truth", true_label)
#             c3.metric("Match", "✅ Yes" if overall_emotion == true_label else "❌ No")

#             st.subheader("Per-frame classification metrics")
#             predicted_labels = dominant_emotions_per_frame
#             true_labels = [true_label] * len(predicted_labels)

#             accuracy = accuracy_score(true_labels, predicted_labels)
#             f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
#             recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
#             precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)

#             m1, m2, m3, m4 = st.columns(4)
#             m1.metric("Accuracy", f"{accuracy:.2f}")
#             m2.metric("F1 (weighted)", f"{f1:.2f}")
#             m3.metric("Recall (weighted)", f"{recall:.2f}")
#             m4.metric("Precision (weighted)", f"{precision:.2f}")

#             with st.expander("Full classification report"):
#                 st.text(classification_report(true_labels, predicted_labels, zero_division=0))
# else:
#     st.info("Upload a video above to get started.")



# # import streamlit as st
# # import cv2
# # import numpy as np
# # import tempfile
# # import os
# # import matplotlib.pyplot as plt
# # from deepface import DeepFace
# # from sklearn.metrics import (
# #     accuracy_score, f1_score, recall_score, precision_score, classification_report
# # )

# # st.set_page_config(page_title="FaceSense: Emotion-Aware Greeting System", page_icon="🙂")

# # st.title("🙂 FaceSense")
# # st.caption("Emotion-Aware Greeting System — upload a short video and get a greeting matched to your expression.")

# # GREETINGS = {
# #     "happy":    "Hey there! Your smile is contagious — great to see you!",
# #     "sad":      "Hi... I can see things feel heavy right now. I'm here with you.",
# #     "angry":    "Hello. Let's take a breath together — I'm here to help, no rush.",
# #     "surprise": "Whoa, welcome! Something exciting going on?",
# #     "fear":     "Hi, it's okay — you're safe here. Let's take it one step at a time.",
# #     "disgust":  "Hello there — let's see how I can turn things around for you.",
# #     "neutral":  "Hi! Good to have you here."
# # }

# # EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# # # ---------- Frame preprocessing helpers ----------

# # def enhance_lighting(frame):
# #     ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
# #     y, cr, cb = cv2.split(ycrcb)
# #     y_eq = cv2.equalizeHist(y)
# #     ycrcb_eq = cv2.merge([y_eq, cr, cb])
# #     return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


# # def denoise(frame):
# #     return cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10)


# # def resize_frame(frame, width=640):
# #     h, w = frame.shape[:2]
# #     scale = width / w
# #     return cv2.resize(frame, (width, int(h * scale)))


# # def extract_frames(video_path, sample_rate=3, max_frames=15):
# #     cap = cv2.VideoCapture(video_path)
# #     frames = []
# #     count = 0
# #     while len(frames) < max_frames:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         if count % sample_rate == 0:
# #             frames.append(frame)
# #         count += 1
# #     cap.release()
# #     return frames


# # # ---------- Core analysis: returns per-frame scores so both tabs can reuse it ----------

# # def analyze_video_per_frame(video_path, detector_backend="opencv"):
# #     frames = extract_frames(video_path)
# #     all_scores = []
# #     errors = []
# #     for i, frame in enumerate(frames):
# #         try:
# #             processed = enhance_lighting(resize_frame(frame))
# #             result = DeepFace.analyze(
# #                 processed,
# #                 actions=['emotion'],
# #                 enforce_detection=False,
# #                 detector_backend=detector_backend
# #             )
# #             all_scores.append(result[0]['emotion'])
# #         except Exception as e:
# #             errors.append(f"Frame {i}: {type(e).__name__}: {e}")
# #     return all_scores, len(frames), errors


# # def average_scores(all_emotion_scores):
# #     confident_scores = [fs for fs in all_emotion_scores if max(fs.values()) > 30]
# #     scores_to_average = confident_scores if confident_scores else all_emotion_scores
# #     emotion_keys = scores_to_average[0].keys()
# #     averaged = {
# #         emotion: float(np.mean([fs[emotion] for fs in scores_to_average]))
# #         for emotion in emotion_keys
# #     }
# #     overall = max(averaged, key=averaged.get)
# #     return overall, averaged


# # @st.cache_resource
# # def warm_up_model():
# #     dummy = np.zeros((100, 100, 3), dtype=np.uint8)
# #     try:
# #         DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
# #     except Exception:
# #         pass
# #     return True


# # warm_up_model()

# # tab_greeting, tab_eval = st.tabs(["👋 Greeting", "📊 Evaluation"])


# # # ---------- Tab 1: end-user greeting flow ----------

# # with tab_greeting:
# #     uploaded_file = st.file_uploader(
# #         "Upload a short video (mp4, mov)", type=["mp4", "mov", "avi"], key="greeting_upload"
# #     )

# #     if uploaded_file is not None:
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
# #             tmp.write(uploaded_file.read())
# #             video_path = tmp.name

# #         st.video(uploaded_file)

# #         with st.spinner("Analyzing expression..."):
# #             all_scores, num_frames_extracted, errors = analyze_video_per_frame(
# #                 video_path, detector_backend="mtcnn"
# #             )

# #         os.unlink(video_path)

# #         if not all_scores:
# #             st.error("No face detected in the video. Try a clearer, well-lit, front-facing clip.")
# #             with st.expander("Debug info"):
# #                 st.write(f"Frames extracted from video: {num_frames_extracted}")
# #                 if num_frames_extracted == 0:
# #                     st.write(
# #                         "0 frames were read from the file — OpenCV likely can't decode this "
# #                         "video's codec on this machine. Try re-exporting as H.264 mp4, or "
# #                         "`pip install opencv-python-headless` in place of opencv-python."
# #                     )
# #                 elif errors:
# #                     st.write("DeepFace raised an error on every extracted frame:")
# #                     for err in errors:
# #                         st.text(err)
# #         else:
# #             overall_emotion, scores = average_scores(all_scores)
# #             st.success(f"Detected emotion: **{overall_emotion}**")
# #             st.write(GREETINGS.get(overall_emotion, "Hello! Welcome."))

# #             with st.expander("See confidence scores"):
# #                 for emotion, score in sorted(scores.items(), key=lambda x: -x[1]):
# #                     st.write(f"{emotion}: {score:.2f}%")
# #     else:
# #         st.info("Upload a video above to get started.")


# # # ---------- Tab 2: chart + accuracy metrics against a known ground-truth emotion ----------

# # with tab_eval:
# #     st.caption(
# #         "Upload a test clip where you held ONE expression the whole time, tell us what that "
# #         "expression was, and this tab charts the per-frame predictions and scores accuracy "
# #         "against that ground truth."
# #     )

# #     col1, col2 = st.columns(2)
# #     with col1:
# #         eval_file = st.file_uploader(
# #             "Upload a test video (mp4, mov, avi)", type=["mp4", "mov", "avi"], key="eval_upload"
# #         )
# #     with col2:
# #         true_label = st.selectbox("Emotion you actually performed in this clip", EMOTIONS, index=6)

# #     detector_backend = st.selectbox(
# #         "Detector backend to evaluate",
# #         ["mtcnn", "opencv", "retinaface"],
# #         index=0,
# #         help="Swap this to compare detector backends against the same clip."
# #     )

# #     run = st.button("Run evaluation", type="primary", disabled=eval_file is None)

# #     if run and eval_file is not None:
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
# #             tmp.write(eval_file.read())
# #             eval_video_path = tmp.name

# #         with st.spinner("Extracting frames and running the model..."):
# #             all_emotion_scores, num_frames_extracted, errors = analyze_video_per_frame(
# #                 eval_video_path, detector_backend
# #             )

# #         os.unlink(eval_video_path)

# #         if not all_emotion_scores:
# #             st.error("No face detected in any sampled frame. Try a clearer, well-lit, front-facing clip.")
# #             with st.expander("Debug info"):
# #                 st.write(f"Frames extracted from video: {num_frames_extracted}")
# #                 if errors:
# #                     st.write("DeepFace raised an error on every extracted frame:")
# #                     for err in errors:
# #                         st.text(err)
# #         else:
# #             dominant_emotions_per_frame = [
# #                 max(fs, key=fs.get) for fs in all_emotion_scores
# #             ]
# #             overall_emotion, averaged_scores = average_scores(all_emotion_scores)

# #             st.subheader("Per-frame emotion trace")
# #             fig, ax = plt.subplots(figsize=(10, 4))
# #             ax.plot(dominant_emotions_per_frame, marker='o')
# #             ax.set_title("Dominant emotion across sampled frames")
# #             ax.set_xlabel("Sampled frame index")
# #             ax.set_ylabel("Detected emotion")
# #             plt.setp(ax.get_xticklabels(), rotation=45)
# #             fig.tight_layout()
# #             st.pyplot(fig)

# #             st.subheader("Overall prediction vs. ground truth")
# #             c1, c2, c3 = st.columns(3)
# #             c1.metric("Predicted (averaged)", overall_emotion)
# #             c2.metric("Ground truth", true_label)
# #             c3.metric("Match", "✅ Yes" if overall_emotion == true_label else "❌ No")

# #             st.subheader("Confidence scores (averaged)")
# #             for emotion, score in sorted(averaged_scores.items(), key=lambda x: -x[1]):
# #                 st.write(f"{emotion}: {score:.2f}%")

# #             st.subheader("Per-frame classification metrics")
# #             predicted_labels = dominant_emotions_per_frame
# #             true_labels = [true_label] * len(predicted_labels)

# #             accuracy = accuracy_score(true_labels, predicted_labels)
# #             f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
# #             recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
# #             precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)

# #             m1, m2, m3, m4 = st.columns(4)
# #             m1.metric("Accuracy", f"{accuracy:.2f}")
# #             m2.metric("F1 (weighted)", f"{f1:.2f}")
# #             m3.metric("Recall (weighted)", f"{recall:.2f}")
# #             m4.metric("Precision (weighted)", f"{precision:.2f}")

# #             with st.expander("Full classification report"):
# #                 st.text(classification_report(true_labels, predicted_labels, zero_division=0))
# #     elif not run:
# #         st.info("Upload a clip, pick the emotion you actually performed, then run the evaluation.")