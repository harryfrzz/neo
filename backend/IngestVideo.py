import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class VideoIngestor:

    def __init__(self, model_name=None):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = None

        if model_name:
            self.load_model(model_name)

    def load_model(self, model_name="openai/clip-vit-base-patch32"):

        if self.model_name == model_name and self.model is not None:
            return

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model_name = model_name

    def unload_model(self):

        self.model = None
        self.processor = None
        self.model_name = None

    def extract_frames(self, video_path, frame_interval=30):

        cap = cv2.VideoCapture(video_path)

        frames = []
        timestamps = []
        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % frame_interval == 0:
                frames.append(frame)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

            frame_id += 1

        cap.release()

        return frames, timestamps

    def embed_frames(self, frames):

        if self.model is None:
            raise RuntimeError("No CLIP model loaded")

        embeddings = []

        for frame in frames:

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)

            embeddings.append(features.cpu().numpy()[0])

        return np.array(embeddings)

    def process_video(self, video_path, filename):

        frames, timestamps = self.extract_frames(video_path)

        embeddings = self.embed_frames(frames)

        chunks = []

        for i, ts in enumerate(timestamps):

            chunks.append({
                "doc_id": filename,
                "timestamp": ts,
                "type": "video_frame"
            })

        return chunks, embeddings