import pytesseract
from PIL import Image
from typing import Optional
from concurrent.futures import Future
import ollama

from shared_executor import executor


class DocumentIngestor:
    def __init__(self, ocr_model: str = ""):
        self.ocr_model = ocr_model

    def ocr_async(self, image_path: str, ocr_model: Optional[str] = None) -> Future:
        model = ocr_model or self.ocr_model

        def _call():
            if model:
                try:
                    response = ollama.generate(
                        model=model,
                        prompt="Extract all text from this image exactly as it appears. "
                               "Do not add any additional comments or formatting.",
                        images=[image_path],
                        keep_alive=-1,
                    )
                    text = response.get("response", "") if isinstance(response, dict) else getattr(response, "response", "")
                    if text.strip():
                        return text
                    print(f"[OCR] '{model}' returned {len(text)} chars (whitespace only), falling back to pytesseract.")
                except Exception as e:
                    print(f"[OCR] Ollama OCR failed ({e}), falling back to pytesseract.")
            return self._pytesseract_ocr(image_path)

        return executor.submit(_call)

    def perform_ocr(self, image_path: str, ocr_model: Optional[str] = None) -> str:
        return self.ocr_async(image_path, ocr_model=ocr_model).result()

    def _pytesseract_ocr(self, image_path: str) -> str:
        try:
            image = Image.open(image_path)
            return pytesseract.image_to_string(image)
        except Exception as e:
            print(f"[OCR] PyTesseract OCR failed: {e}")
            return ""
