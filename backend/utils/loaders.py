import os
import pytesseract
from PIL import Image

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)

# --- CRITICAL RAILWAY FIX ---
# Remove the Windows-specific Tesseract path. 
# On Linux (Railway), the binary will be found automatically in the system PATH.
# ----------------------------

def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext in [".ppt", ".pptx"]:
        return UnstructuredPowerPointLoader(file_path).load()
    elif ext in [".png", ".jpg", ".jpeg"]:
        return load_image_with_ocr(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_image_with_ocr(image_path):
    if os.path.getsize(image_path) > 3_000_000:
        return [Document(page_content="", metadata={"source": image_path, "type": "image", "ocr_confidence": 0.0})]

    try:
        # Pre-processing for better OCR
        image = Image.open(image_path).convert("L")
        image = image.point(lambda x: 0 if x < 140 else 255, '1')

        # Tesseract will be called via the Linux system binary
        text = pytesseract.image_to_string(image, config="--psm 6")

        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        text = "".join(c for c in text if c.isprintable())
    except Exception as e:
        text = ""
        print(f"OCR failed for {image_path}: {e}")

    confidence = estimate_ocr_confidence(text)
    return [Document(page_content=text, metadata={"source": image_path, "type": "image", "ocr_confidence": confidence})]

def estimate_ocr_confidence(text):
    if not text or len(text.strip()) < 20:
        return 0.2
    elif len(text) < 100:
        return 0.5
    else:
        return 0.8
