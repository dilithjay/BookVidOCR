import pytesseract
import numpy as np
import cv2
import fastwer


def perform_ocr(img: np.ndarray, preprocessed_img_save_path: str = None) -> str:
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold image
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    if preprocessed_img_save_path:
        cv2.imwrite(preprocessed_img_save_path, img)

    return pytesseract.image_to_string(img)


def eval_result(ocr_result: str, ground_truth: str):
    return fastwer.score(ocr_result.replace('—', ' ').split(), ground_truth.replace('—', ' ').split(), char_level=True)
