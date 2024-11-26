import cv2
import numpy as np
import easyocr
import pytesseract
import re
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
from ultralytics import YOLO

# MySQL 연결 설정
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="1022",
            database="test_db"
        )
        print("MySQL Database connection successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

# 한국 차량 번호판 형식을 검사하는 함수
def validate_license_plate(plate_text):
    pattern = re.compile(r'\d{2}[가-힣]\d{4}|\d{3}[가-힣]\d{4}')
    return pattern.fullmatch(plate_text) is not None

# MySQL에 번호판을 저장하는 함수
def insert_license_plate_to_db(plate_text):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        query = "INSERT INTO license_plates (plate) VALUES (%s)"
        cursor.execute(query, (plate_text,))
        connection.commit()
        print(f"License plate {plate_text} successfully inserted into the database")
    except Error as e:
        print(f"The error '{e}' occurred while inserting license plate")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# 텍스트 전처리 함수
def clean_text(text):
    return re.sub(r'[^가-힣0-9]', '', text)

# 텍스트에서 번호판 형식만 남기는 함수
def extract_plate_pattern(text):
    match = re.search(r'\d{2}[가-힣]\d{4}|\d{3}[가-힣]\d{4}', text)
    if match:
        return match.group(0)
    return None

# 이미지 전처리 함수 (밝기 및 대비 조절)
def preprocess_plate_image(plate_image):
    # 그레이스케일 변환
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    # 대비 증가
    enhanced = cv2.equalizeHist(gray)
    # 블러링을 통해 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred

# 학습된 YOLOv8 모델을 사용해 번호판 영역 감지 함수
def detect_plate_with_yolo(image, model):
    results = model(image)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_image = image[y1:y2, x1:x2]  # 감지된 번호판 영역
            return plate_image
    return None

# 이미지에서 번호판을 추출하고 EasyOCR 및 Tesseract로 인식하는 함수
def extract_license_plate(image_path, yolo_model_path):
    # 학습된 YOLOv8 모델 불러오기
    model = YOLO(yolo_model_path)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image. Please check the file path.")
        return None

    # YOLOv8 모델을 사용해 번호판 영역 감지
    plate_image = detect_plate_with_yolo(image, model)
    if plate_image is None:
        print("License plate area not detected.")
        return

    # 번호판 확대 (OCR을 위해)
    plate_image = cv2.resize(plate_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 이미지 전처리 (밝기 및 대비 조절 등)
    processed_plate_image = preprocess_plate_image(plate_image)

    # EasyOCR 객체 생성
    reader = easyocr.Reader(['ko'])

    # OCR을 반복하여 유효한 번호판을 찾을 때까지 시도
    valid_plate_text = None
    max_attempts = 5  # 최대 5번 시도
    attempt = 0

    while not valid_plate_text and attempt < max_attempts:
        attempt += 1
        print(f"OCR attempt {attempt}...")

        # OCR 수행
        results = reader.readtext(processed_plate_image)
        plate_text = ''.join([res[1] for res in results])
        cleaned_plate_text = clean_text(plate_text)
        filtered_plate_text = extract_plate_pattern(cleaned_plate_text)

        if filtered_plate_text:
            valid_plate_text = filtered_plate_text
            print(f"License Plate Detected: {valid_plate_text}")
            insert_license_plate_to_db(valid_plate_text)
        else:
            print("License plate not detected or not valid. Retrying...")

    # 번호판 추출된 부분 시각화
    if valid_plate_text:
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Cropped Plate Area - Detected: {valid_plate_text}")
        plt.show()
    else:
        print("Failed to detect a valid license plate after multiple attempts.")

if __name__ == "__main__":
    image_path = r'C:\Users\user\Desktop\assignment\DevOps_project\test.png'
    yolo_model_path = r'C:\Users\user\Desktop\assignment\DevOps_project\best.pt'  # 로컬에 저장된 YOLO 모델 경로 사용
    extract_license_plate(image_path, yolo_model_path)
