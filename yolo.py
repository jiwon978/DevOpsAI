from ultralytics import YOLO

def train_model():
    # YOLO 모델 불러오기
    model = YOLO("yolov8n.pt")

    # 모델 학습 설정
    model.train(data=r"C:\Users\user\Desktop\assignment\DevOps_project\dataset\data.yaml", epochs=50, imgsz=640, device="cuda", workers=0)


if __name__ == '__main__':
    train_model()
