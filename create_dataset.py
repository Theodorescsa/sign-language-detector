import mediapipe as mp
import cv2
import os
import pickle

# Khởi tạo Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'

data_84 = []  # Lưu dữ liệu cho ảnh có 2 bàn tay (84 điểm)
data_42 = []  # Lưu dữ liệu cho ảnh có 1 bàn tay (42 điểm)
labels_84 = []  # Nhãn cho dữ liệu có 2 bàn tay
labels_42 = []  # Nhãn cho dữ liệu có 1 bàn tay

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):  # Lặp qua mỗi ảnh
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        # Kiểm tra xem có ít nhất một bàn tay được phát hiện không
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Lặp qua các điểm landmark của mỗi bàn tay
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            if len(data_aux) == 84:  # Nếu có 2 bàn tay (84 giá trị)
                data_84.append(data_aux)
                labels_84.append(dir_)
            elif len(data_aux) == 42:  # Nếu có 1 bàn tay (42 giá trị)
                data_42.append(data_aux)
                labels_42.append(dir_)
            else:
                print(f"Cảnh báo: Số điểm landmark không chính xác cho ảnh {img_path}")

print(f"Tổng số mẫu dữ liệu với 1 bàn tay: {len(data_42)}")
print(f"Tổng số mẫu dữ liệu với 2 bàn tay: {len(data_84)}")

def load_existing_data(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    return {'data': [], 'labels': []}

# Append new data and labels to the existing structure
def append_data_to_file(file_name, new_data, new_labels):
    existing_data = load_existing_data(file_name)
    
    # Append new data
    existing_data['data'].extend(new_data)
    existing_data['labels'].extend(new_labels)
    
    # Save back to the file
    with open(file_name, 'wb') as f:
        pickle.dump(existing_data, f)

# Example usage
append_data_to_file("data_42.pickle", data_42, labels_42)
append_data_to_file("data_84.pickle", data_84, labels_84)
# # Lưu dữ liệu vào file pickle
# with open("data_42.pickle", 'wb') as f:
#     pickle.dump({'data': data_42, 'labels': labels_42}, f)

# with open("data_84.pickle", 'wb') as f:
#     pickle.dump({'data': data_84, 'labels': labels_84}, f)
