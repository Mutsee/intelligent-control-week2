import numpy as np
import pandas as pd
import cv2 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset dari file CSV
color_data = pd.read_csv('dataset/color-detection-data-set/colors.csv')
x = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Normalisasi data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split data set untuk training dan testing
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Training Model ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Inisialisasi Model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)  # Training Model KNN

# Prediksi Data Test
y_pred = knn.predict(x_test)

# Menghitung Akurasi Model
accuracy = accuracy_score(y_test,y_pred)
print(f'Akurasi Model : {accuracy*100:.2f}%')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height//2, width//2]
    pixel_center_scaled = scaler.transform([pixel_center])

    # Prediksi Warna
    color_pred = knn.predict(pixel_center_scaled)[0]
    cv2.putText(frame, f'Color:{color_pred}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()