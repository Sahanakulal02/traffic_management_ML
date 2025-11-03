# 🚗 Traffic Density Estimation using YOLO

This project estimates **traffic density (Low / Medium / High)** from videos using the **YOLOv8 model**.  
It detects vehicles, counts them, and classifies the traffic condition automatically.

---

## 📋 Overview
- Uses **YOLOv8** for real-time vehicle detection.  
- Classifies density levels as:
  - **Low** – fewer than 10 vehicles  
  - **Medium** – 10 to 25 vehicles  
  - **High** – more than 25 vehicles  
- Displays traffic density live on video feed.  
- Supports both **video input** and **webcam detection**.  

---

## 🧠 Model
Download the trained YOLO model from Google Drive:  
👉 [Download YOLO Model]
https://drive.google.com/file/d/1ht__9ZmwlaHfaxdm3SS0zwZmcL-5bFpo/view?usp=sharing
https://drive.google.com/file/d/1Kx-4rVwpr-F05SkjVvhC6OvDj6KRSDbX/view?usp=sharing

After downloading, place the model file inside the `models` folder.

---

## ⚙️ Steps Performed
1. Set up YOLOv8 model for vehicle detection.  
2. Preprocessed traffic videos for testing.  
3. Defined threshold-based rules for density classification.  
4. Displayed real-time traffic density results on screen.  
5. Enabled both **video file** and **live webcam** input options.  
6. Verified results and tuned thresholds for accuracy.  

---

## 🧩 Future Enhancements
- Add a CNN model for comparison.  
- Generate a confusion matrix and accuracy metrics.  
- Integrate live camera feeds for continuous monitoring.  

---

## 👩‍💻 Author
Developed by **Sahana Kulal**  
BE Computer Science Student | Passionate about AI & Web Development  
