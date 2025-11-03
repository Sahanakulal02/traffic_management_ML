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

## 📁 Project Structure

```
📦 Traffic Density Estimation
├── data/                   # Folder containing datasets (images/videos for training/testing)
│
├── output/                 # Stores output files generated after processing
│   └── processed.mp4       # Example processed video with detected traffic density
│
├── src/                    # Source code folder
│   └── demo.py             # Main Python script to run detection or model inference
│
├── venv/                   # Virtual environment folder (contains dependencies)
│
├── README.md               # Project documentation (this file)
├── requirements.txt        # List of all dependencies required to run the project
├── TODO.md                 # Notes or pending tasks for project development
│
├── yolov8n.pt              # YOLOv8-nano model weights (lightweight version)
└── yolov8s.pt              # YOLOv8-small model weights (more accurate version)
```

---

## ⚙️ Description

This project focuses on **Traffic Density Estimation** using **YOLOv8 object detection** models.  
It processes video inputs to detect vehicles and estimate the overall density (e.g., Low, Medium, High).

---

## 🚀 How to Run

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate   # On Windows
   # or
   source venv/bin/activate       # On macOS/Linux
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Demo**
   ```bash
   python src/demo.py
   ```

4. **Check Output**
   - Processed video will be saved in the `output/` folder as `processed.mp4`.

---

## 🧠 Models Used

| Model File | Description | Use Case |
|-------------|--------------|-----------|
| `yolov8n.pt` | Nano version (fast, lightweight) | Real-time inference |
| `yolov8s.pt` | Small version (more accurate) | Higher precision tasks |

---

## 👩‍💻 Author
Developed by **Sahana Kulal**  
BE Computer Science Student | Passionate about AI & Web Development  
