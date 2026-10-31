# 🚦 UrbanFlow AI
### AI-Powered Smart Traffic Monitoring and Analytics System

UrbanFlow AI is a student-built traffic monitoring and analytics project that uses computer vision to detect and track vehicles in uploaded traffic videos.

The system uses YOLO11 for vehicle detection, ByteTrack for tracking, OpenCV for video processing, and Streamlit for an interactive dashboard.

## ✨ Features

- Vehicle detection: identifies cars, motorcycles, buses, and trucks.
- Vehicle tracking: tracks detected vehicles across video frames.
- Traffic density estimation: classifies traffic as Low, Medium, or High.
- Traffic analytics: displays vehicle counts and category-wise statistics.
- Visual reports: generates charts and downloadable CSV reports.
- Analysis history: stores previous analysis summaries.
- Interactive dashboard: provides navigation between project sections.

## 🛠️ Technologies Used

- Python
- YOLO11
- Ultralytics
- OpenCV
- Streamlit
- Pandas
- NumPy

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/UrbanFlow-AI.git
   cd UrbanFlow-AI
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it on Windows:

   ```powershell
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

The YOLO11 model file is excluded from this repository. Ultralytics can download the required model when the application first loads it.

## 📊 Project Scope and Limitations

- This project is a student portfolio demonstration.
- Vehicle counts are approximate and depend on video quality and tracking performance.
- Traffic density thresholds are demonstration values, not calibrated road-capacity measurements.
- Average detection confidence is not the same as model accuracy.
- The application analyzes uploaded videos; it is not connected to live city traffic cameras.

## 🎯 Project Objective

The objective is to explore how computer vision and AI-based analytics can support traffic monitoring and help present useful traffic information through an interactive dashboard.

## 👩‍💻 Developer

B.Tech AI/ML Student

**Project:** UrbanFlow AI
