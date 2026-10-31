import pandas as pd
import tempfile
import streamlit as st
import os
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import ASSETS


st.set_page_config(
    page_title="UrbanFlow AI",
    page_icon="🚦",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("🚦 UrbanFlow AI")

st.sidebar.write("Smart Traffic Monitoring System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Traffic Analysis",
        "Analysis History",
        "About Project"
    ]
)


st.sidebar.divider()

st.sidebar.caption("AI-Powered Traffic Analytics")

# Page Navigation
if page == "Dashboard":
    st.title("🚦 UrbanFlow AI")
    st.subheader("Smart Traffic Intelligence Dashboard")

    st.write(
        "Welcome to UrbanFlow AI — an AI-powered system "
        "for vehicle detection, tracking and traffic analytics."
    )

    st.divider()

    st.subheader("📊 Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🚗 Vehicle Detection\n\nYOLO11-powered vehicle detection and tracking.")

    with col2:
        st.info("📈 Traffic Analytics\n\nVehicle counting and traffic density classification.")

    col3, col4 = st.columns(2)

    with col3:
        st.info("📋 Analysis Reports\n\nDownload traffic analysis reports in CSV format.")

    with col4:
        st.info("🕘 Analysis History\n\nAccess previously saved traffic analysis records.")

    st.divider()

    st.subheader("🛠️ Technology Stack")

    st.write(
        "Python  |  YOLO11  |  OpenCV  |  Streamlit  |  Pandas"
    )

    st.divider()
    st.subheader("📊 Latest Traffic Analysis")

    history_file = "outputs/analysis_history.csv"

    if os.path.exists(history_file):
        dashboard_history = pd.read_csv(history_file)

        if not dashboard_history.empty:
            latest = dashboard_history.iloc[-1]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Vehicles",
                int(latest["Total Vehicles"])
            )

            col2.metric(
                "Cars",
                int(latest["Cars"])
            )

            col3.metric(
                "Motorcycles",
                int(latest["Motorcycles"])
            )

            col4, col5 = st.columns(2)

            col4.metric(
                "Buses",
                int(latest["Buses"])
            )

            col5.metric(
                "Trucks",
                int(latest["Trucks"])
            )

            st.divider()

            st.subheader("🚦 Latest Traffic Density")

            traffic_density = latest["Traffic Density"]

            if traffic_density == "HIGH":
                st.error(f"Traffic Density: {traffic_density}")

            elif traffic_density == "MEDIUM":
                st.warning(f"Traffic Density: {traffic_density}")

            else:
                st.success(f"Traffic Density: {traffic_density}")

                st.metric(
                "🤖 Average AI Detection Confidence",
                f"{float(latest['Average Confidence (%)']):.1f}%"
            )

        else:
            st.info("No analysis records available yet.")
    else:
        st.info("Run your first traffic analysis to see statistics here.")

    st.caption("UrbanFlow AI | Computer Vision & Traffic Analytics")

elif page == "Traffic Analysis":
    st.title("📹 Traffic Analysis")
    

elif page == "Analysis History":
    st.title("🕘 Analysis History")

    history_file = "outputs/analysis_history.csv"

    if os.path.exists(history_file):
        saved_history = pd.read_csv(history_file)

        st.dataframe(
            saved_history,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Analysis History",
            data=saved_history.to_csv(index=False),
            file_name="UrbanFlow_Analysis_History.csv",
            mime="text/csv"
        )
    else:
        st.info("No analysis history available yet. Run a traffic analysis first.")

elif page == "About Project":
    st.title("ℹ️ About UrbanFlow AI")

    st.write(
        "UrbanFlow AI is an AI-powered Smart Traffic "
        "Monitoring and Analytics System."
    )

    st.subheader("🎯 Project Objective")
    st.write(
        "The objective is to detect, track and analyze "
        "vehicles from traffic videos using Computer Vision."
    )

    st.subheader("🛠️ Technologies Used")

    st.markdown("""
    - Python
    - YOLO11
    - OpenCV
    - Streamlit
    - Pandas
    """)

    st.subheader("✨ Key Features")

    st.markdown("""
    - Vehicle Detection and Tracking
    - Vehicle Counting
    - Traffic Density Classification
    - Traffic Analytics Dashboard
    - Analysis History and CSV Reports
    - AI Detection Confidence
    """)

    st.subheader("📌 Project Type")
    st.write("Computer Vision | Artificial Intelligence | Machine Learning")
    # Show traffic analysis only on its selected page
if page != "Traffic Analysis":
    st.stop()

st.title("🚦 UrbanFlow AI")
st.subheader("AI-Powered Smart Traffic Intelligence System")

st.write(
    "An intelligent traffic monitoring system "
    "that detects vehicles and analyzes traffic density."
)


st.divider()
if page == "Traffic Analysis":

# Traffic Video Upload Section
 st.subheader("📹 Upload Traffic Video")

uploaded_video = st.file_uploader(
    "Choose a traffic video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:
    st.video(uploaded_video)
    st.success("Traffic video uploaded successfully!")

st.divider()

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()

if st.button("Run Traffic Analysis", type="primary"):

    vehicle_classes = {
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }

    counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0
    }

    with st.spinner("AI is analyzing traffic..."):


        
        # Select uploaded video or sample image
        if uploaded_video is not None:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_video:

                temp_video.write(uploaded_video.getbuffer())
                video_path = temp_video.name

            analysis_source = video_path

        else:
            analysis_source = str(ASSETS / "bus.jpg")

        
        # Run vehicle detection and tracking
        results = model.track(
            source=analysis_source,
            classes=list(vehicle_classes.keys()),
            conf=0.25,
            tracker="bytetrack.yaml",
            persist=True,
            stream=False,
            save=True,
            project="outputs",
            name="dashboard",
            exist_ok=True
        )



        
        # Count unique tracked vehicles
        tracked_vehicles = set()

        for result in results:
            if result.boxes.id is not None:

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    track_id = int(box.id[0])

                    vehicle_name = vehicle_classes[class_id]

                    tracked_vehicles.add(
                        (vehicle_name, track_id)
                    )

        # Update vehicle statistics
        for vehicle_name, track_id in tracked_vehicles:
            counts[vehicle_name] += 1

        total = sum(counts.values())


        total = sum(counts.values())

        if total <= 3:
            density = "LOW"
        elif total <= 7:
            density = "MEDIUM"
        else:
            density = "HIGH"

    st.success("Traffic Analysis Completed!")

    st.divider()

    st.subheader("Traffic Overview")

    col1, col2 = st.columns(2)

    col1.metric("Total Vehicles", total)
    col2.metric("Traffic Density", density)

    st.divider()

    st.subheader("Vehicle Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🚗 Cars", counts["Car"])
    c2.metric("🏍️ Motorcycles", counts["Motorcycle"])
    c3.metric("🚌 Buses", counts["Bus"])
    c4.metric("🚚 Trucks", counts["Truck"])

    
    st.divider()

    # Vehicle Analytics
    st.subheader("📊 Vehicle Analytics")

    chart_data = pd.DataFrame({
        "Vehicle": list(counts.keys()),
        "Count": list(counts.values())
    })

    chart_data = chart_data.set_index("Vehicle")

    st.write("Vehicle Type Comparison")

    st.bar_chart(
        chart_data,
        y="Count"
    )

    
    st.write("Vehicle Distribution")

    st.vega_lite_chart(
        chart_data.reset_index(),
        {
            "mark": {
                "type": "arc",
                "innerRadius": 60
            },
            "encoding": {
                "theta": {
                    "field": "Count",
                    "type": "quantitative"
                },
                "color": {
                    "field": "Vehicle",
                    "type": "nominal"
                },
                "tooltip": [
                    "Vehicle",
                    "Count"
                ]
            }
        },
        use_container_width=True
    )

        # Traffic Analysis Summary
    st.divider()
    st.subheader("📋 Traffic Analysis Summary")

    summary_data = pd.DataFrame({
        "Vehicle Type": list(counts.keys()),
        "Detected Count": list(counts.values()),
        "Share (%)": [
            round((count / total) * 100, 1) if total > 0 else 0
            for count in counts.values()
        ]
    })

    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )

        # AI Detection Confidence
    all_confidences = []

    for result in results:
        if result.boxes is not None and result.boxes.conf is not None:
            all_confidences.extend(
                result.boxes.conf.cpu().tolist()
            )
            
            average_confidence = 0.0
    if all_confidences:
        average_confidence = (
            sum(all_confidences) / len(all_confidences)
        ) * 100

        st.metric(
            "Average AI Detection Confidence",
            f"{average_confidence:.1f}%"
        )
    else:
        st.info("No vehicle detections available.")


            # Save Analysis History
    st.divider()
    st.subheader("🕘 Analysis History")

    history_file = "outputs/analysis_history.csv"

    history_record = pd.DataFrame([{
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total Vehicles": total,
        "Cars": counts["Car"],
        "Motorcycles": counts["Motorcycle"],
        "Buses": counts["Bus"],
        "Trucks": counts["Truck"],
        "Traffic Density": density,
        "Average Confidence (%)": round(average_confidence, 1)
    }])

    if os.path.exists(history_file):
        history_record.to_csv(
            history_file,
            mode="a",
            header=False,
            index=False
        )
    else:
        history_record.to_csv(
            history_file,
            index=False
        )

    st.success("Analysis history saved successfully!")

    if os.path.exists(history_file):
        saved_history = pd.read_csv(history_file)

        st.dataframe(
            saved_history,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="📥 Download Complete Analysis History",
            data=saved_history.to_csv(index=False),
            file_name="UrbanFlow_Analysis_History.csv",
            mime="text/csv"
        )

    st.divider()

    # Download Traffic Report
    st.subheader("📥 Download Traffic Report")

    report_data = pd.DataFrame({
        "Metric": [
            "Total Vehicles",
            "Cars",
            "Motorcycles",
            "Buses",
            "Trucks",
            "Traffic Density"
        ],
        "Value": [
            total,
            counts["Car"],
            counts["Motorcycle"],
            counts["Bus"],
            counts["Truck"],
            density
        ]
    })


    
    # Smart Traffic Insights
    st.divider()
    st.subheader("🧠 Smart Traffic Insights")

    if total > 0:

        dominant_vehicle = max(
            counts,
            key=counts.get
        )

        dominant_count = counts[dominant_vehicle]

        percentage = (
            dominant_count / total
        ) * 100

        st.info(
            f"{dominant_vehicle} is the most frequently "
            f"detected vehicle, representing "
            f"{percentage:.1f}% of all detected vehicles."
        )

        if density == "HIGH":
            st.warning(
                "High vehicle volume detected. "
                "Further traffic monitoring may be required."
            )

        elif density == "MEDIUM":
            st.info(
                "Moderate vehicle volume detected."
            )

        else:
            st.success(
                "Low vehicle volume detected."
            )

    else:
        st.info("No vehicles detected in this analysis.")


    csv_report = report_data.to_csv(index=False)

    st.download_button(
        label="Download Traffic Report (CSV)",
        data=csv_report,
        file_name="UrbanFlow_Traffic_Report.csv",
        mime="text/csv"
    )




    st.divider()

    st.subheader("Detection Result")

    
    st.subheader("AI Vehicle Detection Result")

    for result in results:
        annotated_image = result.plot()

        st.image(
            annotated_image,
            caption="YOLO11 Vehicle Detection",
            channels="BGR",
            use_container_width=True
        )


else:
    st.info("Click 'Run Traffic Analysis' to start AI detection.")

st.divider()

st.caption("UrbanFlow AI | Computer Vision & Traffic Analytics")