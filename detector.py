
from ultralytics import YOLO
from ultralytics.utils import ASSETS

model = YOLO("yolo11n.pt")

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

image_path = ASSETS / "bus.jpg"

results = model.predict(
    source=str(image_path),
    classes=list(vehicle_classes.keys()),
    conf=0.25,
    save=True,
    project="outputs",
    name="traffic_analysis",
    exist_ok=True
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        vehicle_name = vehicle_classes[class_id]
        vehicle_counts[vehicle_name] += 1

total_vehicles = sum(vehicle_counts.values())

# Traffic density calculation
if total_vehicles <= 3:
    density = "LOW"
elif total_vehicles <= 7:
    density = "MEDIUM"
else:
    density = "HIGH"

print("\n========== TRAFFIC ANALYSIS ==========")

for vehicle, count in vehicle_counts.items():
    print(f"{vehicle}: {count}")

print("--------------------------------------")
print(f"Total Vehicles: {total_vehicles}")
print(f"Traffic Density: {density}")
print("======================================")
