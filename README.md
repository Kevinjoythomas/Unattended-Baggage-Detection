# Unattended Baggage Detection 🎒🚨

This project implements an **Unattended Baggage Detection System** using **YOLO (You Only Look Once)** for object detection and **Deep SORT** for multi-object tracking. The system monitors baggage and their owners in real-time, detecting instances where an owner leaves their baggage unattended for a **specified threshold distance** and **time**. When a baggage-owner separation is detected, the system **sends an alert** to mitigate potential security risks. 📢

## How It Works ⚙️
- **YOLO** identifies **baggage** and **owners** within video streams.
- **Deep SORT** assigns unique IDs to both the baggage and the owners, enabling continuous tracking across frames. 
- The system measures the **distance** between each baggage and its owner. If the distance exceeds a certain **threshold** and remains so for a specific **duration**, the baggage is flagged as **unattended**.
- Alerts can be generated and forwarded to **security personnel** or displayed on dashboards in real-time.
![](https://github.com/Kevinjoythomas/Unattended-Baggage-Detection/blob/main/img.png)
## Features ✨
- **Real-time Detection**: Processes live video feeds to detect unattended baggage.
- **Accurate Tracking**: Uses **Deep SORT** to maintain consistent tracking of baggage and owners.
- **Configurable Thresholds**: Allows customization of distance and time thresholds.
- **Alert System**: Triggers alerts when unattended baggage is detected, helping **prevent security incidents**.

## Technologies Used 🛠️
- **YOLO**: Fast and accurate object detection.
- **Deep SORT**: Robust tracking algorithm that assigns unique IDs to objects.
- **Python**: Core programming language for implementing the detection system.
- **OpenCV**: For video processing and frame handling.


