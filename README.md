# 🧍‍♀️ Store ROI People Counter (Ranked Analytics)

A **YOLOv8-powered people counting system** that tracks unique individuals across multiple **user-defined ROIs (Regions of Interest)** in a video feed — then ranks these ROIs in real time by **traffic density** (most-to-least visited areas).  

It provides a live display with:
- Current person count per ROI  
- Total unique visitors per ROI  
- Rank among the most and least visited regions  

This project is ideal for **retail store analytics**, **queue monitoring**, and **footfall heatmap generation**.

---

## 🎯 Key Features

✅ **Interactive ROI selection**
- Draw any number of polygonal ROIs directly on the video’s first frame.  
- Resize-friendly and intuitive interface.

✅ **Real-time tracking with YOLOv8**
- Uses object detection + tracking to follow people across frames.  
- Supports GPU or CPU inference.

✅ **Automatic ranking system**
- Each frame dynamically ranks all ROIs:
  - `#M` → Rank among most-visited  
  - `#L` → Rank among least-visited  
- Ranks update as people move between regions.

✅ **Detailed overlays**
- Live visualization showing:  
ROI_1 Cur:3 Tot:25 #M:1 #L:5

yaml
Copy code
→ meaning **currently 3 people**, **25 total unique visitors**, **ranked #1 most visited** and **#5 least visited**.

✅ **Analytics summary**
- Prints top 3 and bottom 3 ROIs every 50 frames  
- Displays final ranked summary at the end of processing

---

## 📂 Project Structure

store_roi_people_count_ranked.py # Main script
storeTraffic.mp4 # Input video (sample)
output_with_counts.mp4 # Output video with overlays
yolov8n.pt # YOLOv8 model weights

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Clone this repository

git clone https://github.com/yourusername/store-people-counter.git
cd store-people-counter
### 2️⃣ Install dependencies
Make sure you have Python 3.8+ installed, then run:


pip install ultralytics opencv-python numpy
### 3️⃣ Download YOLOv8 weights
You can use any YOLOv8 model (e.g., yolov8n.pt, yolov8m.pt, etc.):


from ultralytics import YOLO
YOLO('yolov8n.pt')  # automatically downloads
### ▶️ Usage
Step 1: Select ROIs
Run the script:


python store_roi_people_count_ranked.py
You’ll see the first frame of your video.
Controls:

#### Key	Action
Left Click	Add point (4 per ROI)
u	Undo last point
r	Remove last ROI
s	Start inference
q	Quit without processing

💡 ROIs are drawn in polygonal shapes (4 points each).
Once done, press s to start processing.

Step 2: Processing
During inference:

Bounding boxes appear around detected people

ROI overlays show live stats (Cur / Tot / #M / #L)

Press q anytime to stop processing

Output video is automatically saved as:

output_with_counts.mp4
📊 Example Console Output
sql
Copy code
== ROI Selection Instructions ==
 - Left click to add points (4 clicks per ROI)
 - 'u' undo last point, 'r' remove last ROI, 's' start inference

Processing frames (tracker used: bytetrack.yaml)...

[Frame 100] Top3 most (ROI, count): [(0, 45), (1, 38), (3, 31)]
[Frame 100] Top3 least (ROI, count): [(5, 10), (4, 15), (2, 20)]

Final per-ROI unique person counts:
ROI_1: unique_persons = 95
ROI_2: unique_persons = 72
ROI_3: unique_persons = 43
ROI_4: unique_persons = 21

Top 3 ROIs by unique person count:
  ROI_1: 95
  ROI_2: 72
  ROI_3: 43

Bottom 3 ROIs by unique person count:
  ROI_4: 21
  ROI_5: 10

#### 🧠 How It Works
1. ROI Selection
User selects polygonal areas of interest on the first frame.

2. Person Tracking
YOLOv8 detects people, while ByteTrack maintains persistent IDs across frames.

3. Counting Logic
Each ROI maintains:

A set of unique IDs seen so far

A set of current IDs (this frame)

If a person’s bottom-center point lies inside a polygon, that ROI registers them.

4. Ranking
Every frame:

ROIs are ranked by their cumulative unique counts

Rankings are displayed on-screen as #M and #L

#### ⚙️ Configuration
Edit these constants at the top of the script:

Variable	Description	Example
SOURCE	Input video path	storeTraffic.mp4
OUTPUT	Output file name	output_with_counts.mp4
MODEL	YOLOv8 weights file	yolov8n.pt
CONF	Detection confidence	0.35
DEVICE	"cpu" or "0" (GPU)	"cpu"

#### 🧩 Dependencies
Library	Purpose
Ultralytics YOLOv8	Object detection and tracking
OpenCV	ROI drawing and video processing
NumPy	Geometry and calculations

📈 Potential Use Cases
🏪 Retail store visitor analysis

🚌 Bus station or airport queue management

🏟️ Event crowd monitoring

🧾 Business intelligence dashboards

### 👨‍💻 Author
Shaikh Azan
Developed with the assistance of ChatGPT (OpenAI)

### 📜 License
This project is open-source and licensed under the MIT License.

#### 🌟 Future Improvements
Integrate heatmap visualization for traffic flow

Add real-time dashboard (Flask + WebSocket)

Incorporate line crossing metrics for entry/exit counts

Export analytics to CSV / database
