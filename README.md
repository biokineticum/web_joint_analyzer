🦴 Joint Motion Analyzer - Web Version

Real-time joint motion analysis using MediaPipe and computer vision

Perfect for rehabilitation, sports analysis, research, and biomechanics studies

✨ Features

🎯 Core Capabilities
📹 Video Analysis: Upload and analyze motion videos in real-time
🖼️ Image Analysis: Single image pose detection and joint angle measurement
🦴 Multi-joint Tracking: Comprehensive analysis of 6 major joint groups
📊 Real-time Visualization: Live charts and statistics during processing
💾 Data Export: CSV data and processed video downloads
🎨 Professional UI: Clean, responsive web interface
🦴 Supported Joints


Shoulders (Left/Right): Elbow-Shoulder-Hip angle
Elbows (Left/Right): Wrist-Elbow-Shoulder angle
Hips (Left/Right): Knee-Hip-Shoulder angle
Knees (Left/Right): Ankle-Knee-Hip angle
Wrists (Left/Right): Pinky-Wrist-Elbow angle
Ankles (Left/Right): Foot-Ankle-Knee angle

📈 Analysis Output

Joint Angles: Frame-by-frame angle measurements
Angular Velocities: Rate of change calculations
Statistical Summary: Min, max, average, and range for each joint
Interactive Charts: Plotly-powered visualizations
Processed Video: Annotated video with pose landmarks and angles
🚀 Quick Start

📋 Prerequisites
Python 3.8 or higher
Web browser (Chrome, Firefox, Safari, Edge)
Video files: MP4, AVI, MOV, MKV
Image files: JPG, JPEG, PNG

⚡ Installation
Clone the repository (if public) or download the files:

git clone https://github.com/yourusername/joint-motion-analyzer-web.git
cd joint-motion-analyzer-web

Install dependencies:

pip install -r requirements.txt
Run the application:
streamlit run app.py
Open your browser to http://localhost:8501
📦 Dependencies

streamlit==1.28.1
opencv-python-headless==4.8.1.78
mediapipe==0.10.8
numpy==1.24.3
pandas==2.0.3
plotly==5.17.0
Pillow==10.0.1

🎯 How to Use

📹 Video Analysis
Select Analysis Mode: Choose "📹 Video Upload" from the sidebar
Choose Joints: Select which joints to analyze (left/right for each joint type)
Upload Video: Drag and drop or browse for your video file
Configure Settings:
✅ Show Pose Landmarks (visual skeleton overlay)
✅ Save Processed Video (download annotated video)
Start Analysis: Click "🚀 Start Analysis" and watch real-time processing
Download Results: Get CSV data and processed video when complete
🖼️ Image Analysis
Select Analysis Mode: Choose "🖼️ Image Analysis" from the sidebar
Choose Joints: Select joints to analyze
Upload Image: Upload a photo containing a person
View Results: See pose detection and joint angle measurements instantly
⚙️ Settings & Options
🎯 Show Pose Landmarks: Display skeleton overlay on video/images
💾 Save Processed Video: Enable to download annotated video
✅ Select All / ❌ Clear All: Quick joint selection buttons

Real-time Progress: Live updates during video processing
📊 Output Examples
📈 CSV Data Format

frame,timestamp,left_shoulder_angle,left_shoulder_velocity,right_shoulder_angle,right_shoulder_velocity
0,0.0,145.2,0.0,142.8,0.0
1,0.033,146.1,27.3,143.2,12.1
2,0.066,147.5,42.4,144.0,24.2
...
🎬 Processed Video Includes
✅ Green pose skeleton overlay
✅ Yellow joint angle annotations
✅ Red landmark connection points
✅ Original video background preserved
📊 Statistical Analysis


Average Angle: Mean joint angle throughout motion
Angle Range: Maximum flexibility (max - min angle)
Max/Min Velocity: Peak angular velocity measurements
Frame Count & Duration: Complete motion analysis metrics
🎯 Perfect For
🏥 Medical & Rehabilitation
Patient Progress Tracking: Monitor recovery over time
Physical Therapy: Analyze exercise form and range of motion
Gait Analysis: Study walking and movement patterns
Injury Assessment: Quantify movement limitations
 Sports & Fitness
Athletic Performance: Analyze technique and form
Training Optimization: Identify areas for improvement
Movement Screening: Assess functional movement patterns
Biomechanical Analysis: Study sport-specific motions

🎓 Research & Education
Motion Studies: Quantitative movement research
Biomechanics Research: Academic and clinical studies
Student Training: Educational tool for movement analysis
Data Collection: Standardized measurement protocols

🛠️ Technical Details
🔧 Built With
Streamlit: Web application framework
MediaPipe: Google's pose detection ML framework
OpenCV: Computer vision and video processing
Plotly: Interactive data visualization
Pandas: Data analysis and manipulation

📐 Angle Calculation Method

Uses vector mathematics to calculate angles between three points:

def calculate_angle(a, b, c):
    """Calculate angle at point b formed by points a, b, c"""
    ab = a - b  # Vector from b to a
    cb = c - b  # Vector from b to c
    cosine = dot(ab, cb) / (|ab| * |cb|)
    return arccos(cosine) * 180/π


🎥 Video Processing
Real-time Analysis: Process frames with live progress updates
Efficient Processing: Optimized for performance with large video files
Format Support: MP4, AVI, MOV, MKV formats supported
Quality Preservation: Original video quality maintained in processed output
📸 Screenshots

🎛️ Main Interface

Clean, professional web interface with sidebar controls

📹 Video Processing

Real-time frame analysis with live progress and charts

📊 Results Dashboard

Comprehensive analysis with interactive charts and statistics

🎬 Processed Video Output

Annotated video with pose landmarks and joint angles
🔧 System Requirements

💻 Minimum Requirements
OS: Windows 10, macOS 10.15, Ubuntu 18.04+
Python: 3.8 or higher
RAM: 4GB minimum, 8GB recommended
Storage: 1GB free space
Browser: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
⚡ Recommended Specifications
CPU: Multi-core processor for faster video processing
RAM: 16GB for large video files
GPU: Optional, but improves MediaPipe performance
SSD: Faster storage for video I/O operations

🐛 Troubleshooting
❓ Common Issues
Video won't upload/process:

✅ Check file format (MP4, AVI, MOV, MKV only)

✅ Verify file size (large files may take longer)

✅ Ensure stable internet connection

No pose detected:
✅ Ensure person is clearly visible in frame
✅ Check lighting conditions
✅ Verify minimal occlusion of body parts
Slow processing:
✅ Close other applications to free up resources
✅ Use smaller video files for testing
 Ensure sufficient RAM available

📞 Support

For technical support or questions:
Email: [biokineticum@proton.me]

Issues: Create an issue on GitHub (if public repository)

📄 License

Proprietary Software - All rights reserved.

This software is protected by copyright and proprietary license. See LICENSE.txt for full terms.

❌ Commercial use requires license

❌ Redistribution prohibited
❌ Modification without permission forbidden


✅ Personal/research use permitted

For commercial licensing inquiries, contact: [biokineticum@proton.me]


🙏 Acknowledgments


Mediapipe Team: For the excellent pose detection framework
Streamlit: For the intuitive web app framework
OpenCV Community: For computer vision tools
Python Community: For the amazing ecosystem of libraries
📞 Contact

Developer: Dariusz Mosler

Email: [biokinetiucm@proton.me]

Website: [biokineticum.com]


🦴 Joint Motion Analyzer - Making movement analysis accessible to everyone

Built with ❤️ using Python, Streamlit, and MediaPipe
