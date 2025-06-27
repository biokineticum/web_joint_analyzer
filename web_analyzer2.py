import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os
from io import StringIO
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="🦴 Joint Motion Analyzer",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

class WebJointAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        
        # Joint definitions
        self.joint_definitions = {
            'shoulder': {
                'left': ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
                'right': ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP')
            },
            'elbow': {
                'left': ('LEFT_WRIST', 'LEFT_ELBOW', 'LEFT_SHOULDER'),
                'right': ('RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER')
            },
            'hip': {
                'left': ('LEFT_KNEE', 'LEFT_HIP', 'LEFT_SHOULDER'),
                'right': ('RIGHT_KNEE', 'RIGHT_HIP', 'RIGHT_SHOULDER')
            },
            'knee': {
                'left': ('LEFT_ANKLE', 'LEFT_KNEE', 'LEFT_HIP'),
                'right': ('RIGHT_ANKLE', 'RIGHT_KNEE', 'RIGHT_HIP')
            },
            'wrist': {
                'left': ('LEFT_PINKY', 'LEFT_WRIST', 'LEFT_ELBOW'),
                'right': ('RIGHT_PINKY', 'RIGHT_WRIST', 'RIGHT_ELBOW')
            },
            'ankle': {
                'left': ('LEFT_FOOT_INDEX', 'LEFT_ANKLE', 'LEFT_KNEE'),
                'right': ('RIGHT_FOOT_INDEX', 'RIGHT_ANKLE', 'RIGHT_KNEE')
            }
        }
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ab = a - b
        cb = c - b
        cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    
    def analyze_frame(self, frame, selected_joints):
        """Analyze a single frame for joint angles and velocities"""
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        joint_data = {}
        annotated_frame = frame.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw pose landmarks
            self.drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            
            # Analyze selected joints
            for side, joint_name in selected_joints:
                joint_key = f"{side}_{joint_name}"
                
                try:
                    point_names = self.joint_definitions[joint_name][side]
                    
                    # Extract landmark coordinates
                    points = []
                    for point_name in point_names:
                        landmark_idx = getattr(self.mp_pose.PoseLandmark, point_name).value
                        point = [
                            landmarks[landmark_idx].x * frame_width,
                            landmarks[landmark_idx].y * frame_height
                        ]
                        points.append(point)
                    
                    # Calculate angle
                    angle = self.calculate_angle(points[0], points[1], points[2])
                    joint_data[joint_key] = {
                        'angle': angle,
                        'points': points,
                        'side': side,
                        'joint': joint_name
                    }
                    
                    # Add angle annotation on frame
                    center_point = points[1]  # Middle point (vertex)
                    cv2.putText(annotated_frame, f'{angle:.1f}°', 
                               (int(center_point[0]), int(center_point[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                except (AttributeError, IndexError):
                    joint_data[joint_key] = {'angle': 0, 'points': [], 'side': side, 'joint': joint_name}
        
        return joint_data, annotated_frame
    
    def process_video(self, video_file, selected_joints, progress_container, video_container, chart_container, save_video=False):
        """Process entire video and return analysis results with real-time display"""
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if saving video
        output_video = None
        output_path = None
        if save_video:
            output_path = tempfile.mktemp(suffix='_processed.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        if total_frames <= 0:
            st.error("❌ Could not read video file. Please check the file format.")
            return None, None
        
        results_data = []
        frame_count = 0
        prev_angles = {}
        
        # Create progress bar and status containers
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        
        # Create video and chart placeholders
        video_placeholder = video_container.empty()
        chart_placeholder = chart_container.empty()
        
        # Initialize data for real-time plotting
        plot_data = {f"{side}_{joint}": [] for side, joint in selected_joints}
        timestamps = []
        
        # Process frames with real-time display
        display_interval = max(1, total_frames // 50)  # Show ~50 frames during processing
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            joint_data, annotated_frame = self.analyze_frame(frame, selected_joints)
            
            # Save frame to output video if enabled
            if save_video and output_video is not None:
                output_video.write(annotated_frame)
            
            # Calculate velocities
            timestamp = frame_count / fps if fps > 0 else frame_count * 0.033
            frame_result = {
                'frame': frame_count,
                'timestamp': timestamp
            }
            
            for joint_key, data in joint_data.items():
                angle = data['angle']
                velocity = 0
                
                if joint_key in prev_angles and frame_count > 0:
                    dt = 1.0 / fps if fps > 0 else 0.033
                    velocity = (angle - prev_angles[joint_key]) / dt
                
                frame_result[f'{joint_key}_angle'] = angle
                frame_result[f'{joint_key}_velocity'] = velocity
                prev_angles[joint_key] = angle
                
                # Store for real-time plotting
                if joint_key in plot_data:
                    plot_data[joint_key].append(velocity)
            
            timestamps.append(timestamp)
            results_data.append(frame_result)
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
            
            # Display frame and update chart periodically
            if frame_count % display_interval == 0 or frame_count == total_frames:
                # Convert frame for display
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, caption=f"Frame {frame_count}/{total_frames}", use_container_width=True)
                
                # Update real-time chart
                if len(timestamps) > 1:
                    fig = go.Figure()
                    colors = px.colors.qualitative.Set3
                    
                    for i, (side, joint_name) in enumerate(selected_joints):
                        joint_key = f"{side}_{joint_name}"
                        if joint_key in plot_data and len(plot_data[joint_key]) > 0:
                            color = colors[i % len(colors)]
                            label = f"{side.title()} {joint_name.title()}"
                            
                            fig.add_trace(go.Scatter(
                                x=timestamps[:len(plot_data[joint_key])],
                                y=plot_data[joint_key],
                                name=label,
                                line=dict(color=color),
                                mode='lines'
                            ))
                    
                    fig.update_layout(
                        title="Real-time Joint Angular Velocities",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Angular Velocity (deg/s)",
                        height=400,
                        showlegend=True
                    )
                    
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        cap.release()
        if output_video is not None:
            output_video.release()
        
        os.unlink(tfile.name)  # Clean up temp file
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing completed!")
        
        return pd.DataFrame(results_data), output_path

def main():
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = WebJointAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Header
    st.markdown('<h1 class="main-header">🦴 Joint Motion Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time joint motion analysis using MediaPipe and computer vision")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Analysis mode selection
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["📹 Video Upload", "🖼️ Image Analysis"]
        )
        
        st.markdown("---")
        
        # Joint selection
        st.subheader("🦴 Select Joints to Analyze")
        
        # Create joint selection interface
        selected_joints = []
        
        for joint_name in analyzer.joint_definitions.keys():
            st.write(f"**{joint_name.title()}**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.checkbox(f"Left {joint_name}", key=f"left_{joint_name}"):
                    selected_joints.append(('left', joint_name))
            
            with col2:
                if st.checkbox(f"Right {joint_name}", key=f"right_{joint_name}"):
                    selected_joints.append(('right', joint_name))
        
        st.markdown("---")
        
        # Quick selection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select All"):
                for joint_name in analyzer.joint_definitions.keys():
                    st.session_state[f"left_{joint_name}"] = True
                    st.session_state[f"right_{joint_name}"] = True
                st.rerun()
        
        with col2:
            if st.button("❌ Clear All"):
                for joint_name in analyzer.joint_definitions.keys():
                    st.session_state[f"left_{joint_name}"] = False
                    st.session_state[f"right_{joint_name}"] = False
                st.rerun()
        
        # Analysis settings
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        show_landmarks = st.checkbox("🎯 Show Pose Landmarks", value=True)
        save_processed_video = st.checkbox("💾 Save Processed Video", value=False, 
                                         help="Save video with pose annotations and joint angles")
        
        # Display selected joints count
        if selected_joints:
            st.success(f"📊 {len(selected_joints)} joints selected")
        else:
            st.warning("⚠️ Please select at least one joint")
    
    # Main content area
    if analysis_mode == "📹 Video Upload":
        st.header("📹 Video Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None and selected_joints:
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            # Create layout for video processing
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📺 Video Processing")
                video_container = st.container()
            
            with col2:
                st.subheader("📊 Real-time Analysis")
                chart_container = st.container()
            
            # Progress section
            st.subheader("🔄 Processing Progress")
            progress_container = st.container()
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary"):
                with st.spinner("🔄 Processing video..."):
                    try:
                        # Reset uploaded file pointer
                        uploaded_file.seek(0)
                        
                        # Process video with real-time display
                        results_df, output_video_path = analyzer.process_video(
                            uploaded_file, selected_joints, 
                            progress_container, video_container, chart_container,
                            save_video=save_processed_video
                        )
                        
                        if results_df is not None:
                            st.success("✅ Analysis completed!")
                            
                            # Final results section
                            st.header("📊 Final Results")
                            
                            # Create comprehensive visualization
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("📈 Complete Analysis")
                                
                                # Create interactive plot
                                fig = make_subplots(
                                    rows=2, cols=1,
                                    subplot_titles=('Joint Angles (degrees)', 'Angular Velocities (deg/s)'),
                                    vertical_spacing=0.1
                                )
                                
                                colors = px.colors.qualitative.Set3
                                
                                for i, (side, joint_name) in enumerate(selected_joints):
                                    joint_key = f"{side}_{joint_name}"
                                    color = colors[i % len(colors)]
                                    label = f"{side.title()} {joint_name.title()}"
                                    
                                    if f'{joint_key}_angle' in results_df.columns:
                                        # Angles plot
                                        fig.add_trace(
                                            go.Scatter(
                                                x=results_df['timestamp'],
                                                y=results_df[f'{joint_key}_angle'],
                                                name=f"{label} Angle",
                                                line=dict(color=color),
                                                legendgroup=label
                                            ),
                                            row=1, col=1
                                        )
                                        
                                        # Velocities plot
                                        fig.add_trace(
                                            go.Scatter(
                                                x=results_df['timestamp'],
                                                y=results_df[f'{joint_key}_velocity'],
                                                name=f"{label} Velocity",
                                                line=dict(color=color, dash='dash'),
                                                legendgroup=label,
                                                showlegend=False
                                            ),
                                            row=2, col=1
                                        )
                                
                                fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
                                fig.update_yaxes(title_text="Angle (degrees)", row=1, col=1)
                                fig.update_yaxes(title_text="Velocity (deg/s)", row=2, col=1)
                                fig.update_layout(height=600, showlegend=True)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.subheader("📈 Statistics")
                                
                                # Calculate and display statistics
                                for side, joint_name in selected_joints:
                                    joint_key = f"{side}_{joint_name}"
                                    if f'{joint_key}_angle' in results_df.columns:
                                        angles = results_df[f'{joint_key}_angle']
                                        velocities = results_df[f'{joint_key}_velocity']
                                        
                                        st.markdown(f"**{side.title()} {joint_name.title()}**")
                                        
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("Avg Angle", f"{angles.mean():.1f}°")
                                            st.metric("Max Velocity", f"{velocities.max():.1f}°/s")
                                        with col_b:
                                            st.metric("Range", f"{angles.max() - angles.min():.1f}°")
                                            st.metric("Min Velocity", f"{velocities.min():.1f}°/s")
                                        
                                        st.markdown("---")
                            
                            # Data download section
                            st.subheader("💾 Download Results")
                            
                            # Prepare CSV
                            csv_buffer = StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            # Summary info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Frames", len(results_df))
                            with col2:
                                st.metric("Duration", f"{results_df['timestamp'].max():.1f}s")
                            with col3:
                                st.metric("Joints Analyzed", len(selected_joints))
                            
                            # Download buttons in columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # CSV Download
                                st.download_button(
                                    label="📊 Download CSV Data",
                                    data=csv_data,
                                    file_name=f"joint_analysis_{uploaded_file.name}_{int(time.time())}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # Video Download (if processed video was saved)
                                if save_processed_video and output_video_path and os.path.exists(output_video_path):
                                    with open(output_video_path, "rb") as video_file:
                                        video_bytes = video_file.read()
                                    
                                    st.download_button(
                                        label="🎬 Download Processed Video",
                                        data=video_bytes,
                                        file_name=f"processed_{uploaded_file.name}_{int(time.time())}.mp4",
                                        mime="video/mp4"
                                    )
                                    
                                    # Clean up temporary video file after download
                                    try:
                                        os.unlink(output_video_path)
                                    except:
                                        pass
                                else:
                                    st.info("💡 Enable 'Save Processed Video' to download annotated video")
                            
                            # Show data preview
                            with st.expander("🔍 View Raw Data"):
                                st.dataframe(results_df, use_container_width=True)
                        
                        else:
                            st.error("❌ Failed to process video. Please check the file format and try again.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
                        st.write("Please ensure the video file is valid and try again.")
        
        elif uploaded_file is not None and not selected_joints:
            st.warning("⚠️ Please select at least one joint to analyze")
        
        elif uploaded_file is None:
            st.info("👆 Please upload a video file to begin analysis")
    
    elif analysis_mode == "🖼️ Image Analysis":
        st.header("🖼️ Single Image Analysis")
        
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing a person for pose analysis"
        )
        
        if uploaded_image is not None and selected_joints:
            # Read and display image
            image = Image.open(uploaded_image)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Pose Analysis")
                
                # Analyze image
                joint_data, annotated_image = analyzer.analyze_frame(image_cv, selected_joints)
                
                if show_landmarks:
                    rgb_annotated = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    st.image(rgb_annotated, use_container_width=True)
                else:
                    st.image(image, use_container_width=True)
            
            # Display joint angles
            st.subheader("📊 Joint Angles")
            
            if joint_data:
                angle_data = []
                for joint_key, data in joint_data.items():
                    if data['angle'] > 0:  # Only show detected joints
                        angle_data.append({
                            'Joint': f"{data['side'].title()} {data['joint'].title()}",
                            'Angle (degrees)': f"{data['angle']:.1f}°"
                        })
                
                if angle_data:
                    df_display = pd.DataFrame(angle_data)
                    st.table(df_display)
                else:
                    st.warning("⚠️ No pose detected in the image")
            else:
                st.warning("⚠️ No pose detected in the image")
        
        elif uploaded_image is not None and not selected_joints:
            st.warning("⚠️ Please select at least one joint to analyze")
        
        elif uploaded_image is None:
            st.info("👆 Please upload an image file to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🦴 Joint Motion Analyzer | Built with Streamlit, MediaPipe & OpenCV</p>
        <p>💡 <strong>Tip:</strong> Check 'Save Processed Video' to download video with pose annotations!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
