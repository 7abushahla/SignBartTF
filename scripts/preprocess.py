"""
Video to Keypoint Extraction Script

Extracts 90 keypoints per frame from video files using MediaPipe Holistic
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm


class KeypointExtractor:
    """Extract 90 keypoints from videos using MediaPipe Holistic"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Face mesh indices for 25 selected landmarks
        self.face_indices = [
            # Contour (10)
            10, 338, 297, 332, 172, 152, 397, 103, 67, 109,
            # Right eye (7)
            33, 133, 160, 159, 158, 144, 145,
            # Left eye (7)
            362, 263, 387, 386, 385, 373, 374,
            # Nose (1)
            1
        ]
    
    def extract_keypoints(self, video_path: str, sampling_ms: int = 150) -> np.ndarray:
        """
        Extract keypoints from video
        
        Args:
            video_path: Path to video file
            sampling_ms: Sampling interval in milliseconds
        
        Returns:
            keypoints: Array of shape (num_frames, 90, 2) - [x, y] coordinates
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps * sampling_ms / 1000)
        
        keypoints_list = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.holistic.process(rgb_frame)
                
                # Extract 90 keypoints
                kpts = self._extract_90_keypoints(results, frame.shape)
                keypoints_list.append(kpts)
            
            frame_count += 1
        
        cap.release()
        return np.array(keypoints_list)  # (num_frames, 90, 2)
    
    def _extract_90_keypoints(self, results, frame_shape) -> np.ndarray:
        """Extract 90 keypoints from MediaPipe results"""
        h, w = frame_shape[:2]
        keypoints = np.zeros((90, 2))
        
        # Pose (23 keypoints: upper body only, indices 0-22)
        if results.pose_landmarks:
            for i in range(23):  # Only upper body
                lm = results.pose_landmarks.landmark[i]
                keypoints[i] = [lm.x * w, lm.y * h]
        
        # Left hand (21 keypoints, indices 23-43)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                keypoints[23 + i] = [lm.x * w, lm.y * h]
        
        # Right hand (21 keypoints, indices 44-64)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                keypoints[44 + i] = [lm.x * w, lm.y * h]
        
        # Face (25 selected keypoints, indices 65-89)
        if results.face_landmarks:
            for i, idx in enumerate(self.face_indices):
                lm = results.face_landmarks.landmark[idx]
                keypoints[65 + i] = [lm.x * w, lm.y * h]
        
        return keypoints
    
    def close(self):
        """Release resources"""
        self.holistic.close()


def process_videos(input_dir: str, output_dir: str, sampling_ms: int = 150):
    """Process all videos in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = KeypointExtractor()
    video_files = list(input_path.glob('*.mp4'))
    
    print(f"Processing {len(video_files)} videos...")
    
    for video_file in tqdm(video_files):
        try:
            # Extract keypoints
            keypoints = extractor.extract_keypoints(str(video_file), sampling_ms)
            
            # Save as .npy
            output_file = output_path / f"{video_file.stem}.npy"
            np.save(output_file, keypoints)
            
            print(f"✅ {video_file.name}: {keypoints.shape[0]} frames, {keypoints.shape[1]} keypoints")
        
        except Exception as e:
            print(f"❌ Error processing {video_file.name}: {e}")
    
    extractor.close()
    print(f"✅ Done! Processed files saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keypoints from videos")
    parser.add_argument("--input", required=True, help="Input directory with videos")
    parser.add_argument("--output", required=True, help="Output directory for keypoints")
    parser.add_argument("--sampling", type=int, default=150, help="Sampling interval (ms)")
    
    args = parser.parse_args()
    process_videos(args.input, args.output, args.sampling)

