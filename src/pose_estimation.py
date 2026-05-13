import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseEstimator:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,   # BEST
            # enable_segmentation=False,
            # min_detection_confidence=0.5,
            # min_tracking_confidence=0.5
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            enable_segmentation=False
        )
    def get_landmarks(self, frame):
        rgb = frame[:, :, ::-1]
        result = self.pose.process(rgb)
        return result.pose_landmarks
    
    def draw_pose(self, frame, landmarks):
        if landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=4)
            )
        return frame