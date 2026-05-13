import cv2, os, sys, contextlib, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stderr(fnull):
        import mediapipe as mp

mp_pose = mp.solutions.pose

from src.video_io import load_video, create_writer
from src.pose_estimation import PoseEstimator
from src.angle_utils import calculate_angle, AngleSmoother
from src.squat_logic import SquatCounter
from src.vizualizer import draw_info
from src.logger import AngleLogger
from src.stats_logger import StatsLogger
from src.tqdm import tqdm
import config as config

mp_pose = __import__('mediapipe').solutions.pose

input_folder = "Input"
output_folder = "Output"
angles_folder = "Frame_Angles"
stats_folder = "Stats"


# -------------------- UTILS --------------------

def folder_creation(folder: str):
    os.makedirs(folder, exist_ok=True)


def setup_folders():
    folder_creation(output_folder)
    folder_creation(angles_folder)
    folder_creation(stats_folder)


def list_videos():
    videos = [v for v in os.listdir(input_folder) if v.lower().endswith(('.mp4', '.avi', '.mov'))]
    print("\nAvailable videos:")
    for v in videos:
        print("-", v)


def get_video_paths(video_name):
    video_path = os.path.join(input_folder, video_name)
    name, ext = os.path.splitext(video_name)
    output_path = os.path.join(output_folder, f"{name}_marked.mp4")
    csv_path = os.path.join(angles_folder, f"{name}_angles.csv")
    stats_path = os.path.join(stats_folder, f"{name}_stats.csv")

    return video_path, output_path, csv_path, stats_path, name


def initialize_video(video_path, output_path):
    cap = load_video(video_path)

    if not cap.isOpened():
        raise Exception("Video file not found or cannot be opened")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tqdm.write(f"Video size: {width} {height}")

    writer = create_writer(cap, output_path)

    return cap, writer, total_frames, fps, width, height


def initialize_models_and_utils():
    pose = PoseEstimator()
    counter = SquatCounter(config.KNEE_ANGLE_DOWN, config.KNEE_ANGLE_UP)
    logger = AngleLogger()
    stats_logger = StatsLogger()

    knee_smoother = AngleSmoother()
    hip_smoother = AngleSmoother()

    return pose, counter, logger, stats_logger, knee_smoother, hip_smoother


# -------------------- CORE PROCESSING --------------------

def process_video(cap, writer, total_frames, fps, width, height, video_name,
                  pose, counter, logger, stats_logger,
                  knee_smoother, hip_smoother):

    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame", ncols=100)

    frame_id = 0
    last_stage_change_frame = 0
    prev_stage = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    MIN_GAP = int(0.2 * fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        landmarks = pose.get_landmarks(frame)

        if landmarks:
            lm = landmarks.landmark

            l_vis = lm[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
            r_vis = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

            if l_vis < 0.5 and r_vis < 0.5:
                writer.write(frame)
                frame_id += 1
                continue

            # LEFT
            l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            l_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            l_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            # RIGHT
            r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

            if l_vis > r_vis:
                knee_angle_raw = left_knee_angle
                hip_angle_raw = left_hip_angle
            elif r_vis > l_vis:
                knee_angle_raw = right_knee_angle
                hip_angle_raw = right_hip_angle
            else:
                knee_angle_raw = (left_knee_angle + right_knee_angle) / 2
                hip_angle_raw = (left_hip_angle + right_hip_angle) / 2

            knee_angle = knee_smoother.smooth(knee_angle_raw)
            hip_angle = hip_smoother.smooth(hip_angle_raw)

            if abs(knee_angle - knee_angle_raw) > 30:
                knee_angle = knee_angle_raw

            if abs(hip_angle - hip_angle_raw) > 30:
                hip_angle = hip_angle_raw

            knee_angle = max(0, min(180, knee_angle))
            hip_angle = max(0, min(180, hip_angle))

            if knee_angle < 40 or knee_angle > 180:
                writer.write(frame)
                frame_id += 1
                continue

            count, stage = counter.update(knee_angle, hip_angle)
            pbar.set_postfix({"Squats": counter.count})

            if prev_stage is None and stage == "up":
                prev_stage = stage
                continue

            if stage != prev_stage and (frame_id - last_stage_change_frame) > MIN_GAP:

                current_rep = counter.count

                if stage == "down":
                    stats_logger.log(frame_id, "down", knee_angle, hip_angle, current_rep + 1)

                elif stage == "up":
                    stats_logger.log(frame_id, "up", knee_angle, hip_angle, current_rep)

                last_stage_change_frame = frame_id
                prev_stage = stage

            angles = {
                "knee_angle": knee_angle,
                "hip_angle": hip_angle,
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle
            }

            logger.log(frame_id, angles, video_name)

            frame = draw_info(frame, count, angles)
            frame = pose.draw_pose(frame, landmarks)

            lkx = max(10, min(width - 100, int(l_knee[0] * width)))
            lky = max(30, min(height - 10, int(l_knee[1] * height)))

            rkx = max(10, min(width - 100, int(r_knee[0] * width)))
            rky = max(30, min(height - 10, int(r_knee[1] * height)))

            cv2.putText(frame, str(int(left_knee_angle)), (lkx + 10, lky - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, str(int(right_knee_angle)), (rkx + 10, rky - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        else:
            cv2.putText(frame, "No Pose Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        writer.write(frame)
        frame_id += 1

    pbar.close()
    return counter.count


# -------------------- FINAL PIPELINE --------------------

def run_pipeline(video_name):
    setup_folders()

    video_path, output_path, csv_path, stats_path, name = get_video_paths(video_name)

    cap, writer, total_frames, fps, width, height = initialize_video(video_path, output_path)

    pose, counter, logger, stats_logger, knee_smoother, hip_smoother = initialize_models_and_utils()

    total_squats = process_video(
        cap, writer, total_frames, fps, width, height, video_name,
        pose, counter, logger, stats_logger,
        knee_smoother, hip_smoother
    )

    cap.release()
    writer.release()

    stats_logger.save(stats_path)
    logger.save(csv_path)

    print("\nTotal Squats", total_squats)


# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    list_videos()
    video_name = input("\nEnter video name (with extension): ")
    run_pipeline(video_name)