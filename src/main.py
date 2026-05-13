import cv2 , os , logging , sys, contextlib ,config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stderr(fnull):
        import mediapipe as mp

from src.video_io import load_video, create_writer
from src.pose_estimation import PoseEstimator
from src.angle_utils import calculate_angle, AngleSmoother
from src.squat_logic import SquatCounter
from src.vizualizer import draw_info
from src.logger import AngleLogger
from src.stats_logger import StatsLogger
from tqdm import tqdm

mp_pose = mp.solutions.pose

input_folder = "Input"
output_folder = "Output"
angles_folder = "Frame_Angles"
stats_folder = "Stats"

def folder_creation(folder: str):
    os.makedirs(folder, exist_ok=True)

# create output folder if not exists
folder_creation(output_folder)
folder_creation(angles_folder)
folder_creation(stats_folder)

stats_logger = StatsLogger()
prev_stage = None

# list available videos
videos = [v for v in os.listdir(input_folder) if v.lower().endswith(('.mp4', '.avi', '.mov'))]
print("\nAvailable videos:")
for v in videos:
    print("-", v)

knee_smoother = AngleSmoother()
hip_smoother = AngleSmoother()

video_name = input("\nEnter video name (with extension): ")

video_path = os.path.join(input_folder, video_name)

# output name
name, ext = os.path.splitext(video_name)
output_path = os.path.join(output_folder, f"{name}_marked.mp4")
cap = load_video(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps==0:
    fps=30

if not cap.isOpened():
    raise Exception("Video file not found or cannot be opened")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tqdm.write(f"Video size: {width} {height}")
writer = create_writer(cap, output_path)

pose = PoseEstimator()
pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame",ncols=100)
counter = SquatCounter(config.KNEE_ANGLE_DOWN, config.KNEE_ANGLE_UP)
logger = AngleLogger()

frame_id = 0
last_stage_change_frame = 0
MIN_GAP = int(0.12 * fps)   # frames
prev_angle = None
direction ="up"

PoseLandmark = mp_pose.PoseLandmark

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pbar.update(1)

    landmarks = pose.get_landmarks(frame)

    if landmarks:
        lm = landmarks.landmark

        l_vis = lm[PoseLandmark.LEFT_KNEE.value].visibility
        r_vis = lm[PoseLandmark.RIGHT_KNEE.value].visibility

        if l_vis < config.CONF_THRESH and r_vis < config.CONF_THRESH:
            writer.write(frame)
            frame_id += 1
            continue

        # LEFT side
        l_hip = [lm[PoseLandmark.LEFT_HIP.value].x,
                 lm[PoseLandmark.LEFT_HIP.value].y]
        
        l_knee = [lm[PoseLandmark.LEFT_KNEE.value].x,
                  lm[PoseLandmark.LEFT_KNEE.value].y]

        l_ankle = [lm[PoseLandmark.LEFT_ANKLE.value].x,
                   lm[PoseLandmark.LEFT_ANKLE.value].y]

        l_shoulder = [lm[PoseLandmark.LEFT_SHOULDER.value].x,
                      lm[PoseLandmark.LEFT_SHOULDER.value].y]

        # RIGHT side
        r_hip = [lm[PoseLandmark.RIGHT_HIP.value].x,
                 lm[PoseLandmark.RIGHT_HIP.value].y]

        r_knee = [lm[PoseLandmark.RIGHT_KNEE.value].x,
                  lm[PoseLandmark.RIGHT_KNEE.value].y]

        r_ankle = [lm[PoseLandmark.RIGHT_ANKLE.value].x,
                   lm[PoseLandmark.RIGHT_ANKLE.value].y]

        r_shoulder = [lm[PoseLandmark.RIGHT_SHOULDER.value].x,
                      lm[PoseLandmark.RIGHT_SHOULDER.value].y]
        
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
            # equal visibility → average
            knee_angle_raw = (left_knee_angle + right_knee_angle) / 2
            hip_angle_raw = (left_hip_angle + right_hip_angle) / 2

        knee_angle = knee_smoother.smooth(knee_angle_raw)
        hip_angle = hip_smoother.smooth(hip_angle_raw)

        if abs(knee_angle - knee_angle_raw) > 30:
            knee_angle = 0.7 * knee_angle + 0.3 * knee_angle_raw

        if abs(hip_angle - hip_angle_raw) > 30:
            hip_angle = 0.7 * hip_angle + 0.3 * hip_angle_raw

        knee_angle = max(0, min(180, knee_angle))
        hip_angle = max(0, min(180, hip_angle))

        
        if prev_angle is not None:
            velocity = knee_angle - prev_angle

            if velocity < 0:
                direction = "down"
            else:
                direction = "up"

        prev_angle = knee_angle

        if knee_angle < 40 or knee_angle > 180:
            writer.write(frame)
            frame_id += 1
            continue

        if (frame_id - last_stage_change_frame) > fps * 3:
            if counter.state == "down":
            # reset state if stuck too long
                counter.state = "up"

        count_before = counter.count
        state_before = counter.state

        count, stage = counter.update(knee_angle, hip_angle)
        valid = True
        if stage == "down" and direction != "down":
            valid = False

        if stage == "up" and direction != "up":
            valid = False

        if not valid:
            counter.count = count_before   # revert count
            counter.state = state_before   # revert state
            stage = state_before

        pbar.set_postfix({"Squats": counter.count})
        if prev_stage is None:
            prev_stage = stage
        else:
            if stage != prev_stage and (frame_id - last_stage_change_frame) > MIN_GAP:

                current_rep = counter.count
                if stage == "down":
                    stats_logger.log(frame_id, "down", knee_angle, hip_angle, current_rep+1)

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
        
        logger.log(frame_id, angles,video_name)

        frame = draw_info(frame, count, angles)
        frame = pose.draw_pose(frame, landmarks)
        lkx = max(10, min(width-100, int(l_knee[0]*width)))
        lky = max(30, min(height-10, int(l_knee[1]*height)))

        rkx = max(10, min(width-100, int(r_knee[0]*width)))
        rky = max(30, min(height-10, int(r_knee[1]*height)))

        cv2.putText(frame, str(int(left_knee_angle)), (lkx+10, lky-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.putText(frame, str(int(right_knee_angle)), (rkx+10, rky-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    else:
        cv2.putText(frame,"No Pose Detected", (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    writer.write(frame)
    frame_id += 1

pbar.close()
cap.release()
writer.release()
print("\nTotal Squats", counter.count)
csv_path = os.path.join(angles_folder, f"{name}_angles.csv")

stats_path = os.path.join(stats_folder, f"{name}_stats.csv")
stats_logger.save(stats_path)

logger.save(csv_path)