import cv2,os,sys,logging,shutil,contextlib,time
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse , HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from contextlib import asynccontextmanager
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stderr(fnull):
        import mediapipe as mp

mp_pose = mp.solutions.pose

from src.angle_utils import calculate_angle, AngleSmoother
from src.logger import AngleLogger
from src.pose_estimation import PoseEstimator
from src.squat_logic import SquatCounter
from src.stats_logger import StatsLogger
from src.video_io import load_video, create_writer
from src.vizualizer import draw_info
from src import config

# --------------------- PATHS & FOLDERS --------------------- #

BASE_DIR = os.getcwd()
INPUT_FOLDER = os.path.join(BASE_DIR, "Input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Output")
ANGLES_FOLDER = os.path.join(BASE_DIR, "Frame_Angles")
STATS_FOLDER = os.path.join(BASE_DIR, "Stats")
LOG_FOLDER = os.path.join(BASE_DIR, "Logs")

# -------------------- LOGGER SETUP -------------------- #

def generate_base_name(original_filename):
    name, _  = os.path.splitext(original_filename)
    return f"fastapi_{name}"

def setup_logger():
    os.makedirs(LOG_FOLDER, exist_ok=True)

    logger = logging.getLogger("Squat Analyzer")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(LOG_FOLDER, 'FastAPI.log'))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

app = FastAPI(title='Squat Counter API',
              description='Upload a video of squats and get back the analyzed video with counts and angles, along with CSV files for angles and stats.',
              version='1.0')


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # later restrict this
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

#-------------------- UTILS -------------------- #

def make_folder(folder: str):

    """ Creates a folder if it doesn't exist."""

    os.makedirs(folder,exist_ok=True)
    return folder

def setup_folders()->None: 

    """ Ensures all necessary folders exist."""

    make_folder(INPUT_FOLDER)
    make_folder(OUTPUT_FOLDER)
    make_folder(ANGLES_FOLDER)
    make_folder(STATS_FOLDER)
    make_folder(LOG_FOLDER)

def get_video_paths(video_name):

    """ Generates all output paths for a video """

    video_path = os.path.join(INPUT_FOLDER, video_name)
    name , _ = os.path.splitext(video_name)
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}_marked.mp4")
    csv_path = os.path.join(ANGLES_FOLDER, f"{name}_angles.csv")
    stats_path = os.path.join(STATS_FOLDER, f"{name}_stats.csv")

    return video_path,output_path,csv_path,stats_path,name

def initialize_video(video_path,output_path):

    """ Loads the video and initializes the writer """

    logger.info(f"Loading video: {video_path}")
    cap = load_video(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise Exception(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video properties - FPS: {fps}, Total Frames: {total_frames}, Resolution: {width} x {height}")

    writer = create_writer(cap,output_path)

    return cap,writer,total_frames,fps,width,height

def initialize_models_and_utils():

    """ Initialize pose, counters , loggers and smoothers """
    pose = PoseEstimator()
    counter = SquatCounter(config.KNEE_ANGLE_DOWN,config.KNEE_ANGLE_UP)
    angle_logger = AngleLogger()
    stats_logger = StatsLogger()

    knee_smoother = AngleSmoother()
    hip_smoother = AngleSmoother()

    return pose,counter,angle_logger,stats_logger,knee_smoother,hip_smoother


#-------------------- CORE PROCESSING -------------------- #

def process_video(cap,writer,total_frames,fps,width,height,video_name,pose,counter,angle_logger,stats_logger,knee_smoother,hip_smoother):
    
    """ Main loop to process the video frame by frame """

    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame",ncols=100)

    logger.info("Starting video processing...")

    frame_id=0
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

            angle_logger.log(frame_id, angles, video_name)

            frame = draw_info(frame, count, angles)
            frame = pose.draw_pose(frame, landmarks)

        else:
            cv2.putText(frame, "No Pose Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        writer.write(frame)
        frame_id += 1

    pbar.close()
    logger.info(f" Processing complete. Total Squats Counted: {counter.count}")
    return counter.count

#-------------------- PIPELINE -------------------- #

def run_pipeline(video_name):

    """ Full Pipeline to process a video and return the squat count """

    setup_folders()

    video_path,output_path,csv_path,stats_path,name = get_video_paths(video_name)

    cap,writer,total_frames,fps,width,height = initialize_video(video_path,output_path)

    pose,counter,angle_logger,stats_logger,knee_smoother,hip_smoother = initialize_models_and_utils()

    total_squats = process_video(cap,writer,total_frames,fps,width,height,video_name,pose,counter,angle_logger,stats_logger,
                                 knee_smoother,hip_smoother)
    
    cap.release()
    writer.release()

    stats_logger.save(stats_path)
    angle_logger.save(csv_path)

    return total_squats,output_path,csv_path,stats_path

# -------------------- API ENDPOINT -------------------- #

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Squat Counter API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    text-align: center;
                    background-color: #f4f6f8;
                    margin: 0;
                    padding: 0;
                }
                .container {
                    margin-top: 80px;
                }
                h1 {
                    color: #2c3e50;
                }
                p {
                    color: #555;
                    font-size: 18px;
                }
                .btn {
                    display: inline-block;
                    margin: 10px;
                    padding: 12px 20px;
                    font-size: 16px;
                    color: white;
                    background-color: #3498db;
                    text-decoration: none;
                    border-radius: 6px;
                }
                .btn:hover {
                    background-color: #2980b9;
                }
                .status {
                    margin-top: 20px;
                    color: green;
                    font-weight: bold;
                }
                .footer {
                    margin-top: 50px;
                    font-size: 14px;
                    color: #888;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Squat Counter API</h1>
                <p>Upload a squat video and get automated analysis with counts, angles, and stats.</p>

                <a class="btn" href="/docs">API Docs</a>
                <a class="btn" href="/health">Health Check</a>

                <div class="status">Server is running</div>

                <div class="footer">
                    FastAPI Backend | Video Processing Engine
                </div>
            </div>
        </body>
    </html>
    """
@app.get("/test")
def test():
    return {"msg": "working"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    setup_folders()
    logger.info("All folders ready")

    yield  # app runs here

    # SHUTDOWN (optional)
    logger.info("Shutting down app")

@app.post('/upload-video/')
async def upload_video(background_tasks: BackgroundTasks,file: UploadFile = File(...)):

    """ 
    Upload a video file and process squat counting.
    Returns:
    - Total squat count
    - Ouput video path
    - CSV logs
    """

    try:
        
        if not file.filename.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Only .mp4 files allowed")
        
        base_name = generate_base_name(file.filename)

        input_filename = f"{base_name}.mp4"
        input_path = os.path.join(INPUT_FOLDER, input_filename)

        logger.info(f"Received file : {base_name}")


        # Save uploaded file

        with open(input_path,'wb') as buffer:
            shutil.copyfileobj(file.file,buffer)

        logger.info("File saved. Starting processing...")

        #total_squats,output_path,csv_path,stats_path = run_pipeline(input_filename)
        
        background_tasks.add_task(run_pipeline, input_filename)

        name, _ = os.path.splitext(input_filename)

        return JSONResponse({
            "status": "processing",
            "message": "Video uploaded. Processing started.",
            "input_video": input_filename,
            "output_video": f"/download/video/{name}_marked.mp4",
            "angles_csv": f"/download/angles/{name}_angles.csv",
            "stats_csv": f"/download/stats/{name}_stats.csv"
        })
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Internal server error during video processing: {str(e)}")
    

@app.get("/download/video/{filename}")
def download_video(filename: str):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/download/angles/{filename}")
def download_angles(filename: str):
    path = os.path.join(ANGLES_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename=filename)


@app.get("/download/stats/{filename}")
def download_stats(filename: str):
    path = os.path.join(STATS_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename=filename)

@app.get("/health")
def health():
    return {"status": "ok"}