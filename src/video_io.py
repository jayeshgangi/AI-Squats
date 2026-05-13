import cv2

def load_video(path):
    cap = cv2.VideoCapture(path)
    return cap

def create_writer(cap, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30
    print(f"FPS: {fps}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        raise Exception("Invalid video dimensions")

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise Exception("VideoWriter failed")

    return writer