import cv2

def draw_info(frame, count, angles):
    cv2.putText(frame, f"Squats: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    y = 80
    for k, v in angles.items():
        if isinstance(v, (int, float)):
            cv2.putText(frame, f"{k}: {int(v)}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 30

    return frame