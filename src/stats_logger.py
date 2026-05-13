class StatsLogger:
    def __init__(self):
        self.data = []

    def log(self, frame_id, stage, knee_angle, hip_angle, rep):
        self.data.append({
            "frame": frame_id,
            "rep": rep,
            "stage": stage,
            "knee_angle": round(knee_angle, 2),
            "hip_angle": round(hip_angle, 2)
        })

    def save(self, path):
        if not self.data:
            return

        import csv
        keys = self.data[0].keys()

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.data)