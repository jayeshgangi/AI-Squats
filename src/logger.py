import pandas as pd

class AngleLogger:
    def __init__(self):
        self.data = []

    def log(self, frame_id, angles,filename):
        angles['filename']=filename
        angles['frame'] = frame_id
        self.data.append(angles)

    def save(self, path):
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)