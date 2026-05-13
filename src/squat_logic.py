class SquatCounter:
    def __init__(self, down_thresh, up_thresh):
        self.down_thresh = down_thresh
        self.up_thresh = up_thresh
        self.state = "up"
        self.count = 0

    def update(self, knee_angle, hip_angle):

        # combined signal (more robust)
        angle = (knee_angle * 0.7 + hip_angle * 0.3)

        if self.state == "up" and angle < self.down_thresh:
            self.state = "down"

        elif self.state == "down" and angle > self.up_thresh:
            self.state = "up"
            self.count += 1

        return self.count, self.state