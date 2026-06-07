from collections import deque, Counter

class SequenceBuffer:
    def __init__(self, size=30):
        self.buffer = deque(maxlen=size)

    def add(self, item):
        self.buffer.append(item)

    def get_most_common(self):
        if len(self.buffer) == 0:
            return ""
        return Counter(self.buffer).most_common(1)[0][0]
    
    def clear(self):
        self.buffer = []