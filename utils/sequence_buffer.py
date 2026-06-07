from collections import deque, Counter


class SequenceBuffer:
    def __init__(self, size=30):
        self.size = size
        self.buffer = deque(maxlen=size)

    def add(self, item):
        self.buffer.append(item)

    def get(self):
        return list(self.buffer)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def get_most_common(self):
        if len(self.buffer) == 0:
            return ""
        return Counter(self.buffer).most_common(1)[0][0]

    def clear(self):
        # ✅ FIX: was self.buffer = [] which lost the deque and its maxlen
        self.buffer.clear()