class IncrementalMean(object):
    def __init__(self, size=100):
        self.value = None
        self.size = size

    def add(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value += (x - self.value) / self.size
        return self.value
