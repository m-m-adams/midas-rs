import count_min_sketch

class CMSCounter:
    def __init__(self, nrows: int, ncolumns: int):
        self.current = count_min_sketch.CMS(nrows, ncolumns)
        self.total = count_min_sketch.clone_cms(self.current)
        self.timestamp = 0



