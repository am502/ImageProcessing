import numpy as np

from algorithm_canny.disjoint_set import DisjointSet


class TwoPassLabeling:
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape
        self.labels = np.zeros((self.height, self.width), dtype=int)
        self.disjoint_set = DisjointSet()
        self.i_offset = [0, -1, -1]
        self.j_offset = [-1, -1, 0]

    def label(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.image[i][j] != 0:
                    first_label = 0
                    min_label = self.disjoint_set.get_id()
                    for k in range(len(self.i_offset)):
                        current_label = self.get_label(i + self.i_offset[k], j + self.j_offset[k])
                        if current_label != 0:
                            if first_label == 0:
                                first_label = current_label
                            else:
                                self.disjoint_set.union(first_label - 1, current_label - 1)
                                first_label = current_label
                            if current_label < min_label:
                                min_label = current_label
                    if first_label == 0:
                        self.disjoint_set.make_set()
                        min_label = self.disjoint_set.get_id()
                    self.labels[i][j] = min_label

        for i in range(self.height):
            for j in range(self.width):
                if self.labels[i][j] != 0:
                    self.labels[i][j] = self.disjoint_set.find(self.labels[i][j] - 1) + 1

        return self.labels

    def get_label(self, i, j):
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return 0
        return self.labels[i][j]
