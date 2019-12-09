class DisjointSet:
    def __init__(self):
        self.id = 0
        self.parents = []
        self.ranks = []

    def make_set(self):
        self.parents.append(self.id)
        self.ranks.append(0)
        self.id += 1

    def find(self, x):
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.ranks[x_root] < self.ranks[y_root]:
            self.parents[x_root] = y_root
        elif self.ranks[x_root] > self.ranks[y_root]:
            self.parents[y_root] = x_root
        else:
            self.parents[x_root] = y_root
            self.ranks[y_root] += 1

    def get_id(self):
        return self.id
