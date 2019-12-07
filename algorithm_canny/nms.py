class NonMaximumSuppression:
    def __init__(self, magnitudes, angles):
        self.magnitudes = magnitudes
        self.angles = angles
        self.height, self.width = angles.shape

    def suppress(self):
        suppressed = self.magnitudes.copy()
        for i in range(self.height):
            for j in range(self.width):
                i1, i2, j1, j2 = i, i, j, j
                current_angle = self.angles[i][j]
                if current_angle == 0:
                    j1 -= 1
                    j2 += 1
                elif current_angle == 45:
                    i1 += 1
                    j1 -= 1
                    i2 -= 1
                    j2 += 1
                elif current_angle == 135:
                    i1 += 1
                    j1 += 1
                    i2 -= 1
                    j2 -= 1
                else:
                    i1 += 1
                    i2 -= 1
                current_value = self.get_magnitude_value(i, j)
                first_value = self.get_magnitude_value(i1, j1)
                second_value = self.get_magnitude_value(i2, j2)
                if current_value <= first_value or current_value <= second_value:
                    suppressed[i][j] = 0
                else:
                    suppressed[i][j] = current_value
        return suppressed

    def get_magnitude_value(self, i, j):
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return 0
        else:
            return self.magnitudes[i][j]
