import numpy as np
from matplotlib import pyplot as plt


class Parameters():
    def __init__(self, a=1, b=1, theta=np.pi / 2):
        self.a = a
        self.b = b
        self.theta = theta


    def matrix(self):
        return np.array([[self.a, self.b * np.cos(self.theta)],
                         [0,      self.b * np.sin(self.theta)]])
    

    def parallelogram(self):
        return ((0, self.a, self.b * np.cos(self.theta) + self.a, self.b * np.cos(self.theta), 0),
                (0, 0, self.b * np.sin(self.theta), self.b * np.sin(self.theta), 0))
        return ((0, 0),
                (self.a, 0),
                (self.b * np.cos(self.theta) + self.a, self.b * np.sin(self.theta)),
                (self.b * np.cos(self.theta), self.b * np.sin(self.theta)))


N = 10
X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
XY = np.vstack([X.ravel(), Y.ravel()])

# plt.scatter(XY[0], XY[1])

XY_rotated = Parameters(1, 1, np.pi/4).matrix() @ XY
plt.scatter(XY_rotated[0], XY_rotated[1])
plt.plot(*Parameters(1, 1, np.pi/4).parallelogram())

XY_rerotated = Parameters(2, 1, np.pi/6).matrix() @ np.linalg.inv(Parameters(1, 1, np.pi/4).matrix()) @ XY_rotated
plt.scatter(XY_rerotated[0], XY_rerotated[1])
plt.plot(*Parameters(2, 1, np.pi/6).parallelogram())

for i in range(len(XY_rotated[0])):
    plt.arrow(XY_rotated[0][i], XY_rotated[1][i], XY_rerotated[0][i]-XY_rotated[0][i], XY_rerotated[1][i]-XY_rotated[1][i], head_width=0.01)

plt.show()