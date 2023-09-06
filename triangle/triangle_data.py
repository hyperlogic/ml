#
# Generates training data for triangle MLP.
#
# The triangle is an equalateral triangle centered at the origin
# data consists of:
#   * a 2d position within a (-1, 1) square
#   * a boolean if 2d position is inside of the triangle.
#
import math
import numpy as np

# corners of the equilateral triangle
theta = (2 * math.pi) / 3
corners = [[math.cos(i * theta), math.sin(i * theta)] for i in range(3)]

# midpoints of the triangle along with normal vectors pointing inward.
offset = math.pi / 3
seg_len = -math.cos(1 * theta)
midpoints = [[seg_len * math.cos(i * theta + offset), seg_len * math.sin(i * theta + offset)] for i in range(3)]
normals = [corners[(i + 2) % 3] for i in range(3)]

def is_inside_triangle(p):
    for i in range(3):
        d = p - midpoints[i]
        if np.dot(d, normals[i]) <= 0:
            return False
    return True

def generate_data(num_items):
    points = np.array([np.random.uniform(-1, 1, num_items),np.random.uniform(-1, 1, num_items)]).transpose()
    inside = [is_inside_triangle(p) for p in points]
    return points, inside

points, inside = generate_data(10)

for i in range(len(points)):
    print(f"{points[i]} => {inside[i]}")
