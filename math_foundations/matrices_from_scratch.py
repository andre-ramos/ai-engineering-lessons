import numpy as np

A = np.array([
    [3,-5, 0],
    [1, 6, 7]
])


print(f"Matrix A:\n{A}")
print(f"Shape: {A.shape}") # (rows, columns)

# 
#Addition: Matrices must have the same shape. We add them element-wise.
B = np.array([2,0,2])
print(f"Matrix A+B:\n{A+B}\n")

#Scalar Multiplication: Multiplying a matrix by a single number scales every element.  (A is 2x3, C is 3x2 -> Result is 2x2)
B = np.array([2,0,2])
print(f"Matrix A @ B:\n{A@B}\n")


# A scaling matrix that doubles the x-axis and halves the y-axis
S = np.array([[2, 0],
              [0, 0.5]])

# Each row is a point: [x, y]
# Let's use a square: (0,0), (1,0), (1,1), (0,1)
points = np.array([[0, 0],
                   [1, 0],
                   [1, 1],
                   [0, 1]])

# 3. Apply the transformation
# We transpose the points to shape (2, 4) so we can do S @ points_T
# Or we can do points @ S.T (which keeps the points as rows)
transformed_points = points @ S.T

print("Original Points:\n", points)
print("\nTransformed Points (x doubled, y halved):\n", transformed_points)

#A rotation matrix "spins" a vector around the origin. This is how "Attention" mechanisms in Transformers work—they project vectors into different spaces (Query, Key, Value) to find relationships.
theta = np.radians(45) # Rotate by 45 degrees
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])