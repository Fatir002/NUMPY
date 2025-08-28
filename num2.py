import numpy as np

# ======================
# 3D ARRAYS & INDEXING
# ======================
print("3D ARRAYS & INDEXING")
print("===================")

# Creating a 3D array: Layers[Rows[Columns]]
array_3d = np.array([
    [[1, 2, 3], [4, 5, 6]],       # Layer 0
    [[7, 8, 9], [10, 11, 12]],    # Layer 1
    [[13, 14, 15], [16, 17, 18]], # Layer 2
    [[19, 20, 21], [22, 23, 24]]  # Layer 3
])

print("Shape:", array_3d.shape)  # (4, 2, 3) - 4 layers, 2 rows, 3 columns
print("First layer:\n", array_3d[0])
print("Last layer:\n", array_3d[-1])  # Negative indexing works too!

# Slicing examples
print("First two layers:\n", array_3d[0:2])
print("Every other layer:\n", array_3d[0::2])
print("Reverse order:\n", array_3d[::-1])

# Multidimensional indexing: array[layer, row, column]
print("Element at [0, 0, 0]:", array_3d[0, 0, 0])
print("Elements at layers 0-1, row 0, column 0:", array_3d[0:2, 0, 0])

# ======================
# ARITHMETIC OPERATIONS
# ======================
print("\nARITHMETIC OPERATIONS")
print("====================")

# Scalar arithmetic (applied to each element)
print("Add 1 to all elements:\n", array_3d + 1)
print("Multiply all elements by 3:\n", array_3d * 3)
print("Square all elements:\n", array_3d ** 2)

# Mathematical functions
radii = np.array([1, 2, 3])
print("Areas of circles:", np.pi * radii ** 2)
print("Square roots:\n", np.sqrt(array_3d))
print("Rounded values:\n", np.round(array_3d))
print("Rounded to 2 decimals:", np.round(3.14159, 2))
print("Floor values:\n", np.floor(array_3d))
print("Ceiling values:\n", np.ceil(array_3d))

# ======================
# VECTOR OPERATIONS
# ======================
print("\nVECTOR OPERATIONS")
print("=================")

# Element-wise operations between arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

print("Vector addition:", array1 + array2)
print("Vector multiplication:", array1 * array2)
print("Element-wise exponentiation:", array1 ** array2)  # [1^4, 2^5, 3^6]

# ======================
# COMPARISON OPERATIONS
# ======================
print("\nCOMPARISON OPERATIONS")
print("=====================")

scores = np.array([12, 34, 54, 76, 98, 54, 98, 100])
print("Perfect scores:", scores == 100)
print("Passing scores:", scores >= 60)

# Filter and modify values
scores[scores < 60] = 0  # Set failing scores to 0
print("Adjusted scores:", scores)

# ======================
# BROADCASTING
# ======================
print("\nBROADCASTING")
print("============")

# Arrays with compatible shapes can be operated on together
array_2x4 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
array_1x4 = np.array([[1, 2, 3, 4]])

print("2x4 array shape:", array_2x4.shape)
print("1x4 array shape:", array_1x4.shape)
print("Broadcasted multiplication:\n", array_2x4 * array_1x4)

# ======================
# AGGREGATE FUNCTIONS
# ======================
print("\nAGGREGATE FUNCTIONS")
print("===================")

data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("Total sum:", np.sum(data))
print("Mean:", np.mean(data))
print("Standard deviation:", np.std(data))
print("Variance:", np.var(data))
print("Minimum value:", np.min(data))
print("Maximum value:", np.max(data))
print("Index of min value:", np.argmin(data))
print("Index of max value:", np.argmax(data))
print("Column sums:", np.sum(data, axis=0))  # Sum down columns
print("Row sums:", np.sum(data, axis=1))     # Sum across rows

# ======================
# FILTERING & CONDITIONAL LOGIC
# ======================
print("\nFILTERING & CONDITIONAL LOGIC")
print("============================")

scores = np.array([12, 34, 7, 54, 76, 98, 54, 9, 98, 100])
top_scores = scores[(scores >= 18) | (scores < 80)]
average_scores = scores[(scores >= 18) & (scores < 80)]
even_scores = scores[scores % 2 == 0]
odd_scores = scores[scores % 2 != 0]

print("Top scores:", top_scores)
print("Even scores:", even_scores)

# Using np.where to preserve array shape
graded_scores = np.where(scores > 80, scores, -1)
print("Graded scores (pass/fail):", graded_scores)

# ======================
# RANDOM NUMBER GENERATION
# ======================
print("\nRANDOM NUMBER GENERATION")
print("=======================")

rng = np.random.default_rng()  # Create a random number generator

# Generate random integers
print("Single die roll:", rng.integers(1, 7))  # 7 is excluded
print("Three random numbers:", rng.integers(low=1, high=101, size=3))
print("3x2 random array:\n", rng.integers(low=1, high=101, size=(3, 2)))

# Reproducible results with seed
rng_seeded = np.random.default_rng(seed=42)
print("Reproducible random numbers:", rng_seeded.integers(1, 100, size=3))

# Uniform distribution
print("Uniform random numbers:", np.random.uniform(low=1, high=10, size=3))

# Array manipulation
arr = np.array([1, 2, 3, 4, 5])
rng.shuffle(arr)  # Shuffle in place
print("Shuffled array:", arr)

# Random selection
print("Random choice:", rng.choice(arr))
print("Three random choices:", rng.choice(arr, size=3))
print("3x3 random choices:\n", rng.choice(arr, size=(3, 3)))