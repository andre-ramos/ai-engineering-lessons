class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dimension = len(self.components)

    def __add__(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")    
        return Vector([a + b for a, b in zip(self.components, other.components)])
    
    def __sub__(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")    
        return Vector([a - b for a, b in zip(self.components, other.components)])
    
    def dot(self, other):
        # dot product is the sum of the products of the corresponding components
        # Attention scores in transformers
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")    
        return sum(a * b for a, b in zip(self.components, other.components))
    
    def magnitude(self):
        # magnitude is the square root of the sum of the squares of the components
        return sum(a ** 2 for a in self.components) ** 0.5
    
    def normalize(self):
        # normalization is the vector divided by its magnitude
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector([a / mag for a in self.components])
    
    def cosine_similarity(self, other):
        # cosine similarity is the dot product divided by the product of the magnitudes
        # Same direction:      a · b > 0  (similar)
        # Perpendicular:       a · b = 0  (unrelated)
        # Opposite direction:  a · b < 0  (dissimilar)
        # Used in RAG (Retrieval-Augmented Generation) to measure the similarity between query and document vectors
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")    
        return self.dot(other) / (self.magnitude() * other.magnitude())
    
    def __repr__(self):
        return f"Vector({self.components})"

a = Vector([1, 2, 3])
b = Vector([4, 5, 6])
c = Vector([1, 0, 0])
d = Vector([0, 1, 0])
e = Vector([0, 0, 1])

print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"c · d = {c.dot(d)}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity a and b = {a.cosine_similarity(b):.4f}")
print(f"cosine similarity c and d = {c.cosine_similarity(d):.4f}")
print(f"a.normalize() = {a.normalize()}")