# Chapter 3: Self attention is a mechanism to parse data

# Simplified self-attention: Each token has a importance weight (attention scores), so it references all the input tokens
# Recurring Neural Networks, when generating text, usually translate memmorize the state or a summary, the downside is that you can't look at the full sentence, only the previously tokens generated
# Dot product is a way to determine similarity bewtween vectors:  the dot product is the product of the magnitudes of the two vectors and the cosine of the angle between them. Dot product cares about length × alignment.
# Cosine similarity cancels out the length, leaving only alignment.

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

input_query = inputs[1]
res = 0.

for idx, element in enumerate(inputs[0]):
    print(inputs[0][idx])