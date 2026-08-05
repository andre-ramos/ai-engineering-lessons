# Chapter 3: Self attention is a mechanism to parse data

# Simplified self-attention: Each token has a importance weight (attention scores), so it references all the input tokens
# Recurring Neural Networks, when generating text, usually translate memmorize the state or a summary, the downside is that you can't look at the full sentence, only the previously tokens generated
# Dot product is a way to determine similarity bewtween vectors:  the dot product is the product of the magnitudes of the two vectors and the cosine of the angle between them. Dot product cares about length × alignment.
# Cosine similarity cancels out the length, leaving only alignment.

import torch

# Inputs
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2) #Highlighting the computation that happends

# Attention weights
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

def softmax_naive(x): #simple version
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Sophisticated version of softmax: ",attn_weights_2 ) #)

# Output vector (context vector)
query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)


# Simple self attention mechanism without trainable weights
attn_scores = torch.empty(6,6)

for i, x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j) # These for loops are slow, we could use matrix multiplication instead

print(attn_scores) 

# Using matrix multiplication
attn_scores = inputs @ inputs.T

print(attn_scores)

# Lets normalize the weights

attn_weights = torch.softmax(attn_scores, dim=1)
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)