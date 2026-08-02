import torch
print(torch.__version__)
import tiktoken

from torch.utils.data import Dataset, DataLoader

# 2.6 Data sampling with a sliding window
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Stride is by how many places we are moving the window
def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

with open("text.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Batch Size 1 makes tokens to overlap causing a possible data overfitting
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=4,
    max_length=4,
    stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
second_batch = next(data_iter)
print(first_batch)
print(second_batch)


# 2.7 Creating token embeddings
## Input text -> tokenized text -> token ids -> token embeddings -> positional embeddings -> Input embedddings
inputs_id = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123) # In order to get the same results for the layer
embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # The idea is that we start with random numbers and then these numbers get optimized 
# Embedding layer make lookups easier (relates to matrix multiplication), optmized during the LLM training
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3]))) #index 3 corresponding to the matrix row
print(embedding_layer(torch.tensor([2]))) #index 2 corresponding to the matrix row


# 2.8 Encoding word positions
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)
data_iter = iter(dataloader)
inputs, targets= next(data_iter)

print("Token IDs:", inputs)
print("Inputs shape:", inputs.shape)

# We`ve converted token ids to token embeddings
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape) # 256 dimension

# Positional Embedding Layer
context_length = max_length # Max Length is how many inputs are supported
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length)) # Torch arange makes a placeholder tensor with max lendgth
print(pos_embeddings)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

