import torch 
from torch.utils.data import Dataset, DataLoader
import tiktoken 

# loading text data 
with open("the-verdict.txt", 'r', encoding='utf-8') as f:
    raw_text = f.read()

# Prepare dataset 
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride): #Uses a sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    

# data loader 
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last= True, 
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size = batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


dataloader = create_dataloader_v1(raw_text, batch_size=8)

data_iter = iter(dataloader)
inputs, targets = next(data_iter) # shape will be [8,256]


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(token_embedding_layer) , shape - [50257, 256]

token_embeddings = token_embedding_layer(inputs) # shape - [8,256,256]

context_length = 256
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embedding = pos_embedding_layer(torch.arange(context_length)) # shape -[256,256]


input_embeddings = token_embeddings + pos_embedding # shape [8, 256, 256]

print(input_embeddings)