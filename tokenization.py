#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# In[2]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])


# In[3]:


import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)',text)
print(result)


# In[4]:


result = re.split(r'([,.]|\s)',text)
print(result)


# In[5]:


result = [item for item in result if item.strip()]
print(result)


# In[6]:


text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)',text)
result = [item.strip() for item in result if item.strip()]
print(result)


# In[7]:


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])


# In[8]:


all_words = sorted(preprocessed)
vocab_size = len(all_words)
print(vocab_size)


# In[9]:


vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


# In[10]:


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# In[11]:


tokenizer = SimpleTokenizerV1(vocab)
text = """It's the last he painted, you know,
    Mrs.Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)


# In[12]:


print(tokenizer.decode(ids))


# In[13]:


t = "Hello, world!"
print(tokenizer.encode(t))


# In[14]:


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))


# In[15]:


for i , item in enumerate(list(vocab.items())[-5:]):
    print(item)


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[16]:


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int 
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# In[18]:


text1 = "Hello, do you lke tea?"
text2 = "In th sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1,text2))
print(text)


# In[19]:


tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))


# In[20]:


print(tokenizer.decode(tokenizer.encode(text)))


# ### Byte pair encoding

# In[21]:


get_ipython().system('pip install tiktoken')


# In[22]:


from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))


# In[25]:


tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
print(integers)


# In[26]:


strings = tokenizer.decode(integers)
print(strings)


# ## sliding window

# In[27]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))


# In[28]:


enc_sample = enc_text[50:]


# In[29]:


context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:       {y}")


# In[30]:


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context,"------>", desired)


# In[31]:


for i in range(1, context_size+1):
    context= enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context),"------->", tokenizer.decode([desired]))


# In[ ]:


import torch
from torch.utils.data import  Dataset, DataLoader

class GPTDatabseV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+ max_length]
            target_chunk = token_ids[i+1 : i+ max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    '''def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length + 1
        chunk = self.ids[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y'''
    


# In[1]:


def create_dataloader_v1(txt, batch_size =4, max_length =256, stride =128, shuffle = True,drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset =GPTDatabseV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader 


# In[35]:


with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()
dataLoader = create_dataloader_v1(raw_text, batch_size= 1, max_length=4, stride =1, shuffle= False)
data_iter = iter(dataLoader)
first_batch = next(data_iter)
print(first_batch)


# In[36]:


second_batch  = next(data_iter)
print(second_batch)


# In[38]:


# example (not modifying files)
tokens = list(range(10))      # tokens 0..9
max_length = 4
for stride in (1, 2, 4, 5):
    windows = [tokens[i:i+max_length] for i in range(0, len(tokens)-max_length, stride)]
    print("stride", stride, "->", windows)


# In[ ]:


#ex1
dataLoader = create_dataloader_v1(raw_text, batch_size= 1, max_length=2, stride =2, shuffle= False)
data_iter = iter(dataLoader)
first_batch = next(data_iter)
print(first_batch)


# In[ ]:


#ex2
dataLoader = create_dataloader_v1(raw_text, batch_size= 1, max_length=8, stride =2, shuffle= False)
data_iter = iter(dataLoader)
first_batch = next(data_iter)
print(first_batch)


# In[43]:


dataLoader = create_dataloader_v1(raw_text, batch_size= 8, max_length=4, stride =4, shuffle= False)
data_iter = iter(dataLoader)
inps , outs = next(data_iter)
print("Inputs:\n", inps)
print("Outputs:\n" , outs)


# ## creating token embeddings

# In[46]:


vocab_size =6 
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)


# In[47]:


print(embedding_layer(torch.tensor([3])))


# In[48]:


inputs_ids = torch.tensor([2,3,5,1])
print(embedding_layer(inputs_ids))


# In[49]:


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length =4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                                  stride= max_length, shuffle= False)
data_iter = iter(dataloader)
inputs, outputs = next(data_iter)
print("Token IDs:\n", inputs)
print("\n Inputs shape: \n", inputs.shape)


# In[51]:


token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


# In[52]:


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)


# In[54]:


input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)


# ![image.png](attachment:image.png)

# In[ ]:




