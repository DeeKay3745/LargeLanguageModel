#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch 
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], #your (x1)
    [0.55, 0.87, 0.66], #journey (x2)
    [0.57, 0.85, 0.64], #starts (x3)
    [0.22, 0.58, 0.33], #with (x4)
    [0.77, 0.25, 0.10], #one (x5)
    [0.05, 0.80, 0.55] #step (x6)
])


# In[4]:


query  = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
#print(attn_scores_2)
for i, x_i in enumerate(inputs):
    #print(x_i)
    attn_scores_2[i] = torch.dot(x_i, query)
    #print(attn_scores_2[i])
print(attn_scores_2)


# In[5]:


attn_weights_2_tmp = attn_scores_2 /attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# In[6]:


def softmax_naive(x):
    return torch.exp(x)/ torch.exp(x).sum(dim =0 )
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())


# In[8]:


attn_weights_2 = torch.softmax(attn_scores_2, dim =0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())


# In[9]:


query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)


# 
# 3.3.2 Computing attention weights for all input tokens
# Generalize to all input sequence tokens:
# 
#     Above, we computed the attention weights and context vector for input 2 (as illustrated in the highlighted row in the figure below)
#     Next, we are generalizing this computation to compute all attention weights and context vectors
# ![image.png](attachment:image.png)
#     - (Please note that the numbers in this figure are truncated to two digits after the decimal point to reduce visual clutter; the values in each row should add up to 1.0 or 100%; similarly, digits in other figures are truncated)
# 
#     - In self-attention, the process starts with the calculation of attention scores, which are subsequently normalized to derive attention weights that total 1
#     These attention weights are then utilized to generate the context vectors through a weighted summation of the inputs
# ![image-2.png](attachment:image-2.png)
#     Apply previous step 1 to all pairwise elements to compute the unnormalized attention score matrix:
# 
# 

# In[10]:


attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i , x_j)
print(attn_scores)


# In[11]:


attn_scores = inputs @ inputs.T
print(attn_scores)


# In[12]:


attn_weights = torch.softmax(attn_scores , dim =-1)


# In[13]:


attn_weights


# In[14]:


row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print(row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))


# In[15]:


all_context_vecs = attn_weights @ inputs 
print(all_context_vecs)


# In[16]:


all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


# In[17]:


print("Previous 2nd context vector:", context_vec_2)


# ## Implementing self-attention with trainable weights

# ![image.png](attachment:image.png)

# query (q), key(k), value(v)

# In[18]:


x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2


# In[19]:


torch.manual_seed(123)
w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


# In[20]:


import torch

p_true = torch.nn.Parameter(torch.rand(3), requires_grad=True)
p_false = torch.nn.Parameter(torch.rand(3), requires_grad=False)
x = torch.tensor([1.0, 2.0, 3.0])

loss = (p_true * x).sum() + (p_false * x).sum()
loss.backward()

print("grad true:", p_true.grad)   # non-None
print("grad false:", p_false.grad) # None\


# In[21]:


query_2 = x_2 @ w_query
key_2 = x_2 @ w_key
value_2 = x_2 @ w_value
print(query_2)


# In[22]:


keys = inputs @ w_key
values = inputs @ w_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)


# In[23]:


keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print(attn_scores_22)


# In[24]:


attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)


# In[25]:


d_k  = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k **0.5, dim= -1)
print(attn_weights_2)


# In[26]:


context_vec_2 = attn_weights_2 @ values
print(context_vec_2)


# In[27]:


import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T #Omega
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim =-1)
        context_vec = attn_weights @ values
        return context_vec


# In[28]:


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in,d_out)
print(sa_v1(inputs))


# In[37]:


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T #Omega
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim =-1)
        context_vec = attn_weights @ values
        return context_vec


# In[38]:


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))


# Applying a causal attention mask 
# ![image.png](attachment:image.png)

# In[39]:


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5 ,dim =-1)
print(attn_weights)


# In[40]:


context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)


# In[41]:


masked_simple = attn_weights * mask_simple
print(masked_simple)


# In[42]:


row_sums = masked_simple.sum(dim = -1, keepdim =True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
print(row_sums)


# In[44]:


mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)


# In[45]:


attn_weights = torch.softmax(masked/ keys.shape[-1]**0.5, dim=1)
print(attn_weights)


# Masking additional weights with dropout

# In[51]:


torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
print(dropout(example))   # training mode: zeros and 2s



# In[52]:


torch.manual_seed(123)
print(dropout(attn_weights))


# In[53]:


batch = torch.stack((inputs, inputs), dim =0)
print(batch.shape)


# In[55]:


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries =self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim= -1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


# In[57]:


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in , d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)


# In[58]:


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList(
            [ CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)
            ]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim =-1)


# In[59]:


torch.manual_seed(123)
context_length = batch.shape[1] #this is the number of tokens
d_in, d_out =3,2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
                  


# In[61]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out// num_heads
        self.W_query = nn.Linear(d_in,d_out, bias= qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias =qkv_bias)
        self.W_value = nn.Linear(d_in,d_out, bias =qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal= 1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5 , dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        return context_vec


# In[62]:


a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                    [[0.0772, 0.3565, 0.1479, 0.5331],
                     [0.4066, 0.2318, 0.4545, 0.9737],
                     [0.4606, 0.5159, 0.4220, 0.5786]]]])


# In[63]:


print(a@ a.transpose(2,3))


# In[64]:


first_head = a[0,0,:,:]
first_res = first_head@ first_head.T
print("First head:\n", first_res)

second_head = a[0,1,:,:]
second_res = second_head @ second_head.T
print("Second head:\n", second_res)


# In[65]:


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out =2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# In[ ]:





# In[ ]:





# 
