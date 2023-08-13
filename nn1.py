import torch
import numpy
import matplotlib.pyplot as plt
import torch.nn.functional as F

# load words from names.txt
words = open('names.txt', 'r').read().splitlines()

# create lookup table for characters to integers
chars = list(sorted(list(set(''.join(words))))) 

# Dictionary mapping characters to integers
# stoi['<S>'] = 26
# stoi['<E>'] = 27
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

# Dictionary mapping integers to characters  
itos = {i:s for s, i in stoi.items()}

# create a 27x27 array for improved efficiency
N = torch.zeros((27,27), dtype=torch.int32)

xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.'] 
    for ch1, ch2 in zip(chs, chs[1:]):
        # convert characters to integers
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# input tensor
xs = torch.tensor(xs)

# label tensor 
ys = torch.tensor(ys)

# seed a psudeo random number generator to ensure reproducibility for the purposes of learning
g = torch.Generator().manual_seed(2147483647)

# initialize weights randomly from normal distribution
# note: '@' is pytorch  matrix multiplication
# In W, the first dimension represents the activations of each input on each of the neurons. The number of neurons is given by the second dimension.
# Neurons here are simple - no bias, no non-linearity
W = torch.randn((27, 27), generator=g)

############ 
# Forward pass - run model on all inputs to get predictions
#
# create one hot encodings for xs 
# cast one hot encodings (int64) into float (32)
xenc = F.one_hot(xs, num_classes=27).float()

# 'logits' are log counts. We interpret the random values from the normal distribution used to initialize W as log counts.
logits = xenc @ W 

# Element-wise exponentiate numbers to tranform values into positive values and normalize to obtain probabilities
# Softmax function
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)

# calculate loss
# note arange() is similar to range(), 
# using the average loss (i.e. normalizing by the number of samples)
loss = -probs[torch.arange(5), ys].log().mean()

# backward pass

# update model parameters



yenc = F.one_hot(ys, num_classes=27).float()

# display one hot encodings
""" plt.imshow(xenc)
plt.imshow(yenc)
plt.show() """
      
# print(N[:, 0])

""" plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues') """

# Display figure of bigram occurences as a matrix, with bigram and # occurences in each cell
# for i in range(27):
"""     for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.show()
 """
 
# calculate first row of bigram probabilities
""" p = N[0].float()
p = p / p.sum()
 """

# example: generate 3 numbers and normalize them
# p = torch.rand(3, generator=g)
 
# print(torch.multinomial(p, num_samples=20, replacement=True, generator=g))

# convert bigram counts tensor to a tensor of probabilities
# added 1 to "smooth" the model i.e. to avoid 0 count, which might result in infinite values for loss function.
P = (N+1).float() 

# normalize each elemnt of each row by the sum of the row.
# note the 1st arg is the dimension to reduce
# note "broadcasting rules" allows below operation if, starting at trailing dimension, dimension sizes are equal, one is 1, or one doesn't exist
# 27 27
# 27 1  ---> operation is broadcastable
P /= P.sum(1, keepdim = True)

# iteratively append another character based on the bigram distribution of the last character until '.' is appended
names = []
for i in range(10):
    out=[]
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0: 
            break
    print(''.join(out))
    names.append(''.join(out))
 
# Evaluate using log(likelihood)
log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.'] 
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        n += 1
        
        # log (abc) = log(a) + log (b) + log (c)
        log_likelihood += logprob
        
        # use nomalized negative log likelihood as loss function
        nll = -log_likelihood/n
        
        
    

