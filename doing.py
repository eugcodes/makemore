import torch
import numpy
import matplotlib.pyplot as plt

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

# define a dictionary to count bigram occurrences
# b = {}

# create a 28x28 array for improved efficiency
N = torch.zeros((27,27), dtype=torch.int32)

for w in words:
    # pre-pend special start character and append end character
    chs = ['.'] + list(w) + ['.'] 
    # identify all bigrams in words
    for ch1, ch2 in zip(chs, chs[1:]):
        # convert characters to integers
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # count bigram occurences in 2D tensor
        N[ix1, ix2] += 1
        
        # bigram = (ch1, ch2)
        # count bigram frequencies
        # b[bigram] = b.get(bigram, 0) + 1

# print sorted bigram frequencies
# print(sorted(b.items(), key = lambda kv: -kv[1]))

#print(N[:, 0])

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')

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
#p = N[0].float()
#p = p / p.sum()

# seed a psudeo random number generator
g = torch.Generator().manual_seed(2147483647)

# example: generate 3 numbers and normalize them
# p = torch.rand(3, generator=g)
 
# print(torch.multinomial(p, num_samples=20, replacement=True, generator=g))

# convert bigram counts tensor to a tensor of probabilities
# added 1 to "smooth" the model i.e. to avoid 0 count, which might result in infinite values for loss function.
P = (N+1).float() 

# normalize each elemnt of each row by the sum of the row.
# note the 1st arg is the dimension to reduce
# note "broadcasting rules" allos below operation if starting @ trailing dimension, dimension sizes are equal, one is 1, or one doesn't exist
# 27 27
# 27 1  ---> operation is braodcastable
P = P / P.sum(1, keepdim = True)

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

for w in ["andrejq"]:
    chs = ['.'] + list(w) + ['.'] 
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        n += 1
        
        # log of a product is the sum of the logs
        log_likelihood += logprob
        
        # use nomalized negative log likelihood as loss function
        nll = -log_likelihood/n
        
print(nll)
        
        
    

