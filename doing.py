import torch
import numpy
import matplotlib.pyplot as plt

# load words from names.txt
words = open('names.txt', 'r').read().splitlines()

# create lookup table for characters to integers
chars = list(sorted(list(set(''.join(words))))) 

# Dictionary mapping characters to integers
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

# Dictionary mapping integers to characters  
itos = {i:s for s, i in stoi.items()}

# define a dictionary to count bigram occurrences
# b = {}

# create a 28x28 array for improved efficiency
N = torch.zeros((28,28), dtype=torch.int32)

for w in words:
    # pre-pend special start character and append end character
    chs = ['<S>'] + list(w) + ['<E>'] 
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

# print(N)

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')

# Display figure of bigram occurences as a matrix, with bigram and # occurences in each cell
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.show()


        
         

