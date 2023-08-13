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
# N = torch.zeros((27,27), dtype=torch.int32)

# input
xs = []

# output labels
ys = []

# For now, let's just use the first word as input
for w in words:
    chs = ['.'] + list(w) + ['.'] 
    for ch1, ch2 in zip(chs, chs[1:]):
        # convert characters to integers
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# input tensor
xs = torch.tensor(xs)

# labels tensor 
ys = torch.tensor(ys)

# number of examples
num = xs.nelement()
print('# examples: ', num)

# seed a psudeo random number generator to ensure reproducibility for the purposes of learning
g = torch.Generator().manual_seed(2147483647)

# initialize network (1 hidden layer) weights randomly from normal distribution. Note: default is normal distribution
# In W, the first column represents the activations of each input on the first neuron. The number of neurons is given by the second dimension. 
# Neurons here are simple - linear, no bias
W = torch.randn((27, 27), generator=g, requires_grad=True)

alpha = 50
gd_Iterations = 200

############ 
# Gradient descent

for k in range(gd_Iterations):
    
    ############ 
    # Forward pass - run model on all inputs to get predictions
    #
    # Transform our inputs (i.e. a character) into a form that can be fed into the input layer (i.e. one hot encodings) 
    # cast one hot encodings (int64) into float (32)
    xenc = F.one_hot(xs, num_classes=27).float()

    # display xenc and yenc one hot encodings
    """ plt.imshow(xenc)
    plt.imshow(yenc)
    plt.show() """

    # note: '@' is pytorch matrix multiplication
    # xenc @ W gives us the firing rate of the inputs on each the neurons in the first hidden layer
    # thus (xenc @ W)[3,13] gives us the firing rate of the 3rd input on the 13th neuron (as the dot product)
    # 'logits' are log counts. We interpret the random values from the normal distribution used to initialize W as log counts. Basically, it's a transformation of the normal distribution to a range that with the properties of a probability distribution (i.e. the values are positive and sum to 1). To do this, we element-wise exponentiate and normalize them. This is called the Softmax activation function. 
    logits = xenc @ W 
    counts = logits.exp() # exponentiate to get counts
    probs = counts / counts.sum(1, keepdim=True) # normalize each of the rows to get a probability distribution 
    # This has the effect of simple neural network with 1 hidden layer with no bias and no non-linearity
    # Applying a single input gives us the probability distribution of each of the possible outputs

    ############ 
    # calculate loss (negative log likelihood)
    # note arange() is similar to range()
    # probs[torch.arange(5), ys] gives us the probability assigned by our current model to each of the outputs for each of the 5 inputs
    # use the average loss 
    loss = -probs[torch.arange(num), ys].log().mean()
    loss += 0.01*(W**2).mean() # Regulazation term for smoothing (preventing overfitting)
    print(loss.item())

    ############ 
    # backward pass
    W.grad = None # set gradients to zero
    loss.backward()
    
    ############ 
    # update model parameters
    W.data += -alpha *W.grad
 
  # Sample from the Neural Net
g = torch.Generator().manual_seed(2147483647)
 
for i in range(5):
    out=[]
    ix = 0
    while True:
        # p = P[ix]
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W # predict log counts
        counts = logits.exp() # counts
        p = counts / counts.sum(1, keepdim=True) # probabilities for next character
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0: 
            break
    print(''.join(out))