import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

block_size = 3
batch_size = 50
Cdim = 10
seed = 8566354565
epochs = 100000
nhidden = 200

with open("most_used_words.txt") as file:
    words = file.read().split()

chars = sorted(set(".".join(words)))
atoi = {a:i for i,a in enumerate(chars)}
itoa = {i:a for i,a in enumerate(chars)}

# Preparing data
def prepare_dataset(words):
    X, Y = [], []
    for word in words:
        st = [0] * block_size
        for ch in word+".":
            X.append(st)
            Y.append(atoi[ch])
            st = st[1:] + [atoi[ch]]
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y    

random.seed(77)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = prepare_dataset(words[:n1])
Xdev, Ydev = prepare_dataset(words[n1:n2])
Xte, Yte = prepare_dataset(words[n2:])

# parameters of model
g = torch.Generator().manual_seed(seed)
C = torch.randn((27, Cdim),                  generator=g)
W1 = torch.randn((Cdim*block_size, nhidden), generator=g) * 0.15 #(5/3)/((Cdim*block_size)**0.5) # kaiming init
# b1 = torch.randn((nhidden),                  generator=g) * 0.01
W2 = torch.randn((nhidden, 27),              generator=g) * 0.01
b2 = torch.randn((27),                       generator=g) * 0
bngain = torch.ones(1, nhidden)
bnbias = torch.zeros(1, nhidden)
bnmean = torch.zeros(1, nhidden)
bnstd = torch.zeros(1, nhidden)
parameters = [C, W1, W2, b2, bngain, bnbias]
for parm in parameters:
    parm.requires_grad = True
print("total parameters in model:", sum(parm.nelement() for parm in parameters))

# train model
loss_list = []
bg = torch.Generator().manual_seed(seed)
for i in range(epochs):
    # batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=bg)
    # forward
    h_mul = C[Xtr[ix]].view(-1, Cdim*block_size) @ W1 #+ b1
    # batch normalization
    bnmeani = h_mul.mean(0, keepdim=True)
    bnstdi = h_mul.std(0, keepdim=True)
    h_mul = bngain * (h_mul - bnmeani) / bnstdi + bnbias
    with torch.no_grad():
        bnmean = 0.999*bnmean + 0.001*bnmeani
        bnstd = 0.999*bnstd + 0.001*bnstdi
    h = torch.tanh(h_mul)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    # backward
    for parm in parameters:
        parm.grad = None
    loss.backward()
    # update
    lr = 0.1 if i<50000 else 0.01
    for parm in parameters:
        parm.data += -lr*parm.grad
    loss_list.append(loss.item())
print(loss.item())
plt.plot(loss_list[::1000])

# validating model
@torch.no_grad()
def loss(split):
    X, Y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte)
    }[split]
    h_mul = C[X].view(-1, Cdim*block_size) @ W1 #+ b1
    h_mul = bngain * (h_mul - bnmean) / bnstd + bnbias
    h = torch.tanh(h_mul)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    return loss.item()

print(loss('train'))
print(loss('val'))
print(loss('test'))

# sampling from model
@torch.no_grad()
def sample():
    g = torch.Generator().manual_seed(seed)
    for _ in range(10):
        out = ""
        st = [0]*block_size
        while True:
            h_mul = C[torch.tensor(st)].view(-1, Cdim*block_size) @ W1 #+ b1
            h_mul = bngain * (h_mul - bnmean) / bnstd + bnbias
            h = torch.tanh(h_mul)
            logits = h @ W2 + b2
            prob = F.softmax(logits, dim=1)
            ix = torch.multinomial(prob, num_samples=1, generator=g).item()
            out += itoa[ix]
            st = st[1:] + [ix]
            print(itoa[ix], end="")
            if(ix==0): break
        print()
sample()
