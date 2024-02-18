import torch
import random

block_size = 3
Cdim = 10
seed = 8566354565

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
C = torch.randn((27, Cdim), generator=g)
W1 = torch.randn((Cdim*block_size, 200), generator=g)
b1 = torch.randn((200), generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn((27), generator=g)
parameters = [C, W1, b1, W2, b2]
for parm in parameters:
    parm.requires_grad = True
print("total parameters in model:", sum(parm.nelement() for parm in parameters))

# train model
bg = torch.Generator().manual_seed(seed)
for i in range(100000):
    # batch
    ix = torch.randint(0, Xtr.shape[0], (50,), generator=bg)
    # forward
    h = torch.tanh(C[Xtr[ix]].view(-1, Cdim*block_size) @ W1 + b1)
    logits = h @ W2 + b2
    loss = torch.nn.functional.cross_entropy(logits, Ytr[ix])
    # backward
    for parm in parameters:
        parm.grad = None
    loss.backward()
    # update
    lr = 0.1 if i<50000 else 0.01
    for parm in parameters:
        parm.data += -lr*parm.grad
    if i%5000==0: print(loss.item())
print(loss.item())

# validating model
h = torch.tanh(C[Xtr].view(-1, Cdim*block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = torch.nn.functional.cross_entropy(logits, Ytr)
print(loss.item())
h = torch.tanh(C[Xdev].view(-1, Cdim*block_size) @ W1 + b1)
logits = h @ W2 + b2
loss = torch.nn.functional.cross_entropy(logits, Ydev)
print(loss.item())

# sampling from model
g = torch.Generator().manual_seed(seed)
for _ in range(10):
    out = ""
    st = [0]*block_size
    while True:
        h = torch.tanh(C[torch.tensor(st)].view(-1, Cdim*block_size) @ W1 + b1)
        logits = h @ W2 + b2
        prob = torch.nn.functional.softmax(logits, dim=1)
        ix = torch.multinomial(prob, num_samples=1, generator=g).item()
        out += itoa[ix]
        st = st[1:] + [ix]
        print(itoa[ix], end="")
        if(ix==0): break
    print()
