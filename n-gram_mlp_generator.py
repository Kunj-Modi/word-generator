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
vocab_size = 27
learning_rate = 0.15

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

# layers
class Linear:
    def __init__(self, lay_in, lay_out, bias=True):
        self.weights = torch.randn((lay_in, lay_out), generator=g) / lay_in**0.5
        self.bias = torch.zeros(lay_out) if bias else None

    def __call__(self, X):
        self.out = X @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weights] + ([] if self.bias is None else [self.bias])


class Tanh:
    def __call__(self, X):
        self.out = torch.tanh(X)
        return self.out

    def parameters(self):
        return []


class BatchNormalization:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)

    def __call__(self, X):
        if self.training:
            meani = X.mean(0, keepdim=True)
            vari = X.var(0, keepdim=True)
        else:
            meani = self.mean
            vari = self.var
        Xhat = (X - meani) / torch.sqrt(vari + self.eps)
        self.out = self.gamma * Xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.mean = (1 - self.momentum) * self.mean + self.momentum * meani
                self.var = (1 - self.momentum) * self.var + self.momentum * vari
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


# creating model
g = torch.Generator().manual_seed(seed)
C = torch.randn((27, Cdim), generator=g)
layers = [
    Linear(Cdim*block_size, nhidden, bias=False), BatchNormalization(nhidden), Tanh(),
    Linear(        nhidden, nhidden, bias=False), BatchNormalization(nhidden), Tanh(),
    Linear(        nhidden, nhidden, bias=False), BatchNormalization(nhidden), Tanh(),
    Linear(        nhidden, vocab_size, bias=False), BatchNormalization(vocab_size),
]
with torch.no_grad():
    layers[-1].gamma *= 0.1
    for layer in layers[::3]:
        layer.weights *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for parm in parameters:
    parm.requires_grad = True
print("total parameters in model:", sum(parm.nelement() for parm in parameters))

# train model
# for debuging
# ------------------------------------------ #
# loss_list = []
# ud = []
# ------------------------------------------ #
bg = torch.Generator().manual_seed(seed)
for i in range(epochs):
    # batch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=bg)
    X, Y = Xtr[ix], Ytr[ix]
    # forward
    X = C[X]
    X = X.view(X.shape[0], -1)
    for layer in layers:
        X = layer(X)
    loss = F.cross_entropy(X, Y)
    # backward
    for layer in layers:
        layer.out.retain_grad()
    for parm in parameters:
        parm.grad = None
    loss.backward()
    # update
    lr = 0.13
    for parm in parameters:
        parm.data += -lr*parm.grad
    # for debuging
    # ---------------------------------------------------------------------------------------- #
    # loss_list.append(loss.item())
    # with torch.no_grad():
    #     ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])
    # if i >= 1000:
    #     break
    # ---------------------------------------------------------------------------------------- #
print(loss.item())

# for debuging
# ---------------------------------------------------------------------------------------- #
# plt.plot(loss_list[::])
# plt.figure(figsize=(20, 4))
# legends = []
# for i,p in enumerate(parameters):
#   if p.ndim == 2:
#     plt.plot([ud[j][i] for j in range(len(ud))])
#     legends.append('param %d' % i)
# plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
# plt.legend(legends);
# ---------------------------------------------------------------------------------------- #

# validating model
@torch.no_grad()
@torch.no_grad()
def loss(split):
    X, Y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte)
    }[split]
    X = C[X]
    X = X.view(X.shape[0], -1)
    for layer in layers:
        X = layer(X)
    loss = F.cross_entropy(X, Y)
    return loss.item()

for layer in layers[1::3]:
    layer.training = False
print(loss('train'))
print(loss('val'))
print(loss('test'))
for layer in layers[1::3]:
    layer.training = True

# sampling from model
@torch.no_grad()
def sample():
    g = torch.Generator().manual_seed(seed)
    for _ in range(10):
        out = ""
        st = [0]*block_size
        while True:
            X = C[torch.tensor(st)]
            X = X.view(-1, Cdim*block_size)
            for layer in layers:
                X = layer(X)
            prob = F.softmax(X, dim=1)
            ix = torch.multinomial(prob, num_samples=1, generator=g).item()
            out += itoa[ix]
            st = st[1:] + [ix]
            print(itoa[ix], end="")
            if(ix==0): break
        print()

for layer in layers[1::3]:
    layer.training = False
sample()
for layer in layers[1::3]:
    layer.training = True
