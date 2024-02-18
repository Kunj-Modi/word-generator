import torch

# import data
with open("most_used_words.txt") as file:
    words = file.read().split()

# creating input and outout
X, Y = [], []
for word in words:
    X.append(".")
    Y.append(word[0])
    for x, y in zip(word, word[1:]+"."):
        X.append(x)
        Y.append(y)

for x, y in zip(X[:5], Y):
    print(x, "->", y)

chars = sorted(set(".".join(words)))
atoi = {a:i for i,a in enumerate(chars)}
itoa = {i:a for i,a in enumerate(chars)}

Xi = torch.tensor(list(map(lambda x: atoi[x], X)))
Yi = torch.tensor(list(map(lambda x: atoi[x], Y)))
num = Xi.nelement()
print(num)

Xenc = torch.nn.functional.one_hot(Xi, num_classes=27).float()
g = torch.Generator().manual_seed(37469124)
W = torch.randn((27,27), generator=g, requires_grad=True)


# Gradient decent
for _ in range(100):
    # Forward pass
    B = (Xenc @ W).exp()
    probs = B/B.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), Yi].log().mean() + 0.1*(W**2).mean()
    if _%10==0: print(loss.item())
    # Backward pass
    W.grad = None
    loss.backward()
    # Update
    W.data += -50 * W.grad

# sampling from model
g = torch.Generator().manual_seed(37469124)
ix = 0
for _ in range(10):
    while True:
        prob = probs[ix]
        ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
        print(itoa[ix], end="")
        if(ix==0): break
    print()
