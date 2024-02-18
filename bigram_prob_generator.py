import torch
import matplotlib.pyplot as plt

with open("most_used_words.txt") as file:
    words = file.read().split()

chars = sorted(set(".".join(words)))
atoi = {a:i for i,a in enumerate(chars)}
itoa = {i:a for i,a in enumerate(chars)}

N = torch.zeros([27, 27], dtype=torch.int32) + 1 # add 1 to normalize

for word in words:
    for x, y in zip("."+word, word+"."):
        N[atoi[x]][atoi[y]] += 1

probs = N / N.sum(1, keepdim=True)

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itoa[i] + itoa[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis('off')

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

# calculate loss
log_likelihood = 0.0
n = 0
for word in words:
    for x, y in zip(word, word[1:]+"."):
        prob = probs[atoi[x]][atoi[y]]
        n += 1
        log_likelihood += torch.log(prob)

print(log_likelihood)
print("loss =", -log_likelihood/n)
