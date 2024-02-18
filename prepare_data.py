with open("google-books-common-words.txt") as file:
    text = file.read()
text_list = text.split()
words = [text_list[i].lower() for i in range(0, len(text_list), 2)]
print(len(words))

new_text = "\n".join(words)
with open("most_used_words.txt", "w") as file:
    file.write(new_text)
