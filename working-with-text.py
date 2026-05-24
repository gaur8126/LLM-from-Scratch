with open("the-verdict.txt", 'r', encoding='utf-8') as f:
    text = f.read()

print(f"{text[:99]} \n Total number of characters: {len(text)}")
