import re
# Takes in a .asm file from Zydis output and reformats the assembly for Xbyak
f = open("./asm/mkl16x64.asm", "r")
text = f.read()
for line in text.splitlines():
    noAddr = re.sub(r'0x\w+\s+', '', line)
    withParens = noAddr.replace(' ', '(', 1) + ');'
    output = re.sub(r'(\w)(mm)?word ptr', r'\1word', withParens)
    if output.count('(') == 0:
        output = output.replace(');', '();', 1)
    print(output)