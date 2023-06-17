from math import comb

for i in range(5, 30):
    print(i)
    if i%2 == 0:
        print(comb(int(i), int(i/2)))
    else:
        print(comb(i, int((i-1)/2)))