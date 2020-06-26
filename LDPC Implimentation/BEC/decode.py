#!/usr/bin/env python3

# Decoder for BEC (binary erasure channel)

import numpy as np 
import matplotlib.pyplot as plt
import math

img = np.load('../binary_image.npy')
lx, ly = len(img), len(img[0])

code = np.loadtxt("../encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("../G.dat"), np.loadtxt("../H.dat")
n, k = G.shape[1], G.shape[0]

img = np.array(img).flatten() #vectorising the matrix
size = len(img)

row = [[] for _ in range(len(H))] # Analogous to an adjacency list reperesentation of a graph. For storing indices of non-zero entries (corresponding to a particular row or column) in H. 
col = [[] for _ in range(n)]

for i in range(len(H)):
	for j in range(n):
		if(H[i][j] == 1):
			row[i].append(j)
			col[j].append(i)

wc, wr = len(col[0]), len(row[0])

# Utility functions
def count_e(arr): # function for counting the no. of erasure symbols in a code block
    return np.count_nonzero(arr == -1)

def xor(arr, n):
	res = 0
	for i in range(len(arr)):
		if i != n:
			res += arr[i]
	return res % 2

def assign(arr, colIndex):
    bit = 0
    for i in col[colIndex]:
        if arr[i] != -1:
            bit = arr[i]
            break
    
    for i in col[colIndex]:
        arr[i] = bit

    return arr

def BEC(data, p): # channel simulation
	b = np.array(np.random.rand(len(data)) < p, dtype = np.int)
	ans = np.zeros(len(data))
	for i in range(len(data)):
		if(b[i]):
			ans[i] = -1
		else:
			ans[i] = data[i]
	return ans

def belief_prop(bits): # decoding
    L = H.copy()
    res = np.zeros(math.ceil(k*len(bits)/n))

    for i in range(int(len(bits)/n)):
        r = bits[n*i : n*(i+1)]

        for j in range(len(L)):
            L[j] = H[j]*r

        for _ in range(10): # no. of iterations
            # row operations
            for j in range(len(L)):
                index = 0
                cnt = 0
                for l in range(n):
                    if L[j][l] == -1:
                        index = l
                        cnt += 1
                
                if cnt == 1: # if exactly one variable in a row is erased
                    L[j][index] = xor(L[j], index)

            # column operations
            for j in range(n):
                if count_e(L[:, j]) != wc:
                    L[:, j] = assign(L[:, j], j)
                    for l in col[j]:
                        if L[l][j] != -1:
                            r[j] = L[l][j]
                            break            

            res[k*i : k*(i+1)] = r[0:k]

    return res


p = [0.1, 0.2, 0.3, 0.4, 0.5] # probability of erasure
BER = []

for i in p:
    print("For p =", i)
    r = BEC(code, i) # received bits. Here, '-1' is the erasure symbol.
    decod = belief_prop(r)
    error = (img != decod).sum()
    print("No. of incorrectly decoded bits:", error)
    BER.append(error/size)
    print("Bit Error rate:", error/size, "\n")

plt.plot(p, BER, marker = 'o')
plt.xlabel('p')
plt.ylabel('BER')
plt.savefig('../figs/BEC.png')
plt.show() 