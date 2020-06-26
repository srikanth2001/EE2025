# SISO decoder for an AWGN channel using belief propagation/message passing algorithm and hard decoder using Gallagher-A decoding algorithm.

import numpy as np
import matplotlib.pyplot as plt 
import time, math

T = 1e-6
f_s, f_c = 50e6, 2e6
n1 = int(T * f_s)
img = np.load('../binary_image.npy')
lx, ly = len(img), len(img[0])

code = np.loadtxt("../encoded_bits.dat")
code_len = len(code)
G, H = np.loadtxt("../G.dat"), np.loadtxt("../H.dat")

n, k = G.shape[1], G.shape[0]
M = 2

img = np.array(img).flatten() #vectorising the matrix
size = len(img)

row = [[] for _ in range(len(H))] # Analogous to the adjacency list representation of the Tanner graph. For storing indices of non-zero entries (corresponding to a particular row or column) in H 
col = [[] for _ in range(n)]

for i in range(len(H)):
	for j in range(n):
		if(H[i][j] == 1):
			row[i].append(j)
			col[j].append(i)

wr = len(row[0])

def modulate(b1, Eb, i): # BPSK modulation function
    t = np.linspace((i-1)*T, i*T, int(T*f_s))
    res = np.zeros(len(t))
    if(b1 == 0):    
        for j in range(len(t)):        
            res[j] = np.cos(2*np.pi*f_c*t[j])
    
    if(b1 == 1):    
        for j in range(len(t)):
            res[j] = -np.cos(2*np.pi*f_c*t[j])

    return res

def WGN(variance, length): # function to generate white-gaussian-noise
    mu = 0
    res = np.zeros((length, n1))
    for i in range(length):
        res[i] = np.random.normal(mu, np.sqrt(variance), n1)
    return res

def decompose(x, Eb):
    cos = [np.cos(2*np.pi*f_c*i) for i in np.linspace(0, T, n1)] # basis function
    res = np.dot(x, cos)/len(x)
    return res

def demodulate(r_sym):
    S_space = [1, -1]
    dist = np.zeros((code_len, M)) # matrix to store the distance values from the 4 symbols
    for i in range(code_len):
	    for j in range(M):
		    dist[i][j] = np.linalg.norm(r_sym[i] - S_space[j]) # calculating the distance

    demod = np.zeros(code_len)
    bit_array = [0, 1]
    i = 0
    while i < code_len:
	    index = np.argmin(dist[i]) # minimum distance 
	    demod[i] = bit_array[index]
	    i += 1

    return demod

#################################################################################################################
# utility functions
def xor(arr, n):
	res = 0
	for i in range(len(arr)):
		if i != n:
			res += arr[i]
	return res % 2

def findMax(arr, index, n):
	res = -1
	for i in range(len(col[index])):
		if(i != n):
			res = max(res, arr[col[index][i]])

	return res

def findMin(arr, index, n):
	res = 2
	for i in range(len(col[index])):
		if(i != n):
			res = min(res, arr[col[index][i]])

	return res

def second_min(arr, pos, rowIndex):
    minEl = 1e6
    for i in row[rowIndex]:
        if (i != pos):
            minEl = min(minEl, arr[i])

    return minEl

def computeTanh(arr, rowIndex):
    res = 0
    for j in row[rowIndex]:
        res += abs(np.log(np.tanh(abs(arr[j]/2))))   
    return res
    

def findPos(arr, rowIndex):
    res = 1e6
    index = 0
    for i in row[rowIndex]:
        if(res > arr[i]):
            res = arr[i]
            index = i
    return index

def sgn(arr, rowIndex):
    res = 1
    for i in row[rowIndex]:
        res *= np.sign(arr[i])
    return res

#############################################################################################################
# decoding functions

def beliefProp(demod, var): # decoding using belief-propagation
    L = H.copy()
    decod = np.zeros(math.ceil(k*len(demod)/n))

    for i in range(int(len(demod)/n)):
        r = demod[n*i : n*(i+1)]
        for j in range(len(L)):
            L[j] = H[j] * r # Analogous to adjacency matrix representation of the Tanner graph

        currDecision = np.zeros(n) # to store the current decision

        for _ in range(10):
            # row operations
            for j in range(len(L)):
                x = L[j]                
                S = sgn(x, j)
                l_m = computeTanh(x, j)

                for l in row[j]:
                    temp = abs(l_m - abs(np.log(np.tanh(abs(L[j][l]/2)))))
                    if(temp == 0):
                        temp = 1e-7

                    x[l] = S * np.sign(L[j][l]) * abs(np.log(np.tanh(temp/2)))

                L[j] = x

            # column operations
            for j in range(n):
                sum_j = r[j] + np.sum(L[:, j])
                for l in col[j]:
                    L[l][j] = sum_j - L[l][j]

                if sum_j < 0:
                    currDecision[j] = 1
                else:
                    currDecision[j] = 0
            
            decod[k*i : k*(i+1)] = currDecision[0:k]

    return decod

###########################################################################################################

def beliefProp_Minsum(demod, var): # decoding using belief-propagation with min-sum approximation
    L = H.copy()
    decod = np.zeros(math.ceil(k*len(demod)/n))

    for i in range(int(len(demod)/n)):
        r = demod[n*i : n*(i+1)]
        for j in range(len(L)):
            L[j] = H[j] * r 

        currDecision = np.zeros(n) # to store the current decision

        for _ in range(10): # no. of iterations
            for j in range(len(L)):
                # row operations
                x = L[j]
                pos = findPos(abs(x), j) # using min-sum approximation to |log(tanh(|x|/2))| function
                m1 = abs(x[pos])
                m2 = second_min(abs(x), pos, j)

                S = sgn(x, j)
                for l in row[j]:
                    if(l == pos):
                        x[l] = S * np.sign(x[l]) * m2
                    else:
                        x[l] = S * np.sign(x[l]) * m1

                L[j] = x

            # column operations
            for j in range(n):
                sum_j = r[j] + np.sum(L[:, j])
                for l in col[j]:
                    L[l][j] = sum_j - L[l][j]

                if sum_j < 0:
                    currDecision[j] = 1
                else:
                    currDecision[j] = 0
            
            decod[k*i : k*(i+1)] = currDecision[0:k]

    return decod

############################################################################################################

def gallagher(bits):
    L = H.copy()
    res = np.zeros(math.ceil(k*len(bits)/n))

    for i in range(int(len(bits)/n)):		
	    r = bits[n*i : n*(i+1)]
	    c = r

	    for j in range(len(L)):
		    L[j] = H[j] * r

	    for _ in range(10): # no. of iterations
		    for j in range(len(H)):
			    for l in range(n):
					#column operations
				    temp = findMax(L[:, l], l, j)
				    if(temp == findMin(L[:, l], l, j)): # if all entries of the column are equal.
					    L[j][l] = temp
					    c[l] = temp
				    else:
					    L[j][l] = r[l]

                    #row operations
				    L[j][l] = xor(L[j], l)

		    res[k*i : k*(i+1)] = c[0:k]				

    return res 

################################################################################################################

Eg = 1e-6
E_avg = Eg/2

#Computing average energy per bit
Eb =(E_avg*n)/(np.log2(M)*k)
Eb_N0_dB = [-10, -5, 0, 5, 10] #values of E_b/N_0 in dB
Eb_N0 = [10**(i/10) for i in Eb_N0_dB] #corresponding values of E_b/N_0
BER_b = []
BER_g = []
BER_m = []

print("Belief Propagation: ##########################################\n")
for j in range(len(Eb_N0)):
    print("For E_b/N_0 =", Eb_N0_dB[j], "dB")
    N_0 = Eb/Eb_N0[j]
    w = WGN(f_s * N_0/2, code_len) # white gaussian noise

    s = np.zeros((code_len, n1))
    for i in range(code_len):
        s[i] = modulate(code[i], Eb, i) # modulating

    r = s + w # received signal
    r_sym = np.zeros(code_len)

    for i in range(code_len):
        r_sym[i] = decompose(r[i], Eb) # converting from signal to symbols

    decod = beliefProp(r_sym, f_s * N_0/2) # decoding
    #decod = 0
    decod_m = beliefProp_Minsum(r_sym, f_s * N_0/2)

    error = (img != decod).sum()
    error_m = (img != decod_m).sum()

    print("No. of incorrectly decoded bits (without min-sum):", error)
    print("Bit Error rate (without min-sum):", error/size)
    print("No. of incorrectly decoded bits (with min-sum):", error_m)
    print("Bit Error rate (with min-sum):", error_m/size, "\n")
    BER_b.append(error/size)
    BER_m.append(error_m/size)    
    # decod = decod.reshape(lx, ly) 
    # plt.imshow(decod, 'gray')
    # plt.show()

print("Gallagher-A decoding: ########################################## \n")
for j in range(len(Eb_N0)):
    print("For E_b/N_0 =", Eb_N0_dB[j], "dB")
    N_0 = Eb/Eb_N0[j]
    w = WGN(f_s*N_0/2, code_len) # white gaussian noise

    s = np.zeros((code_len, n1))
    for i in range(code_len):
        s[i] = modulate(code[i], Eb, i) # modulating

    r = s + w # received signal
    r_sym = np.zeros(code_len)

    for i in range(code_len):
        r_sym[i] = decompose(r[i], Eb) # converting from signal to symbols

    demod = demodulate(r_sym) # demodulating

    decod = gallagher(demod) # decoding 
    error = (img != decod).sum()
    print("No. of incorrectly decoded bits:", error)
    BER_g.append(error/size)
    print("Bit Error rate:", error/size, "\n")

plt.figure()
plt.semilogy(Eb_N0_dB, BER_b, label = "Belief-propagation (BP)", marker = 'o')
plt.semilogy(Eb_N0_dB, BER_m, label = "BP with min-sum", marker = 'o')
plt.semilogy(Eb_N0_dB, BER_g, label = "Gallagher-A", marker = 'o')
plt.legend()
plt.xlabel('$E_b/N_0$ in dB')
plt.ylabel('BER')
plt.savefig('../figs/AWGN.png')
plt.show()

