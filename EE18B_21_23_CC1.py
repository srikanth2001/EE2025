import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import itertools

def HammingDist(a,b,n):
	count = 0
	for i in range(n):
		if(a[i] != b[i]):
			count += 1
	return count

def demod(s,nbits):	
	r = np.zeros(50*nbits)
	w = np.random.normal(0,np.sqrt(f_s*N_0[y]/2),50*nbits)
	r = s + w

	#Demodulator:
	n = np.linspace(0,50*nbits,50*nbits,endpoint = False)
	#Each Q represents the particular quadrant and corresponding QAM point in (1,1), (1,-1), (-1,-1), (-1,1)
	Q1 = (np.sqrt(T)/2)*np.cos(2*np.pi*f_c*n/f_s) + (np.sqrt(T)/2)*np.sin(2*np.pi*f_c*n/f_s)
	Q2 = (np.sqrt(T)/2)*np.cos(2*np.pi*f_c*n/f_s) - (np.sqrt(T)/2)*np.sin(2*np.pi*f_c*n/f_s)
	Q3 = -(np.sqrt(T)/2)*np.cos(2*np.pi*f_c*n/f_s) + (np.sqrt(T)/2)*np.sin(2*np.pi*f_c*n/f_s)
	Q4 = -(np.sqrt(T)/2)*np.cos(2*np.pi*f_c*n/f_s) - (np.sqrt(T)/2)*np.sin(2*np.pi*f_c*n/f_s)

	x_new = np.zeros(nbits*2) #Initializing the output bitstream
	
	#Minimum Distance Decoding
	temp1 = (r - Q1)**2
	temp2 = (r - Q2)**2
	temp3 = (r - Q3)**2
	temp4 = (r - Q4)**2
	min1 = np.zeros(nbits)
	min2 = np.zeros(nbits)
	min3 = np.zeros(nbits)
	min4 = np.zeros(nbits)
	
	for i in range(0,nbits):
		for j in range(i*50,(i+1)*50):
			min1[i] += temp1[j]
			min2[i] += temp2[j]
			min3[i] += temp3[j]
			min4[i] += temp4[j]
			
		if(min(min1[i],min2[i],min3[i],min4[i]) == min1[i]):
			x_new[2*i], x_new[2*i+1] = 0,0
				
		elif(min(min1[i],min2[i],min3[i],min4[i]) == min2[i]):
			x_new[2*i], x_new[2*i+1] = 0,1
			
		elif(min(min1[i],min2[i],min3[i],min4[i]) == min3[i]):
			x_new[2*i], x_new[2*i+1] = 1,0
				
		else:
			x_new[2*i], x_new[2*i+1] = 1,1
	
	return x_new
	
def channel_decode(x_new,n,k):
	res = np.matmul(G1.T,mseq.T).T
	res = np.array(res)
	for i in range(16):
		for j in range(n):
			res[i][j] %= 2
			
	b_new = np.zeros(120000)
	
	#Minimum Hamming Distance Decoding
	for i in range(120000//k):
		y_n = x_new[n*i : n*(i+1)]
		minimum = HammingDist(y_n,res[0],n)
		minpos = 0
		for j in range(1,16):
			dH = HammingDist(y_n,res[j],n)
			if(dH < minimum):
				minimum = dH
				minpos = j
				
		b_new[k*i: k*(i+1)] = mseq[minpos]
		
	return b_new

MSS = np.load('mss.npy')
plt.imshow(MSS, 'gray')
plt.title('Original Image')
plt.show()

#Number of Bits = 120000

b = np.matrix(MSS).getA1() #Bits Stream as a Single Array

G1 = np.matrix([[1,1,1,1,0,0,0,0],[1,1,0,0,1,1,0,0],[1,0,1,0,1,0,1,0],[0,1,1,0,1,0,0,1]])

#Channel Encoder
x = np.zeros(240000,dtype = int)
temp2 = np.zeros(4,dtype = int)
for i in range(30000):
	for j in range(4):
		temp2[j] = b[4*i + j]
		
	temp1 = np.matmul(G1.T,temp2)
	temp1 = np.array(temp1)
	for j in range(8):
		x[8*i + j] = temp1[0][j]%2
	
f_c = 2e6
f_s = 50e6
T = 1e-6

#Modulator
s = []
qam = np.zeros(240000)
for i in range(240000):
	if(x[i] == 0):
		qam[i] = 1
	elif(x[i] == 1):
		qam[i] = -1
	
	
for i in range(120000):
	samples_interval = np.linspace(i*50,(i+1)*50,50,endpoint = False)
	s.append(qam[2*i]*np.cos(2*np.pi*f_c*samples_interval/f_s) + qam[2*i+1]*np.sin(2*np.pi*f_c*samples_interval/f_s))
	
s = np.matrix(s)
s = s.getA1() #Transmitting Waveform

#n/k = 2 as n = 8, k = 4
n_k = 2
E_b = T*n_k/2 #Energy per information bit

Eb_N0_dB = np.array([-2,0,2,4,6],dtype = float)
Eb_N0 = 10**(Eb_N0_dB/10)
N_0 = E_b/(Eb_N0)

Var = np.array([20,12,7,5])
Eb_N0_dB_x = 10*np.log(E_b*f_s/(2*Var))

mseq = [list(i) for i in itertools.product([0, 1], repeat=4)] #Mod-16 Counter - Useful for all 4-bit possibilites of 0s and 1s
mseq = np.array(mseq)

BER_y = np.zeros(4)
#For Given Variance values
for y in range(4):
	x_new = demod(s,120000)
	b_new = channel_decode(x_new,8,4)
		
	error_bits = np.count_nonzero(b_new - b)
	print("For "+ "Variance = " + '{}:'.format(Var[y]))
	print("Actual Number of Error Bits = {}".format(error_bits))
	print()
	
	BER_y[y] = error_bits/120000
	
	MSS_received = np.reshape(b_new,(400,300))
	plt.imshow(MSS_received,'gray')
	plt.title("For "+ "$\sigma^2$ = " + '{}'.format(Var[y]))
	plt.show()

plt.semilogy(Eb_N0_dB_x,BER_y,marker = 'o')
plt.xlabel('Variance in dB')
plt.ylabel('BER')
plt.title('BER vs. $E_b/N_0$')
plt.show()
print()

BER_y = np.zeros(5)

#For given Eb/N0 values
for y in range(5):
	x_new = demod(s,120000)
	b_new = channel_decode(x_new,8,4)
	
	error_bits = np.count_nonzero(b_new - b)
	BER_y[y] = error_bits/120000
	print("For " + '{}'.format(Eb_N0_dB[y]) + "dB:")
	print("Actual Number of Error Bits = {}".format(error_bits))
	print()
	
	MSS_received = np.reshape(b_new,(400,300))
	plt.imshow(MSS_received,'gray')
	plt.title("For " + '{}'.format(Eb_N0_dB[y]) + "dB")
	plt.show()

plt.semilogy(Eb_N0_dB,BER_y,marker = 'o')
plt.xlabel('$E_b/N_0$ in dB')
plt.ylabel('BER')
plt.title('BER vs. $E_b/N_0$')
plt.show()
