import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
MonaLisa = np.load('binary_image.npy')
plt.imshow(MonaLisa,'gray')
plt.title('Original Pic')
plt.show()

f_c = 2e6
f_s = 50e6
T = 1e-6
#Number of 	Bits = nbits = 11000
#Number of Symbols = nbits/2 = 5500
#Duration = nbits*T/2 = 5.5ms
#Total Number Of samples = duration*f_s = 275000
#Time Interval Per Sample = f_s*T = 50

b = np.matrix(MonaLisa).getA1() #Bits Stream as a Single Array
print(b)

s = []
x = np.zeros(11000)
for i in range(11000):
	if(b[i] == 0):
		x[i] = 1
	elif(b[i] == 1):
		x[i] = -1
	
	
for i in range(5500):
	samples_interval = np.linspace(i*50,(i+1)*50,50,endpoint = False)
	s.append(x[2*i]*np.cos(2*np.pi*f_c*samples_interval/f_s) + x[2*i+1]*np.sin(2*np.pi*f_c*samples_interval/f_s))
	
s = np.matrix(s)
s = s.getA1() #Transmitting Waveform

E_av = T
E_b = E_av/2 #Energy per information bit
print("E_b = "+str(E_b))
print()

Eb_N0_dB = np.array([-10, -5, 0, 5])
Eb_N0 = 10**(Eb_N0_dB/10)
N_0 = E_b/(Eb_N0)

for x in range(4):
	r = np.zeros(275000)
	w = np.random.normal(0,np.sqrt(f_s*N_0[x]/2),275000)
	r = s + w

	#Demodulator:
	n = np.linspace(0,275000,275000,endpoint = False)
	#Each Q represents the particular quadrant and corresponding QAM point in (1,1), (1,-1), (-1,-1), (-1,1)
	Q1 = (np.sqrt(T)/2)*np.cos(2*np.pi*f_c*n/f_s) + (np.sqrt(T)/2)*np.sin(2*np.pi*f_c*n/f_s)
	Q2 = (np.sqrt(T)/2)*np.cos(2*np.pi*f_c*n/f_s) - (np.sqrt(T)/2)*np.sin(2*np.pi*f_c*n/f_s)
	Q3 = -(np.sqrt(T)/2)*np.cos(2*np.pi*f_c*n/f_s) + (np.sqrt(T)/2)*np.sin(2*np.pi*f_c*n/f_s)
	Q4 = -(np.sqrt(T)/2)*np.cos(2*np.pi*f_c*n/f_s) - (np.sqrt(T)/2)*np.sin(2*np.pi*f_c*n/f_s)

	b_new = np.zeros(11000) #Initializing the output bitstream
	
	#Minimum Distance Decoding
	temp1 = (r - Q1)**2
	temp2 = (r - Q2)**2
	temp3 = (r - Q3)**2
	temp4 = (r - Q4)**2
	min1 = np.zeros(5500)
	min2 = np.zeros(5500)
	min3 = np.zeros(5500)
	min4 = np.zeros(5500)
	
	for i in range(0,5500):
		for j in range(i*50,(i+1)*50):
			min1[i] += temp1[j]
			min2[i] += temp2[j]
			min3[i] += temp3[j]
			min4[i] += temp4[j]
			
		if(min(min1[i],min2[i],min3[i],min4[i]) == min1[i]):
			b_new[2*i], b_new[2*i+1] = 0,0
				
		elif(min(min1[i],min2[i],min3[i],min4[i]) == min2[i]):
			b_new[2*i], b_new[2*i+1] = 0,1
			
		elif(min(min1[i],min2[i],min3[i],min4[i]) == min3[i]):
			b_new[2*i],b_new[2*i+1] = 1,0
				
		else:
			b_new[2*i],b_new[2*i+1] = 1,1
			
	print("For " + '{}'.format(Eb_N0_dB[x]) + "dB:")
	print("Variance = {}".format(f_s*N_0[x]/2))		
	print("Actual Number of Error Bits = {}".format(np.count_nonzero(b_new - b)))
	BER = norm.sf(np.sqrt(2*Eb_N0[x]))
	print("BER for " +str(Eb_N0_dB[x])+"dB = "+ str(BER))
	print("Expected Number of Error Bits = {}".format(11000*BER))
	print()
	
	MonaLisa_received = np.reshape(b_new,(110,100))
	plt.imshow(MonaLisa_received,'gray')
	plt.title("For " + '{}'.format(Eb_N0_dB[x]) + "dB")
	plt.show()
