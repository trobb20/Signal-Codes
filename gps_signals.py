#Signals
#Teddy Robbins ME193 GPS & Inertial w/ Pratap Misra

import numpy as np 
import matplotlib.pyplot as plt 

def XOR(bits):
	#Takes an array of bits and computes the exclusive or of the input
	state=bits[0]
	for i in range(1,bits.shape[0]):
		next_bit=bits[i]
		if state==next_bit:
			state=0
		else:
			state=1
	return state

def shift_register(register,newBit):
	#Shifts a register and adds new bit to the front
	register = np.insert(register,0,newBit)
	register = register[0:-1]
	return register

def binaryVectorToHex(b):
	#Takes a binary vector (array) and converts it to hexadecimal
	b_string = ''
	for i in b:
		b_string=b_string+str(int(i))
	hex_out = hex(int(b_string,2))
	return hex_out

def make_code(S1,S2):
	#Makes a GPS C/A PRN code with phase selectors S1 and S2
	#returns code as binary vector and hexadecimal

	code = np.zeros(1023)	#initialize code

	G1 = np.ones(10)		#initialize registers
	G2 = np.ones(10)

	for i in range(1023):

		#First register
		G1_det = np.array([G1[2],G1[9]])
		newBitG1 = XOR(G1_det) #compare G1 3 to G1 10
		G1_out = G1[9]

		#Second Register
		G2_det = np.array([G2[1],G2[2],G2[5],G2[7],G2[8],G2[9]])
		newBitG2 = XOR(G2_det)
		phases = np.array([G2[S1-1],G2[S2-1]])
		G2_out = XOR(phases)

		#Compare both register outputs
		out = XOR(np.array([G1_out,G2_out]))
		code[i] = out

		#Shift both registers
		G1 = shift_register(G1,newBitG1)
		G2 = shift_register(G2,newBitG2)

	return (code,binaryVectorToHex(code))

def transformToOneNegativeOne(code):
	#Transforms [0,1] code to [1,-1]
	new_code = np.zeros(code.shape)
	for i in range(code.shape[0]):
		if code[i] == 0:
			new_code[i] = 1
		else:
			new_code[i] = -1
	return new_code

def delay_code(code,delay):
	#Delays a code by a chipping delay
	indices = range(-delay,code.shape[0]-delay)
	new_code = np.take(code,indices,mode='wrap')
	return new_code

def cross_correlate(code1,code2,delay):
	#Computes cross correlation of two codes given a delay
	R = 1/1023 * np.sum(np.multiply(code1,delay_code(code2,delay)))
	return R

def auto_correlate(code,delay):
	#Computes auto correlation of a code given a delay
	return cross_correlate(code,code,delay)

#Code for answering questions is commented out and can be used as needed

############### Question 1 ###############
# code,hexcode=make_code(1,9)
# print(binaryVectorToHex(code[0:16]))
# print(binaryVectorToHex(code[-16:]))
# print(hexcode)

# fig1 = plt.figure()
# plt.subplot(2,1,1)
# plt.title('First 16 Chips')
# plt.step(np.arange(1,17,1),code[0:16])
# plt.subplot(2,1,2)
# plt.title('Last 16 Chips')
# plt.step(np.arange(1,17,1),code[-16:])
# fig1.tight_layout(pad=1)
# plt.show()

############### Question 2 ###############
# PRN19,PRN19hex = make_code(3,6)
# PRN25,PRN25hex = make_code(5,7)
# PRN5,PRN5hex = make_code(1,9)

# x1 = transformToOneNegativeOne(delay_code(PRN19,350))
# x2 = transformToOneNegativeOne(delay_code(PRN25,905))
# x3 = transformToOneNegativeOne(delay_code(PRN5,75))

# PRN19_transform = transformToOneNegativeOne(PRN19)
# PRN25_transform = transformToOneNegativeOne(PRN25)
# PRN5_transform = transformToOneNegativeOne(PRN5)

# PRN19_delay = delay_code(PRN19_transform,200)

# noise = 4*np.random.randn(1023)

# summed = x1+x2+x3+noise

# n_array = []
# R_array = []
# for n in range(1023):
# 	n_array.append(n)
# 	R_array.append(cross_correlate(PRN19_transform,summed,n))

# plt.figure()
# plt.title('Crosscorrelation of PRN19 and x1+x2+x3+noise')
# plt.plot(n_array,R_array)
# plt.show()

# fig = plt.figure()
# plt.subplot(4,1,1)
# plt.title('x1, x2, x3, vs Noise')
# plt.step(np.arange(1,1024,1),x1)
# plt.ylim((np.min(noise),np.max(noise)))
# plt.subplot(4,1,2)
# plt.step(np.arange(1,1024,1),x2)
# plt.ylim((np.min(noise),np.max(noise)))
# plt.subplot(4,1,3)
# plt.step(np.arange(1,1024,1),x3)
# plt.ylim((np.min(noise),np.max(noise)))
# plt.subplot(4,1,4)
# plt.step(np.arange(1,1024,1),noise)
# plt.ylim((np.min(noise),np.max(noise)))
# fig.tight_layout(pad=1)
# plt.show()