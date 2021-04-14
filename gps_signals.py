#Signals
import numpy as np 
import matplotlib.pyplot as plt 

def XOR(bits):
	state=bits[0]
	for i in range(1,bits.shape[0]):
		next_bit=bits[i]
		if state==next_bit:
			state=0
		else:
			state=1
	return state

def shift_register(newBit,register):
	register = np.insert(register,0,newBit)
	register = register[0:-1]
	return register

def binaryVectorToHex(b):
	b_string = ''
	for i in b:
		b_string=b_string+str(int(i))
	hex_out = hex(int(b_string,2))
	return hex_out

def make_code(S1,S2):

	code = np.zeros(1023)

	G1 = np.ones(10)
	G2 = np.ones(10)

	for i in range(1023):
		# print('G1 is: %s'%(str(G1)))
		# print('G2 is: %s'%(str(G2)))

		G1_det = np.array([G1[2],G1[9]])
		newBitG1 = XOR(G1_det) #compare G1 3 to G1 10
		G1_out = G1[9]
		# print('---G1 shift register---')
		# print('Comparing %d to %d'%(G1[2],G1[9]))
		# print('Got %d'%newBitG1)
		# print('Outputting %d'%G1_out)

		G2_det = np.array([G2[1],G2[2],G2[5],G2[7],G2[8],G2[9]])
		newBitG2 = XOR(G2_det)
		phases = np.array([G2[S1-1],G2[S2-1]])
		G2_out = XOR(phases)
		# print('---G2 shift register---')
		# print('Comparing %d to %d to %d to %d to %d to %d'%(G2[1],G2[2],G2[5],G2[7],G2[8],G2[9]))
		# print('Got %d'%newBitG2)
		# print('Phase selector %d to %d'%(G2[S1-1],G2[S2-1]))
		# print('Outputting %d'%G2_out)

		# print('--- Output ---')
		out = XOR(np.array([G1_out,G2_out]))
		code[i] = out
		# print('Outputting: %d'%out)

		G1 = shift_register(newBitG1,G1)
		G2 = shift_register(newBitG2,G2)

		# print('--- Shifting ---')
		# print('G1 is: %s'%(str(G1)))
		# print('G2 is: %s'%(str(G2)))
		# print('----------------')

	return (code,binaryVectorToHex(code))

code,hexcode=make_code(1,9)
print(binaryVectorToHex(code[0:16]))
print(binaryVectorToHex(code[-16:]))
print(hexcode)

fig1 = plt.figure()
plt.subplot(2,1,1)
plt.title('First 16 Chips')
plt.step(np.arange(1,17,1),code[0:16])
plt.subplot(2,1,2)
plt.title('Last 16 Chips')
plt.step(np.arange(1,17,1),code[-16:])
fig1.tight_layout(pad=1)
plt.show()