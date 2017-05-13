import numpy as np
from matplotlib.pyplot import *

N = 10  #number of steps
P = 2*N+1 #number of positions in each direction 

coin0 = np.array([1,0,0]) # |0> This will be tails
coin1 = np.array([0,1,0]) # |1> This will be heads
coin2 = np.array([0,0,1]) # |2> This will be our vertical shifting

C00 = np.outer(coin0,coin0)
C01 = np.outer(coin0,coin1)
C02 = np.outer(coin0,coin2)
C10 = np.outer(coin1,coin0)
C11 = np.outer(coin1,coin1)
C12 = np.outer(coin1,coin2)
C20 = np.outer(coin2,coin0)
C21 = np.outer(coin2,coin1)
C22 = np.outer(coin2,coin2)

A = np.array([[1/np.sqrt(2.),0,1/np.sqrt(2.)],[0,-1/np.sqrt(2.),1/np.sqrt(2.)],[1/np.sqrt(2.),-1/np.sqrt(2.),0]]) #A represents our unitary matrix in coin space
C_hat = (A[0,0]*C00 + A[0,1]*C01+A[0,2]*C02 + A[1,0]*C10 + A[1,1]*C11 + A[1,2]*C12 + A[2,0]*C20+A[2,1]*C21+A[2,2]*C22 )

ShiftLeft  = np.kron(np.eye(2),np.roll(np.eye(P), 1, axis=0))
ShiftRight = np.kron(np.eye(2),np.roll(np.eye(P), -1, axis=0))
ShiftVert  = np.kron(np.roll(np.eye(2), -1, axis=0),np.eye(P)) #Moving between rungs of the ladder

S_hat = np.kron(ShiftLeft, C00) + np.kron(ShiftRight, C11) + np.kron(ShiftVert, C22) 


U = S_hat.dot(np.kron(np.eye(2*P),C_hat))

posn0 = np.zeros(2*P)
posn0[P] = 1 # np.array indexing starts from (0) so index (P) is central in 0 to 2P

Cvec = np.array([1/np.sqrt(2.),0,-1j/np.sqrt(2.)]) # Initial vector of the coin space
initialCoin = (Cvec[0]*coin0+Cvec[1]*coin1+Cvec[2]*coin2)
psi0 = np.kron(posn0,initialCoin) #Initial spin 

# psiN = np.linalg.matrix_power(U, N).dot(psi0)
psiN = np.linalg.matrix_power(U, N).dot(psi0)

s = np.kron(np.eye(2*P),np.array((1,1,1)))
prob3d = np.reshape((s.dot((psiN.conjugate()*psiN).real)),(2,P))

fig = figure()
matplotlib.pyplot.imshow(prob3d.real, cmap='hot', interpolation='nearest')
#savefig('n16_HeatMap.png')
show()
