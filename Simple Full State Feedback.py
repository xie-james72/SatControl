import numpy as np
import matplotlib.pyplot as plt
#Simple state space model of a point-like object with mass in a 2D environment.

#initial conditions
(x1, y1, x2, y2) = (20, 10, -10, -20) #Initial state vector: xy position, xy velocity
state = np.array([[x1],[y1],[x2],[y2]], dtype = float)
(dx1, dy1, dx2, dy2) = (0,0,0,0)

m = 10 #kg
K1 = -0.04*m
K2 = -0.04*m
K3 = -0.1*m
K4 = -0.1*m

endpoint = np.array([[100], [100], [0], [0]], dtype = float) #ending state

statestor = [state]

#Since x and y are independent coordinates, they will have independent inputs.
A = np.array(([0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]), dtype = float)
B = np.array([[0,0], [0,0], [1/m,0], [0,1/m]], dtype = float)
K = np.array([[K1,0,K3,0],[0,K2,0,K4]], dtype = float)

#closed loop controller
k = 1
error = endpoint - state
while np.sum(abs(error)) > 1:
#for k in range(0,100):
    error = endpoint - state
    dstate = np.matmul(A,state) - np.matmul(B, np.matmul(K, error))
    state = state + dstate
    statestor.append(state)
    k = k + 1

#data conditioning & plotting
x1state = []
x2state = []
x3state = []
x4state = []
for l in statestor:
    x1state.append(float(l[0]))
    x2state.append(float(l[1]))
    x3state.append(float(l[2]))
    x4state.append(float(l[3]))

t = np.linspace(0,k,k)
plt.plot(t,x1state)
plt.plot(t,x2state)
plt.plot(t,x3state)
plt.plot(t,x4state)