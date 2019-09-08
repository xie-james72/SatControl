import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles as place
from math import atan2

#Weird issue where the noise does not affect anything unless you add it to the a
#priori estimate of P...

#noise variance
(nx,ny,na) = (0.1,0.1,0.2)
N = np.array([[nx,0,0],[0,ny,0],[0,0,na]], dtype = float)
H = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype = float) #sensor selector matrix

#Define trajectory of a circle
r = 3 #radius of circle
rate = 0.2 #angular travel rate; "dad" technically
t = 0
vd = r*rate #desired net speed

#Define initial state error
(xe, ye, ae) = (5.0, 1.0, np.pi)
(xd, yd, ad) = (0.0, 0.0, 0.0)
P = np.array([[100,0,0],[0,100,0],[0,0,100]], dtype = float) #initial guess of the noise covariance

state = np.array([[xe],[ye],[ae]], dtype = float)
xdstor = [xd]
ydstor = [yd]
xstor = [xe + xd]
ystor = [ye + yd]

#Trajectory tracking
while np.sum(abs(state)) > 0.1:
    #Define desired vehicle response model
    xd = r*np.sin(rate*t) #desired x position
    yd = r - r*np.cos(rate*t) #desired y position
    dxd = r*rate*np.cos(rate*t) #desired x velo
    dyd = r*rate*np.sin(rate*t) #desired y velo
    ad = atan2(dyd,dxd) #desired angular position
    
    #State-variable form after linearization about an equilibrium point (i forgot which one):
    #The equilibrium point would likely be at 0 error and 0 input.   
    A = np.array([[0,0,-vd*np.sin(ad)],
                   [0,0,vd*np.cos(ad)],
                   [0,0,0]],dtype = float)
    B = np.array([[np.cos(ad),0],
                   [np.sin(ad),0],
                   [0,1]],dtype = float)
    poles = [p*rate for p in [1,0.6,0.8]]
    K = place(A,B,poles,method='KNV0').gain_matrix
    u = np.matmul(K,state)
    
    #Kalman a-priori estimate
    dstate = np.matmul(A,state) + np.matmul(B,u)
    state = state + dstate
    P = np.matmul(A,np.matmul(P,np.transpose(A))) + N #covariance of the system
    
    #Kalman a-posteriori estimate
    noise = np.array([np.random.normal(0,[nx,ny,na])], dtype = float).T
    sensor = np.matmul(H,state) + noise #sensor measurement
    
    Q1 = np.matmul(H, np.matmul(P,np.transpose(H))) + N
    Q2 = np.matmul(P, np.matmul(np.transpose(H), np.linalg.inv(Q1))) #placeholder calculations
    state = state + np.matmul(Q2,sensor - np.matmul(H,state)) #re-estimating state
    P = np.matmul(np.identity(3)- np.matmul(Q2,H), P) #re-estimating variability
    
    #store for later use
    xdstor.append(xd)
    xstor.append(xd + state[0])
    ydstor.append(yd)
    ystor.append(yd + state[1])

    t = t + 1

plt.plot(xstor, ystor)
plt.plot(xdstor, ydstor)

#next step:
#Convert model to follow an orbit path with orbit-style coordinates
#Convert vehicle model into a satellite rather than... i think its a unicycle???

#Build a satellite model with perfect controllibility of its states
#Implement a 3-axis reaction wheel model
#Implement a 4-axis reaction wheel model
#Variance of (sun sensors, gyro, accelerometer, GPS, voltage/current reader for rxn wheels)