import numpy as np
import pylab as plt

R = 0.08
m = 1
v0 = 100
C = 0.47
v0x = (np.sqrt(3)/2)*v0
v0y = (1/2)*v0
x0 = 0
y0 = 0
v0 = v0x
w0 = v0y
X0 = np.array([x0,y0,v0,w0])
t0 = 0
tf = 10
h = 1e-3
N_steps = int((tf-t0)/h)

def f(X):
    x,y,v,w = X
    fx = v
    fy = w
    fv = -(np.pi*R**2*rho*C/(2*m))*v*np.sqrt(v**2+w**2)
    fw = -(np.pi*R**2*rho*C/(2*m))*w*np.sqrt(v**2+w**2) - g
    return np.array([fx,fy,fv,fw])

def k1(X,f):
    return h*f(X)
def k2(X,f):
    return h*f(X+.5*k1(X,f))
def k3(X,f):
    return h*f(X+.5*k2(X,f))
def k4(X,f):
    return h*f(X+k3(X,f))
def delta_x_t_h(X,f):
    return (1/6)*(k1(X,f)+2*k2(X,f)+2*k3(X,f)+k4(X,f))
	
def estimate_zero(X):
    index_1 = np.where(X[:,1]==np.min(X[:,1][X[:,1]>0]))[0][0]
    index_2 = np.where(X[:,1]==np.max(X[:,1][X[:,1]<0]))[0][0]
    tot = np.min(X[:,1][X[:,1]>0])-np.max(X[:,1][X[:,1]<0])
    w_1 = np.min(X[:,1][X[:,1]>0])/tot
    w_2 = -np.max(X[:,1][X[:,1]<0])/tot
    guess = w_1*X[index_1,0]+w_2*X[index_2,0]
    return guess
	
def make_plots(planet):
	if planet == 'Earth':
		rho = 1.22
		g = 9.8
		t = [t0]
		X = [X0]
		for i in range(1,N_steps):
			t.append(t[i-1]+h)
			new_X = X[i-1]+delta_x_t_h(X[i-1],f)
			X.append(new_X)
		t = np.array(t)
		X = np.array(X)

		fg, ax = plt.subplots(1,1,figsize=(12,8))
		plt.plot(X[:,0],X[:,1],lw=3)
		#plt.axhline(0,c='k',linestyle='--')
		plt.ylim(0,60)
		plt.xlabel('Horizontal Distance',fontsize=18)
		plt.ylabel('Height',fontsize=18)
		plt.title('Trajectory of 1kg Cannon Ball on Earth',fontsize=20)
		plt.tick_params(axis='both',labelsize=16)
		plt.show()
			
		masses = np.linspace(.25,8,20)
		zeros = []
		for m in masses:
			t = [t0]
			X = [X0]
			i = 1
			while (i < N_steps and X[-1][1])>=0:
				t.append(t[i-1]+h)
				new_X = X[i-1]+delta_x_t_h(X[i-1],f)
				X.append(new_X)
				i = i + 1
			t = np.array(t)
			X = np.array(X)
			zeros.append(estimate_zero(X))
			
		fg, ax = plt.subplots(1,1,figsize=(12,8))
		masses3 = [1,2,4,8]
		for m in masses3:
			t = [t0]
			X3 = [X0]
			for i in range(1,N_steps):
				t.append(t[i-1]+h)
				new_X = X3[i-1]+delta_x_t_h(X3[i-1],f)
				X3.append(new_X)
			t = np.array(t)
			X3 = np.array(X3)
			plt.plot(X3[:,0],X3[:,1],lw=3,label='m='+str(m)+' kg')
		plt.xlabel('Horizontal Distance',fontsize=18)
		plt.ylabel('Height',fontsize=18)
		plt.title('Trajectories of Cannon Balls of Various Masses on Earth',fontsize=20)
		plt.tick_params(axis='both',labelsize=16)
		plt.legend(fontsize=18)
		plt.ylim(0,150)
		plt.xlim(-10,650)
		plt.show()

		fg, ax = plt.subplots(1,1,figsize=(12,8))
		plt.scatter(masses,zeros,lw=3)
		plt.xlabel('Mass (kg)',fontsize=18)
		plt.ylabel('Distance Travelled (m)',fontsize=18)
		plt.title('Distance Travelled of Cannonballs of Different Masses on Earth',fontsize=20)
		plt.tick_params(axis='both',labelsize=16)
		plt.show()
	elif planet == 'Mars':
		rho = 0.20
		g = 3.71
		t = [t0]
		X = [X0]
		for i in range(1,N_steps):
			t.append(t[i-1]+h)
			new_X = X[i-1]+delta_x_t_h(X[i-1],f)
			X.append(new_X)
		t = np.array(t)
		X = np.array(X)

		fg, ax = plt.subplots(1,1,figsize=(12,8))
		plt.plot(X[:,0],X[:,1],lw=3)
		#plt.axhline(0,c='k',linestyle='--')
		plt.ylim(0,60)
		plt.xlabel('Horizontal Distance',fontsize=18)
		plt.ylabel('Height',fontsize=18)
		plt.title('Trajectory of 1kg Cannon Ball on Earth',fontsize=20)
		plt.tick_params(axis='both',labelsize=16)
		plt.show()
			
		masses = np.linspace(.25,8,20)
		zeros = []
		for m in masses:
			t = [t0]
			X = [X0]
			i = 1
			while (i < N_steps and X[-1][1])>=0:
				t.append(t[i-1]+h)
				new_X = X[i-1]+delta_x_t_h(X[i-1],f)
				X.append(new_X)
				i = i + 1
			t = np.array(t)
			X = np.array(X)
			zeros.append(estimate_zero(X))
			
		fg, ax = plt.subplots(1,1,figsize=(12,8))
		masses3 = [1,2,4,8]
		for m in masses3:
			t = [t0]
			X3 = [X0]
			for i in range(1,N_steps):
				t.append(t[i-1]+h)
				new_X = X3[i-1]+delta_x_t_h(X3[i-1],f)
				X3.append(new_X)
			t = np.array(t)
			X3 = np.array(X3)
			plt.plot(X3[:,0],X3[:,1],lw=3,label='m='+str(m)+' kg')
		plt.xlabel('Horizontal Distance',fontsize=18)
		plt.ylabel('Height',fontsize=18)
		plt.title('Trajectories of Cannon Balls of Various Masses on Earth',fontsize=20)
		plt.tick_params(axis='both',labelsize=16)
		plt.legend(fontsize=18)
		plt.ylim(0,150)
		plt.xlim(-10,650)
		plt.show()

		fg, ax = plt.subplots(1,1,figsize=(12,8))
		plt.scatter(masses,zeros,lw=3)
		plt.xlabel('Mass (kg)',fontsize=18)
		plt.ylabel('Distance Travelled (m)',fontsize=18)
		plt.title('Distance Travelled of Cannonballs of Different Masses on Earth',fontsize=20)
		plt.tick_params(axis='both',labelsize=16)
		plt.show()
		
make_plots('Mars')