# Adiabatic parcel with buoyancy-driven velocity cloud microphyscics model 

import numpy as np
import math
import scipy.special as scsp
import matplotlib.pyplot as plt
import random


#________________________________________CONSTANTS
z_max = 1500		# maximum height 			[m]
z_0 = 0		# base height 				[m]
L = int(10*z_max)	# z-grid
theta_0 = 289		# initial potential temperature 	[K]
r_v0 = 9*1E-3		# initial vapor mixing ratio 
r_l0 = 0		# initial liquid mixing ratio
p_R = 1E5		# reference pressure 			[Pa]
K = .61		# adiabatic constant
c_p = 1005		# constant pressure heat capacity	[J / kg K]
g = 9.81		# gravitational constant		[m / s ** 2]
R_v = 461.5		# water vapor gas constant		[J / K mol]
R_d = 287		# dry air gas constant			[J / K mol]
e_0 = 611		# reference partial pressure		[Pa]
T_0 = 273.15		# reference temperautre (0 C)		[K]
L_v = 2.5E6		# water vapor latent heat		[J / kg]
eps = .622		# gas-constants ratio
tolerance = 1E-15	# tolerance in Delta method
L_r = int(1E3)		# length of r-array
r_min = .05E-6		# minimum radius			[m]
r_max = .3E-6		# maximum radius			[m]
dt = .05		# timestep				[s]
C_gamma = 1.5E-9	# C gamma
N_particles = 100	# number of particles
K_0 = 2.4E-2		# initial K
rho_water = 997	# water density			[kg / m ** 3]
rho_d = 1.225		# dry air density			[kg / m ** 3]
C_d = .47		# C dry
R_0 = 50		# reference gas constant		[J / K mol]
D_0 = 2.21E-5		# reference D
z_inv = 840		# inversion height			[m]

#________________________________________SOME ARRAYS INITIALISATION
z = np.linspace(0,z_max,L)
theta_v = np.zeros(L)
p_0 = np.zeros(L)
T = np.zeros(L)
r_v = np.zeros(L)
r_l = np.zeros(L)
r_arr = np.linspace(r_min,r_max,L_r)
CDF_arr = np.zeros(L_r)
particles_r = np.zeros(N_particles)
dry_r = np.zeros(N_particles)

#________________________________________PARTIAL PRESSURE FUNCTION FOR SATURATED AIR
def e_s(T):
	return e_0 * np.exp(-L_v * ((1 / T)-(1 / T_0)) / R_v)
	
#________________________________________PARTIAL PRESSURE FUNCTION	
def e(r_v, p):
	return r_v * p / (r_v + eps)

#________________________________________MIXING RATIO FUNCTION
def r_vs(T, p):
	return eps * e_s(T) / (p - e_s(T))

#________________________________________F(Delta) FUNCTION
def F(T, p, r_vi, Delta):
	return r_vs(T, p) - r_vi + Delta

#________________________________________dF/dDelta FUNCITON
def dF(T, p, Delta):
	return (eps * p *L_v * L_v * e_s(T) / ((p - e_s(T)) ** 2 * R_v * T ** 2 * c_p))+1

#________________________________________TEMPERATURE FUNCTION
def Temp(T, Delta):
	return T + L_v * Delta / c_p

#________________________________________FIND ROOT FOR Delta BY NEWTON RAPHSON METHOD	
def NRroot(Delta, T, p, r_vi):
	T = Temp(T, Delta)
	Delta = Delta - F(T, p, r_vi, Delta) / dF(T, p, Delta)
	
	while True:
		if F(T, p, r_vi, Delta) < tolerance and Delta >= 0 :
			break
		else:
			T = Temp(T, Delta)
			Delta = Delta - F(T, p, r_vi, Delta) / dF(T, p, Delta)
	return Delta

#________________________________________RETURN C_kappa FOR GIVEN DRY RADIUS
def C_kappa(r_d):
	return K * r_d ** 3

#________________________________________EQUILIBRIUM WET RADIUS R_eq FINDER
def R_eq_find(A, B, S, R_eq, tolerance):
	F = S - (A / R_eq) + (B / pow(R_eq, 3))
	dF = A / pow(R_eq, 2) - (3.0 * B / pow(R_eq, 4))
	R_eq = R_eq - F / dF
	
	while True:
		if F < tolerance and Delta >= 0 :
			break
		else:
			F = S - (A / R_eq) + (B / pow(R_eq, 3))
			dF = A / pow(R_eq, 2) - (3 * B / pow(R_eq, 4))
			R_eq = R_eq - F / dF
	return R_eq
	
	
#________________________________________PROBABILITY DISTRIBUTION FUNCTION
mu = np.log(.075)
sig_sq = pow(np.log(1.6),2)
def PDF(r,mu,sig_sq):
	r* = 1E6
	
	return np.exp(-(pow(np.log(r) - mu, 2)) / (2 * sig_sq))/(r * np.sqrt(sig_sq * 2 * math.pi))

#________________________________________CUMULATIVE DISTRIBUTION FUNTION	
def CDF(r, mu, sig_sq):
	r* = 1E6
	
	return 0.5 + 0.5 * scsp.erf((np.log(r) - mu) / (np.sqrt(2 * sig_sq)))

#________________________________________TIME EVOLUTION OF RADIUS r FUNCTION
def rEvolution(r, D, A, B, S, dt):
	return r + D * dt * ((S / r)-(A / pow(r, 2))+ B / pow(r, 4))
	
#________________________________________EFFECTIVE DIFFUSION FUNCTION
def D(T):
	return 1/((-1+L_v / (R_v * T)) * L_v * rho_water / (K_0 * T) + rho_water * R_v * T / (D_0 * e_s(T)))

#________________________________________PRINT PDF FUNCTION		
def printPDF(particles_r,tStep,w,PDFmin,PDFmax):
	plt.clf()
	plt.hist(np.log(particles_r), 50, range = (1.1*PDFmin, 0.9*PDFmax))
	
	plt.ylim(0,40)
	
	tStr = str(tStep)
	tStr = tStr.rjust(5,'0')
	name = 'PDFt'+tStr+'.png'
	
	plt.title('PDF for w =  '+str(w)+' tStep  = '+tStr)
	plt.savefig(name)

#________________________________________w FINDING FUNCTION	
def wFind(z, w, rho, R, theta_v, p_0, dt, T, particles_r, dry_r, S, ksi, Delta):
	if z > z_inv:
		theta_v0 = theta_0 + pow(z - z_inv, 1 / 3)
	else:
		theta_v0 = theta_0
		
	b_t = g * (theta_v - theta_v0) / theta_v0
	f = -(3 / 8) * rho_0 * C_d * w * np.abs(w) / (pow(rho_0, 1 / 3) * R * pow(rho, 2 / 3))
	
	z_star = z + dt * w
	w_star = w + dt * (b_t + f)
	
	if Delta! = 0:
		Delta = 0
		for l in range(len(particles_r)):
				rOld = particles_r[l]
				rNew = rEvolution(rOld, D(T), C_gamma, C_kappa(dry_r[l]), S, dt)
				Delta+ = ksi * (rNew ** 3 - rOld ** 3)
		theta_v_tdt = theta_v + L_v * Delta / (c_p * pow(p_0 / p_R, K))
	else:
		theta_v_tdt = theta_v
	
	if z > z_inv:
		theta_v0 = theta_0 + pow(z_star - z_inv, 1 / 3)
	b_tdt = g * (theta_v_tdt - theta_v0) / theta_v0
	z_tdt = z + 0.5 * dt * (w + w_star)
	w_tdt = w + 0.5 * dt * (b_t + b_tdt) + f
	
	return (z_tdt, w_tdt)

#________________________________________CALCULATING PROFILES WITH NR METHOD
activate = 0
print('calculating...')
for i in range(L):
	if (i == 0):
		theta_v[i] = theta_0
		r_v[i] = r_v0
		r_l[i] = r_l0
		p_0[i] = p_R
		T[i] = theta_v[i] * pow(p_0[i] / p_R, K)
		pi = p_0[i]
		Ti = T[i]
	else:
		pi = pow((pow(p_0[i-1], K) - g * pow(p_R, K) * (z[i] - z[i-1]) / (c_p * theta_0)), 1 / K)
		Ti = theta_v[i-1] * pow(pi / p_R, K)
	if r_v[i-1] < r_vs(Ti, pi) and activate == 0 and i != 0:
		theta_v[i] = theta_0
		r_v[i] = r_v0
		r_l[i] = r_l0
		thetabar_v = (theta_v[i] + theta_v[i-1]) / 2.0
		p_0[i] = pow((pow(p_0[i-1], K) - g * pow(p_R, K) * (z[i] - z[i-1]) / (c_p * thetabar_v)), 1 / K)
		T[i] = theta_v[i] * pow(p_0[i] / p_R, K)
		if r_v[i-1] > r_vs(T[i], p_0[i]):
			activate = 1
	elif i! = 0:
		Delta = 0
		Delta = NRroot(Delta, T[i-1], p_0[i-1], r_v[i-1])
		theta_v[i] = theta_v[i-1] + L_v * Delta / (c_p * pow(p_0[i-1] / p_R, K))
		thetabar_v = (theta_v[i] + theta_v[i-1]) / 2.0
		p_0[i] = pow((pow(p_0[i-1], K)- g * pow(p_R, K) * (z[i] - z[i-1]) / (c_p * thetabar_v)), 1 / K)
		T[i] = theta_v[i]*pow(p_0[i]/p_R,K)
		r_v[i] = r_v[i-1] - Delta
		r_l[i] = r_l[i-1] + Delta

#________________________________________CALCULATING PROFILES DEPENDENT OF VELOCITY w		
L_w = int(4.1E5)
w = np.zeros(L_w)
w[0] = float(input('All profiles calculated. Please enter initial velocity of a parcel:\n'))
c = float(input('Please enter concentration (in 1/cm^-3):\n'))
c* = 1E6
print('calculating profiles dependent on w...')

#________________________________________INITIALISATION OF ARRAYS WITH LENGTH DEPENDENT ON w
#________________________________________INDEX _w SYMBOLISES w-DEPENENCE
t_grid = int(3E3)
t_w = np.linspace(0, t_grid, t_grid)
z_w = np.zeros(L_w)
p_0w = np.zeros(L_w)
theta_vw = np.zeros(L_w)
T_w = np.zeros(L_w)
r_vw = np.zeros(L_w)
r_lw = np.zeros(L_w)
S_w = np.zeros(L_w)
var = np.zeros(L_w)
R_w = np.zeros(L_w)
rho_w = np.zeros(L_w)
time = np.zeros(L_w)
delta_theta = 1E-3
p_0w[0] = p_0[0];	theta_vw[0] = theta_v[0] + delta_theta;	T_w[0] = T[0] + delta_theta;
r_vw[0] = r_v[0];	S_w[0] = e(r_vw[0], p_0w[0]) / e_s(T_w[0]) - 1;	rho_0 = p_0w[0] / (R_d * T_w[0]);
rho_w[0] = rho_0;	R_w[0] = R_0


#________________________________________CONSTRUCTING CDF ARRAY FOR FURTHER INTERPOLATION
for i in range(len(r_arr)):
	CDF_arr[i] = CDF(r_arr[i], mu, sig_sq)
	

#________________________________________TAKING N_particles NUMBER OF RANDOM RADII OBEYING CDF
for k in range(len(particles_r)):
	psi = random.random()
	while psi < CDF(r_min, mu, sig_sq) or psi > CDF(r_max, mu, sig_sq):
		psi = random.random()
	j = 0
	for i in range(len(CDF_arr)):
		if CDF_arr[i] > psi:
			j = i - 1
			break

	CDF1 = CDF_arr[j];	CDF2 = CDF_arr[j+1];	r1 = r_arr[j];	r2 = r_arr[j+1]
	r = r1 + (psi - CDF1) * (r2 - r1) / (CDF2 - CDF1)
	dry_r[k] = r
	r = R_eq_find(C_gamma, C_kappa(r), S_w[0], r, tolerance)
	particles_r[k] = r
	
#________________________________________CALCULATION OF PROFILES FOR RISING PARCEL
var[0] = np.var(np.log(particles_r))
#var[0] = np.var(particles_r)
PDFmin = np.log(min(particles_r)) # USED AS LIMIT FOR PDF PLOTS
PDFmax = np.log(max(particles_r)) # USED AS LIMIT FOR PDF PLOTS
n = c / rho_d
ksi = n / N_particles
last_index = L_w - 1
act = 0
f = open("results.txt","w")
for i in range(int(L_w)):
	print('progress: ',round(1E2 * i / L_w),'%\t\tz: ',round(z_w[i]))
	var[i] = np.var(np.log(particles_r))
	#var[i] = np.var(particles_r)
	time[i] = dt * i
	
	'''if i%500 =  = 0 or i =  = 1:
		printPDF(particles_r,i,w,PDFmin,PDFmax)''' # TO SEE PDF FOR CHOSEN i's
		
	if S_w[i-1] < 0 and i != 0 and activate == 0: # UNDER CONDENSATION LEVEL PROFILES AS IN EXC. 1
		theta_vw[i] = theta_0 + delta_theta
		r_vw[i] = r_v0
		r_lw[i] = r_l0
		thetabar_vw = (theta_vw[i] + theta_vw[i-1]) / 2.0
		p_0w[i] = pow((pow(p_0w[i-1], K) - g * pow(p_R, K) * (z_w[i] - z_w[i-1]) / (c_p * thetabar_vw)), 1 / K)
		T_w[i] = theta_vw[i] * pow(p_0w[i] / p_R, K)
		rho_w[i] = rho_w[i-1] * p_0w[i-1] * T_w[i] / (p_0w[i] * T_w[i-1])
		R_w[i] = R_w[i-1] * pow(rho_w[i-1] / rho_w[i], 1 / 3)
		S_w[i] = e(r_vw[i], p_0w[i]) / e_s(T_w[i]) - 1
		Delta = 0
		
		if S_w[i] >= 0:
			activate = 1
			
	elif i != 0:			# ABOVE CONDENSATION LEVEL WE TRY TO FIND DELTA
		Delta = 0
		for l in range(len(particles_r)):
			rOld = particles_r[l]
			rNew = rEvolution(rOld, D(T_w[i-1]), C_gamma, C_kappa(dry_r[l]), S_w[i-1], dt)
			Delta+ = ksi * (rNew ** 3 - rOld ** 3)
			particles_r[l] = rNew
		Delta *= 4.0 * math.pi * rho_water / 3.0
		theta_vw[i] = theta_vw[i-1] + L_v * Delta / (c_p * pow(p_0w[i-1] / p_R, K))
		thetabar_vw = (theta_vw[i] + theta_vw[i-1]) / 2.0
		p_0w[i] = pow((pow(p_0w[i-1] , K) - g * pow(p_R,K) * (z_w[i] - z_w[i-1]) / (c_p * thetabar_vw)), 1 / K)
		T_w[i] = theta_vw[i] * pow(p_0w[i] / p_R, K)
		rho_w[i] = rho_w[i-1] * p_0w[i-1] * T_w[i] / (p_0w[i] * T_w[i-1])
		R_w[i] = R_w[i-1] * pow(rho_w[i-1] / rho_w[i], 1 / 3)
		r_vw[i] = r_vw[i-1] - Delta
		r_lw[i] = r_lw[i-1] + Delta
		S_w[i] = e(r_vw[i], p_0w[i]) / e_s(T_w[i]) - 1
		if act == 0:	# THERE APPEARS A SMALL SINGULARITY AT CONDENSATION LEVEL, HERE I
				# ELIMINATE IT
			theta_vw[i] = theta_vw[i-1]
			p_0w[i] = p_0w[i-1]
			T_w[i] = T_w[i-1]
			r_vw[i] = r_vw[i-1]
			r_lw[i] = r_lw[i-1]
			var[i] = var[i-1]
			S_w[i] = S_w[i-1]
			act = 1
			
	if i+1! = L_w:
		(z_w[i+1], w[i+1]) = wFind(z_w[i], w[i], rho_w[i], R_w[i], theta_vw[i], p_0w[i], dt, T_w[i], particles_r, dry_r, S_w[i], ksi, Delta)
		if z_w[i+1] > z_max:
			last_index = i
			break
			
	if i == 0 or (z_w[i] > 800 and i % 1E1 == 0):
		results_string = str(i)+'\t'+str(z_w[i])+'\t'+str(w[i])+'\t'+str(theta_vw[i])+'\t'+str(T_w[i])+'\t'+str(p_0w[i])+'\t'+str(S_w[i])+'\t'+str(r_lw[i])+'\t'+str(r_vw[i])+'\n'
		f.write(results_string)

theta_vw = theta_vw[:last_index]
z_w = z_w[:last_index]
T_w = T_w[:last_index]
r_vw = r_vw[:last_index]
S_w = S_w[:last_index]
var = var[:last_index]
w = w[:last_index]
r_lw = r_lw[:last_index]
p_0w = p_0w[:last_index]
R_w = R_w[:last_index]
rho_w = rho_w[:last_index]
time = time[:last_index]

#________________________________________PLOTS
plt.figure(0)
plt.plot(theta_v,z,'r',markersize = 1,label = 'Saturation adjustment')
plt.plot(theta_vw,z_w,'b',markersize = .5,label = 'Adiabatic parcel')
plt.xlabel('Potential temperature $\Theta$ [K]')
plt.ylabel('Height z [m]')
plt.ylim(bottom = 0)
plt.legend()
#plt.title('$\Theta(z)$',fontsize = 24)
plt.savefig('theta.png',format = "png")

plt.figure(2)
plt.plot(p_0/1E2,z,'r',markersize = 1,label = 'Saturation adjustment')
plt.plot(p_0w/1E2,z_w,'b',markersize = 1,label = 'Adiabatic parcel')
plt.xlabel('Pressure $p$ [hPa]')
plt.ylabel('Height z [m]')
plt.ylim(bottom = 0)
plt.legend()
#plt.title('$p(z)$',fontsize = 24)
plt.savefig('p.png',format = "png")

plt.figure(3)
plt.plot(T,z,'r',markersize = 1,label = 'Saturation adjustment')
plt.plot(T_w,z_w,'b',markersize = 1,label = 'Adiabatic parcel')
plt.xlabel('Temperature T [K]')
plt.ylabel('Height z [m]')
plt.ylim(bottom = 0)
plt.legend()
#plt.title('T(z)',fontsize = 24)
plt.savefig('T.png',format = "png")

fig, ax  =  plt.subplots()
ax.plot(r_v*1E3,z,'r',markersize = 1,label = '$r_v$ for saturation adjustment',linewidth = 3)
ax.plot(r_vw*1E3,z_w,'b',markersize = 1,label = '$r_v$ for adiabatic parcel')
ax.set_xlabel('Water vapor mixing ratio $r_v$ [g/kg]')
ax.set_ylabel('Height z [m]')
ax2 = plt.twiny()
ax2.set_xlabel('Liquid water mixing ratio $r_l$ [g/kg]')
ax2.plot(r_l*1E3,z,'r--',markersize = 1,label = '$r_l$ for saturation adjustment',linewidth = 3)
ax2.plot(r_lw*1E3,z_w,'b--',markersize = 1,label = '$r_l$ for adiabatic parcel')
fig.legend(bbox_to_anchor = (0.25, -0.15, 0.5, 0.5))
plt.ylim(0,z_max)
plt.savefig('r.png')

plt.figure(5)
plt.plot(S_w*1E2,z_w,'r',markersize = 1)
#plt.xlim(-.1,max(S_w)*1E2)
plt.xlabel('Supersaturation $S$ [%]')
plt.ylabel('Height z [m]')
plt.xlim(-.5,.5)
plt.ylim(bottom = 0)
#plt.title('Supersaturation',fontsize = 24)
plt.savefig('S.png',format = "png")

plt.figure(6)
plt.plot(var,z_w,'r',markersize = 1)
plt.xlabel('Variance of log($r$)')
plt.ylabel('Height z [m]')
#plt.title('Variance',fontsize = 24)
plt.savefig('var.png',format = "png")

plt.figure(7)
plt.plot(w,z_w,'r',markersize = 1)
plt.xlabel('Velocity $w$ [m/s]')
plt.ylabel('Height z [m]')
#plt.title('w',fontsize = 24)
plt.ylim(bottom = 0)
plt.savefig('w.png',format = "png")

plt.figure(9)
plt.plot(time,w,'r',markersize = 1)
plt.xlabel('Time $t$ [s]')
plt.ylabel('Velocity $w$ [m/s]')
#plt.title('w',fontsize = 24)
plt.xlim(0,max(time))
plt.savefig('wt.png',format = "png")

plt.figure(10)
time_cut = int(.9*len(time))
plt.plot(time[time_cut:],w[time_cut:],'r',markersize = 1)
plt.xlabel('Time $t$ [s]')
plt.ylabel('Velocity $w$ [m/s]')
#plt.title('w',fontsize = 24)
plt.xlim(time[time_cut],max(time))
plt.savefig('wt_cut.png',format = "png")
