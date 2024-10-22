'''
=========================================================================================================================
================================================== NonLinear MDOF =======================================================
=========================================================================================================================

By M. Eng. Joseph Jaramillo, National University of Engineering
e-mail: jjaramillod@uni.edu.pe
Date - 22/10/2024

This example model a multi-degree-of-freedom (MDOF) damped system commonly used in earthquake engineering of a three-story 
building. It conducts a nonlinear dynamic (time history)  analysis using the El centro 1940 earthquake as the input ground
motion.
'''

import openseespy.opensees as ops
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import plot_conf
# Base units
m = 1                        # Meters
s = 1                        # Seconds
kN = 1                       # Kilo Newtons

# Derivated units
g = 9.81*m/s**2              # Gravity
cm = 1e-2*m                  # Centimeter
mm = 1e-3*m                  # Milimeter
Ton = kN*s**2/m              # Ton

# Parameters - data
N = 3                        # N° DOF
h = 0.05                     # Damping ratio
dt = 0.02                    # Time step
dt_out = 0.01                # Output time step
tFinal = 35                  # Analysis stop time
m1 = 0.1*Ton                 # Mass / floor 
m2 = 0.1*Ton
m3 = 0.1*Ton 
Py1 = 0.55*kN                 # Yielding strength / floor
Py2 = 0.45*kN                 
Py3 = 0.30*kN                 
K1 = 60*kN/m                 # Stiffness / floor
K2 = 50*kN/m
K3 = 30*kN/m
b = 0.01                     # Strain-hardening ratio

######## Model ###############
ops.wipe()					                     # clear memory of all past model definitions
ops.model('basic', '-ndm', 1, '-ndf', 1) 		 # Define the model builder, ndm=#dimension, ndf=#dofs

# Create nodes
ops.node(0, 0)
ops.node(1, 0, '-mass', m1)
ops.node(2, 0, '-mass', m2)
ops.node(3, 0, '-mass', m3)

# Define boundary condition
ops.fix(0, 1)

# Material definition
ops.uniaxialMaterial('Steel01', 1, Py1, K1, b)
ops.uniaxialMaterial('Steel01', 2, Py2, K2, b)
ops.uniaxialMaterial('Steel01', 3, Py3, K3, b)

# Element definition
ops.element('zeroLength', 1, 0, 1, '-mat',    1   , '-dir', 1, '-doRayleigh', 1)
ops.element('zeroLength', 2, 1, 2, '-mat',    2   , '-dir', 1, '-doRayleigh', 1)
ops.element('zeroLength', 3, 2, 3, '-mat',    3   , '-dir', 1, '-doRayleigh', 1)

# Set Rayleigh damping
w1, w2, w3 = np.array(ops.eigen('-fullGenLapack', 3))**0.5
a0 = 2*h*w1*w2/(w1+w2)
a1 = 2*h/(w1+w2)
ops.rayleigh(a0, .0, .0, a1)                     # RAYLEIGH damping

# Natural periods
print('\nNatural periods:')
print(f'T1 = {2*np.pi/w1:.3f} [s]')
print(f'T2 = {2*np.pi/w2:.3f} [s]')
print(f'T3 = {2*np.pi/w3:.3f} [s]')

# Define the dynamic analysis 
load_tag = 1
patter_tag = 1
direc = 1
gm = np.genfromtxt(r'pyExamples\el_centro.th')   # Reading ground motion
ops.timeSeries('Path', load_tag, '-dt', dt, '-values', *gm, '-factor', g)
ops.pattern('UniformExcitation', patter_tag, direc, '-accel', load_tag)

# Run the dynamic analysis
Gamma = 0.5
Beta = 0.25
tol = 1.0e-12
itrs = 100
ops.wipeAnalysis()
ops.algorithm('Newton')
ops.system('BandGen')
ops.numberer('Plain')
ops.constraints('Plain')
ops.integrator('Newmark', Gamma, Beta)
ops.analysis('Transient')
ops.test('NormUnbalance', tol, itrs)
#
num_steps = int(tFinal/dt_out+1)
t = np.arange(0.0, num_steps*dt_out, dt_out)[:num_steps]
rA = np.zeros((N, num_steps))                    # Relative acceleration with respect to the ground
rD = np.zeros((N, num_steps))                    # Relative displacements with respect to the ground
eF = np.zeros((N, num_steps))                    # Spring force

for i in range(num_steps):
    ops.analyze(1, dt_out)
    rD[:,i] = [ops.nodeDisp(1, 1), ops.nodeDisp(2, 1), ops.nodeDisp(3, 1)]
    eF[:,i] = [ops.eleForce(1, 1), ops.eleForce(2, 1), ops.eleForce(3, 1)]
    rA[:,i] = [ops.nodeAccel(1, 1), ops.nodeAccel(2, 1), ops.nodeAccel(3, 1)]
ops.wipe()

#### Matplotlib plots ####
# Plotting ground and relative accelerations
z = lambda x: np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)

fig, axs = plt.subplots(N+1, figsize=(8, 5), sharex=True, sharey=True)
fig.suptitle('Relative Accelerations')

t_gm = np.arange(0.0, len(gm)*dt, dt)[:len(gm)]
gm *= g
peak = z(gm)
axs[0].plot(t_gm, gm, f'k', label=f'Ground   peak: {peak:.2f} [m/s2]')

for i, ax in enumerate(axs[1:]):
    peak = z(rA[i])
    ax.plot(t, rA[i], f'C{i+1}', label=f'Floor {i+1}   peak: {peak:.2f} [m/s2]')

for ax in axs:
    ax.set_xlim(.0, tFinal)
    ax.set_ylim(-8, 8)
    ax.grid()
    ax.legend(loc=(0.72, 1.0))
    ax.label_outer()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(4))

axs[1].set_ylabel('Acceleration [m/s2]', loc='top')
axs[1].yaxis.set_label_coords(-0.05, .5)
axs[3].set_xlabel('Time [s]')

plt.subplots_adjust(0.08, 0.1, 0.97, 0.9, 0.1, 0.4)
plt.savefig(r'D:\workspace\OpenSeesPyDoc\_static\nonlinear_mdof_rel_accel.jpg', )
# plt.show()

# Plotting relative displacements
rD /= mm
fig, axs = plt.subplots(N, figsize=(8, 3.8))
fig.suptitle('Relative Displacements')

for i, ax in enumerate(axs):
    peak = z(rD[i])
    ax.plot(t, rD[i], f'C{i+1}', label=f'Floor {i+1}   peak: {peak:.2f} [mm]')
    ax.set_xlim(.0, tFinal)
    ax.set_ylim(-75, 75)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.grid()
    ax.legend(loc=(0.72, 1.0))
    ax.label_outer()
axs[1].set_ylabel('Displacement [mm]')
axs[2].set_xlabel('Time [s]')
plt.subplots_adjust(0.08, 0.1, 0.97, 0.9, 0.1, 0.4)
plt.savefig(r'D:\workspace\OpenSeesPyDoc\_static\nonlinear_mdof_rel_disp.jpg', )
plt.show()


# # fig, axs = plt.subplots(1, N, figsize=(10, 5), sharex=True, sharey=True)

# # ax = axs[0]
# # ax.plot(rD[0,:], -eF[0,:])


# # ax = axs[1]
# # ax.plot(rD[1,:] - rD[0,:], -eF[1,:])

# # ax = axs[2]
# # ax.plot(rD[2,:] - rD[1,:], -eF[2,:])


# # plt.show()

