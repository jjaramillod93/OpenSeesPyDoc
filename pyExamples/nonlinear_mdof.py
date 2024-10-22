import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np

# Base units
m = 1                        # Meters
s = 1                        # Seconds
kN = 1                       # Kilo Newtons

# Derivated units
g = 9.81*m/s**2
cm = 1e-2*m
mm = 1e-3*m

# Parameters - data
n = 3                        # N° DOF to use
h = 0.05                     # Samping based
dt = 0.02                    # Time step
dt_out = 0.01                # Output time step
tFinal = 32                  # Stop time
mass = 208*kN/g              # Mass/floor 
Py = 400*kN                  # Yielding strength
K1 = 1155*kN/cm              # Stiffness/floor
K2 = 1092*kN/cm
K3 = 1073*kN/cm
b = 0.01                     # 

######## Model ###############
ops.wipe()					                     # clear memory of all past model definitions
ops.model('basic', '-ndm', 1, '-ndf', 1) 		 # Define the model builder, ndm=#dimension, ndf=#dofs

# Create nodes
ops.node(0, 0)
ops.node(1, 0, '-mass', mass)
ops.node(2, 0, '-mass', mass)
ops.node(3, 0, '-mass', mass)

# Define boundary condition
ops.fix(0, 1)

# Material definition
ops.uniaxialMaterial('Steel01', 1, Py, K1, b)
ops.uniaxialMaterial('Steel01', 2, Py, K2, b)
ops.uniaxialMaterial('Steel01', 3, Py, K3, b)

# Element definition
ops.element('zeroLength', 1, 0, 1, '-mat',    1   , '-dir', 1, '-doRayleigh', 1)
ops.element('zeroLength', 2, 1, 2, '-mat',    2   , '-dir', 1, '-doRayleigh', 1)
ops.element('zeroLength', 3, 2, 3, '-mat',    3   , '-dir', 1, '-doRayleigh', 1)

# Set Rayleigh damping
w1, w2 = np.array(ops.eigen('-fullGenLapack', 2))**0.5 # '-fullGenLapack'
a0 = 2*h*w1*w2/(w1+w2)
a1 = 2*h/(w1+w2)
ops.rayleigh(a0, .0, .0, a1) # RAYLEIGH damping

# Define the dynamic analysis 
load_tag = 1
patter_tag = 1
direc = 1
ops.timeSeries('Path', load_tag, '-dt', dt, '-filePath', r'pyExamples\el_centro.th', '-factor', g)
ops.pattern('UniformExcitation', patter_tag, direc, '-accel', load_tag)

# Run the dynamic analysis
ops.wipeAnalysis()
ops.algorithm('Newton')
ops.system('BandGen')
ops.numberer('Plain')
ops.constraints('Plain')
Gamma = 0.5
Beta = 0.25
ops.integrator('Newmark', Gamma, Beta)
ops.analysis('Transient')
tol = 1.0e-12
itrs = 100
ops.test('NormUnbalance', tol, itrs)
#
num_steps = int(tFinal/dt_out+1)
t = np.arange(0.0, num_steps*dt_out, dt_out)[:num_steps]
rd = np.zeros((n, num_steps))                               # Relative displacements with respect to the base
#
for i in range(num_steps):
    ops.analyze(1, dt_out)
    rd[:,i] = [ops.nodeDisp(1, 1), ops.nodeDisp(2, 1), ops.nodeDisp(3, 1)]
ops.wipe()

rd /= mm

fig, axs = plt.subplots(n, figsize=(12, 6), sharex=True, sharey=True)

ax = axs[0]
ax.plot(t, rd[0,:], 'k', label='Floor 1')
ax.set_xlim(.0, t[-1])
ax.legend()

ax = axs[1]
ax.plot(t, rd[1,:], 'k', label='Floor 2')
ax.set_ylabel('Relative Displacement [mm]')
ax.legend()

ax = axs[2]
ax.plot(t, rd[2,:], 'k', label='Floor 3')
ax.set_xlabel('Time [s]')
ax.legend()

plt.savefig(r'D:\workspace\OpenSeesPyDoc\_static\nonlinear_mdof.jpg')
plt.tight_layout()
plt.show()


