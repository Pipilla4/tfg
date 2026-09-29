import qutip as qt
import matplotlib.pyplot as plt
import numpy as np

beta = 1
steps = 100

# S
w_S=1
H_S = w_S*qt.sigmaz()/2
state_S = qt.qeye(2) / 2
# M
w_M=1
H_M = w_M*qt.sigmaz()/2
state_M =  (-beta * H_M).expm()
state_M = state_M / state_M.tr()
# A
w_A=1
H_A = w_A*qt.sigmaz()/2
state_A =  (-beta * H_A).expm()
state_A = state_A / state_A.tr()
#interaction
g=0.2
H_int = g/2 * (qt.tensor(qt.sigmax(), qt.sigmax())+ qt.tensor(qt.sigmay(), qt.sigmay()) + qt.tensor(qt.sigmaz(), qt.sigmaz()))
H0 = qt.tensor(H_S, qt.qeye(2), qt.qeye(2)) + qt.tensor(qt.qeye(2), H_M, qt.qeye(2))+ qt.tensor(qt.qeye(2), qt.qeye(2), H_A)

tau1 = 0.2
H_SM = H0 + qt.tensor(H_int, qt.qeye(2))
U_SM = (-1j * H_SM * tau1).expm()

tau2 = 0.2
H_MA = H0 + qt.tensor(qt.qeye(2), H_int)
U_MA = (-1j * H_MA * tau2).expm()

# evolution
energy0 = qt.expect(H_S, state_S).real
energy_diff = [] 
SMstate = qt.tensor(state_S, state_M)
for n in range(steps):
    # interaction
    SMAstate = U_SM * qt.tensor(SMstate, state_A) * U_SM.dag()
    SMAstate = U_MA * SMAstate * U_MA.dag()
    SMstate = SMAstate.ptrace([0,1])

    # enforce Hermiticity and renormalize to avoid numerical drift
    SMstate = (SMstate + SMstate.dag())/2
    SMstate = SMstate / SMstate.tr()

    # compute energy difference
    energy1 = qt.expect(qt.tensor(H_S, qt.qeye(2)), SMstate).real
    energy_diff.append(energy1 - energy0)
    energy0 = energy1

#plot
plt.plot(energy_diff)
plt.show()
