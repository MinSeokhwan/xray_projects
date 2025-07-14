import time
from matplotlib import pyplot as plt
from hybrid_scintillator import *
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

pairs = 30
mu_Ti = 1.107e2
mu_Cl = 5.725e1
mu_C  = 2.373
mu_O  = 5.952
TiFractions = 0.3923
OFractions = 0.4066
ClFractions = 0.0719
CFractions = 0.1290
density_sl = 1.3 + TiFractions/0.4 #g/cm^3
mu_gamma_sl = density_sl * 1e-4 * (mu_Ti * TiFractions + mu_O * OFractions + mu_Cl * ClFractions + mu_C * CFractions)
mu_gamma_scint = 1e-4 * 1.02 * (2.562e-1)
mu_electron_sl = 38.95632914365327 * 1e-3
mu_electron_scint = 25.77332803210813 * 1e-3
mu_sl_l = 0.4383617656171
mu_l_scint = 0.
C_sl = 1.
C_scint = 1.

μ = get_μ(mu_gamma_scint, mu_electron_scint, mu_l_scint, mu_gamma_sl, mu_electron_sl, mu_sl_l)
rates = ConversionRates(C_scint, C_sl, 1.)

initial_scintillator_thicknesses = 1. * torch.ones(pairs)
initial_sl_thicknesses = 0.1 * torch.ones(pairs)
initial_thicknessses = torch.cat((initial_scintillator_thicknesses , initial_sl_thicknesses))
initial_thicknessses_rand = torch.rand(2*pairs)
initial_thicknessses_rand[:pairs] += 1

theory_N = lambda thicknesses: total_n_l(thicknesses[:pairs], thicknesses[pairs:], μ, rates, pairs)


class Model(nn.Module):

    def __init__(self, x0):
        super(Model, self).__init__()
        self.thicknesses = nn.Parameter(x0.clone())

    def forward(self):
        return theory_N(self.thicknesses)

hybrid_scintillator_model = Model(initial_thicknessses_rand)

optimizer = optim.Adam(hybrid_scintillator_model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

ref = theory_N(initial_thicknessses).detach().numpy()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(initial_sl_thicknesses, 'r', label='stopping layer thicknesses')
ax.set_xlabel('SL thickness [um]')
ax.set_ylabel('stopping layer', color='r')

ax_twin = ax.twinx()
ax_twin.set_ylabel('scintillator', color='b')
line_twin, = ax_twin.plot(initial_scintillator_thicknesses, 'b', label='scintillator thicknesses')
# ax.set_xscale("log")
ax.set_ylim((0, 1))
ax_twin.set_ylim((0, 1))

def barrier(x):
    if x.min() <= 0:
        return torch.inf
    else:
        return 1 / x.min()

losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    output = hybrid_scintillator_model()
    loss = -output + barrier(hybrid_scintillator_model.thicknesses)
    losses.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()
    scheduler.step()


    line.set_ydata(hybrid_scintillator_model.thicknesses[pairs:].detach().numpy())
    line_twin.set_ydata(hybrid_scintillator_model.thicknesses[:pairs].detach().numpy())
    enhancement = output.detach().numpy() / ref
    ax.set_title('iter ' + str(epoch) + ' enhancement = ' + str(enhancement))
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

enhancement = theory_N(hybrid_scintillator_model.thicknesses) / theory_N(initial_thicknessses)
print('enhancement =', enhancement)


