include("multilayer_structure.jl")
import Pkg; Pkg.add("Optim"); Pkg.add("Flux")
using Optim, Flux

pairs = 1
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
mu_electron_sl = 38.95632914365327
mu_electron_scint =25.77332803210813 
mu_sl_l = 0.4383617656171
mu_l_scint = 0.
C_sl = 10.
C_scint = 1.

μ = get_μ(mu_gamma_scint, mu_electron_scint, mu_l_scint, mu_gamma_sl, mu_electron_sl, mu_sl_l)
rates = ConversionRates(C_scint, C_sl, 1.)

initial_scintillator_thicknesses = 1. * ones(pairs)
initial_sl_thicknesses = 0.1 * ones(pairs)
initial_thicknessses = [initial_scintillator_thicknesses , initial_sl_thicknesses]
initial_thicknessses = reduce(hcat, initial_thicknessses)
theory_N(thicknesses) = total_n_l(thicknesses[begin:pairs], thicknesses[pairs:end], μ, rates)

result = optimize(theory_N, initial_scintillator_thicknesses, Newton(); autodiff = :forward)
