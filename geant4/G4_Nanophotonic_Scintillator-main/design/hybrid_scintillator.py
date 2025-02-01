import torch

class AbsorptionCoeff:
    def __init__(self, γ:float, e:float, l:float):
        self.γ = γ # X/γ-rays absorption coefficient
        self.e = e # Effective electronic absorption coefficient 
        self.l = l # Visible light absorption coefficient

class GeneralAbsorptionCoeff:
    def __init__(self, scint:AbsorptionCoeff, sl:AbsorptionCoeff):
        self.scint = scint # Absorption coefficients of the scintillator material
        self.sl = sl # Absorption coefficients of the stopping layer

class ConversionRates:
    def __init__(self, C_sl:float, C_scint:float, Y:float):
        self.C_sl = C_sl # conversion rate from x/γ-rays to electrons in the stopping layer
        self.C_scint = C_scint # conversion rate from x/γ-rays to electrons in the scintillator
        self.Y = Y # scintillation yield - conversion rate from electrons to visible light in the scintillator

def n_γ_k(
    scintillatorThicknesses, # Vector of thicknesses of the scintillator layers
    deielctricThicknesses, # Vector of thicknesses of the dielectric layers
    μ:GeneralAbsorptionCoeff, # All absorption coefficients of the multilayer structure
    k:int # The index of the pair
    ):

    D_sl = (deielctricThicknesses[:k]).sum()
    D_scint = (scintillatorThicknesses[:k]).sum()
    n_γ = torch.exp(-μ.sl.γ * D_sl - μ.scint.γ * D_scint)
    return n_γ

def n_e_k(
        scintillatorThicknesses, # Vector of thicknesses of the scintillator layers
        deielctricThicknesses, # Vector of thicknesses of the dielectric layers
        μ:GeneralAbsorptionCoeff, # All absorption coefficients of the multilayer structure
        rates:ConversionRates, # Conversion rates of x/γ-rays, electrons, and light
        k:int, # The index of the pair
        n_e_km1 # Electron reaching the layers - serves as initial condition
    ):

    if k > 0:
        exp_scint_e_l = torch.exp(-μ.scint.e * scintillatorThicknesses[k-1] - μ.sl.e * deielctricThicknesses[k])
        ne_prev = n_e_km1 * exp_scint_e_l
    else:
        ne_prev = 0.

    n_γ = n_γ_k(scintillatorThicknesses, deielctricThicknesses, μ, k)

    μ_sl_γ_e = μ.sl.γ / (μ.sl.e - μ.sl.γ)
    exp_sl_γ_e = torch.exp(-μ.sl.γ * deielctricThicknesses[k]) - torch.exp(-μ.sl.e * deielctricThicknesses[k])
    zeta_sl = μ_sl_γ_e * exp_sl_γ_e
    ne_sl = n_γ * rates.C_sl * zeta_sl

    μ_scint_γ_e = μ.scint.γ / (μ.scint.e - μ.scint.γ)
    if k > 0:
        exp_scint_γ_e = torch.exp(-μ.scint.γ * scintillatorThicknesses[k-1]) - torch.exp(-μ.scint.e * scintillatorThicknesses[k-1])
    else:
        exp_scint_γ_e = 0.

    zeta_scint = μ_scint_γ_e * exp_scint_γ_e
    ne_scint = n_γ * rates.C_scint * zeta_scint
    
    ne = ne_prev + ne_sl + ne_scint
    return ne

def n_l_k(
        scintillatorThicknesses, # Vector of thicknesses of the scintillator layers
        deielctricThicknesses, # Vector of thicknesses of the dielectric layers
        μ:GeneralAbsorptionCoeff, # All absorption coefficients of the multilayer structure
        rates:ConversionRates, # Conversion rates of x/γ-rays, electrons, and light
        k:int, # The index of the pair
        n_e_km1 # Electron reaching the layers - serves as initial condition
    ):
    
    n_γ = n_γ_k(scintillatorThicknesses, deielctricThicknesses, μ, k)
    n_e = n_e_k(scintillatorThicknesses, deielctricThicknesses, μ, rates, k, n_e_km1)

    μ_scint_γ_l = μ.scint.γ / (μ.scint.l - μ.scint.γ)
    μ_scint_γ_e = μ.scint.γ / (μ.scint.e - μ.scint.γ)
    μ_scint_e_l = μ.scint.e / (μ.scint.l - μ.scint.e)

    exp_scint_γ_l = torch.exp(-μ.scint.γ * scintillatorThicknesses[k]) - torch.exp(-μ.scint.l * scintillatorThicknesses[k])
    exp_scint_e_l = torch.exp(-μ.scint.e * scintillatorThicknesses[k]) - torch.exp(-μ.scint.l * scintillatorThicknesses[k])
    print('μ.scint.γ', μ.scint.γ, 'μ.scint.l', μ.scint.l)
    ζ_scint_γ_l = μ_scint_γ_l * exp_scint_γ_l
    ζ_scint_e_l = μ_scint_e_l * exp_scint_e_l

    n_e_absorbed = ζ_scint_e_l * n_e
    cur_n_l = n_γ * torch.exp(-μ.sl.γ * deielctricThicknesses[k]) * rates.C_scint * μ_scint_γ_e * (ζ_scint_e_l - ζ_scint_γ_l)
    print('n_e_absorbed', n_e_absorbed, 'cur_n_l', cur_n_l, 'ζ_scint_γ_l', ζ_scint_γ_l, 'ζ_scint_e_l', ζ_scint_e_l)
    n_l = cur_n_l + n_e_absorbed
    return n_l, n_e

def total_n_l(
        scintillatorThicknesses, # Vector of thicknesses of the scintillator layers
        deielctricThicknesses, # Vector of thicknesses of the dielectric layers
        μ:GeneralAbsorptionCoeff, # All absorption coefficients of the multilayer structure
        rates:ConversionRates, # Conversion rates of x/γ-rays, electrons, and light
        numPairs = None
    ):

    if numPairs is None:
        numPairs = len(scintillatorThicknesses)
    N = 0
    prev_n_e = 0.
    for k in range(numPairs):
        cur_n_l, cur_n_e = n_l_k(scintillatorThicknesses, deielctricThicknesses, μ, rates, k, prev_n_e)
        
        # Computing the light attenuation until reaching the end of the structure
        D_scint_k = (scintillatorThicknesses[(k+1):]).sum()
        D_sl_k = (deielctricThicknesses[(k+1):]).sum()
        N += torch.exp(-(μ.scint.l * D_scint_k + μ.sl.l * D_sl_k)) * cur_n_l
        
        prev_n_e = cur_n_e 
    return N

def get_μ(μ_scint_γ, μ_scint_e, μ_scint_l, μ_sl_γ, μ_sl_e, μ_sl_l):
    μ_scint = AbsorptionCoeff(μ_scint_γ, μ_scint_e, μ_scint_l)
    μ_sl = AbsorptionCoeff(μ_sl_γ, μ_sl_e, μ_sl_l)
    μ = GeneralAbsorptionCoeff(μ_scint, μ_sl)
    return μ

def N_l_total(
    scintillatorThicknesses,
    deielctricThicknesses,
    μ_scint_γ:float, μ_scint_e:float, μ_scint_l:float, μ_sl_γ:float, μ_sl_e:float, μ_sl_l:float,
    C_sl:float, C_scint:float
    ):
    Y = 1.
    return total_n_l(
        scintillatorThicknesses,
        deielctricThicknesses,
        get_μ(μ_scint_γ, μ_scint_e, μ_scint_l, μ_sl_γ, μ_sl_e, μ_sl_l),
        ConversionRates(C_sl, C_scint, Y))
