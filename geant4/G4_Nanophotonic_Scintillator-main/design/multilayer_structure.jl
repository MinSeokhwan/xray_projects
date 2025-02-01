struct AbsorptionCoeff
    γ::Float64 # X/γ-rays absorption coefficient
    e::Float64 # Effective electronic absorption coefficient 
    l::Float64 # Visible light absorption coefficient
end

struct GeneralAbsorptionCoeff
    scint::AbsorptionCoeff # Absorption coefficients of the scintillator material
    sl::AbsorptionCoeff # Absorption coefficients of the stopping layer
end

struct ConversionRates
    C_sl::Float64 # conversion rate from x/γ-rays to electrons in the stopping layer
    C_scint::Float64 # conversion rate from x/γ-rays to electrons in the scintillator
    Y::Float64 # scintillation yield - conversion rate from electrons to visible light in the scintillator
end

function n_γ_k(
    scintillatorThicknesses::Vector{Float64}, # Vector of thicknesses of the scintillator layers
    deielctricThicknesses::Vector{Float64}, # Vector of thicknesses of the dielectric layers
    μ::GeneralAbsorptionCoeff, # All absorption coefficients of the multilayer structure
    k::Int64 # The index of the pair
    )::Float64

    D_sl = sum(deielctricThicknesses[begin:(k-1)])
    D_scint = sum(scintillatorThicknesses[begin:(k-1)])
    n_γ = exp(-μ.sl.γ * D_sl - μ.scint.γ * D_scint)
    return n_γ
end

function n_e_k(
        scintillatorThicknesses::Vector{Float64}, # Vector of thicknesses of the scintillator layers
        deielctricThicknesses::Vector{Float64}, # Vector of thicknesses of the dielectric layers
        μ::GeneralAbsorptionCoeff, # All absorption coefficients of the multilayer structure
        rates::ConversionRates, # Conversion rates of x/γ-rays, electrons, and light
        k::Int64, # The index of the pair
        n_e_km1::Float64 # Electron reaching the layers - serves as initial condition
    )::Float64

    if k > 1
        exp_scint_e_l = exp(-μ.scint.e * scintillatorThicknesses[k-1] - μ.sl.e * deielctricThicknesses[k])
        ne_prev = n_e_km1 * exp_scint_e_l
    else
        ne_prev = 0.
    end

    n_γ = n_γ_k(scintillatorThicknesses, deielctricThicknesses, μ, k)

    μ_sl_γ_e = μ.sl.γ / (μ.sl.e - μ.sl.γ)
    exp_sl_γ_e = exp(-μ.sl.γ * deielctricThicknesses[k]) - exp(-μ.sl.e * deielctricThicknesses[k])
    zeta_sl = μ_sl_γ_e * exp_sl_γ_e
    ne_sl = n_γ * rates.C_sl * zeta_sl

    μ_scint_γ_e = μ.scint.γ / (μ.scint.e - μ.scint.γ)
    if k > 1
        exp_scint_γ_e = exp(-μ.scint.γ * scintillatorThicknesses[k-1]) - exp(-μ.scint.e * scintillatorThicknesses[k-1])
    else
        exp_scint_γ_e = 0.
    end
    zeta_scint = μ_scint_γ_e * exp_scint_γ_e
    ne_scint = n_γ * rates.C_scint * zeta_scint
    
    ne = ne_prev + ne_sl + ne_scint
    return ne
end

function n_l_k(
        scintillatorThicknesses::Vector{Float64}, # Vector of thicknesses of the scintillator layers
        deielctricThicknesses::Vector{Float64}, # Vector of thicknesses of the dielectric layers
        μ::GeneralAbsorptionCoeff, # All absorption coefficients of the multilayer structure
        rates::ConversionRates, # Conversion rates of x/γ-rays, electrons, and light
        k::Int64, # The index of the pair
        n_e_km1::Float64 # Electron reaching the layers - serves as initial condition
    )
    
    n_γ = n_γ_k(scintillatorThicknesses, deielctricThicknesses, μ, k)
    n_e = n_e_k(scintillatorThicknesses, deielctricThicknesses, μ, rates, k, n_e_km1)

    μ_scint_γ_l = μ.scint.γ / (μ.scint.l - μ.scint.γ)
    μ_scint_γ_e = μ.scint.γ / (μ.scint.e - μ.scint.γ)
    μ_scint_e_l = μ.scint.e / (μ.scint.l - μ.scint.e)

    exp_scint_γ_l = exp(-μ.scint.γ * scintillatorThicknesses[k]) - exp(-μ.scint.l * scintillatorThicknesses[k])
    exp_scint_e_l = exp(-μ.scint.e * scintillatorThicknesses[k]) - exp(-μ.scint.l * scintillatorThicknesses[k])

    ζ_scint_γ_l = μ_scint_γ_l * exp_scint_γ_l
    ζ_scint_e_l = μ_scint_e_l * exp_scint_e_l

    n_e_absorbed = ζ_scint_e_l * n_e
    n_l_k = n_γ * exp(-μ.sl.γ * deielctricThicknesses[k]) * rates.C_scint * μ_scint_γ_e * (ζ_scint_e_l - ζ_scint_γ_l)

    n_l = n_l_k + n_e_absorbed
    return n_l, n_e
end

function total_n_l(
        scintillatorThicknesses::Vector{Float64}, # Vector of thicknesses of the scintillator layers
        deielctricThicknesses::Vector{Float64}, # Vector of thicknesses of the dielectric layers
        μ::GeneralAbsorptionCoeff, # All absorption coefficients of the multilayer structure
        rates::ConversionRates # Conversion rates of x/γ-rays, electrons, and light
    )::Float64
    numPairs = size(scintillatorThicknesses)[1]
    N = 0
    prev_n_e = 0.
    for k = 1:numPairs
        cur_n_l, cur_n_e = n_l_k(scintillatorThicknesses, deielctricThicknesses, μ, rates, k, prev_n_e)
        
        # Computing the light attenuation until reaching the end of the structure
        D_scint_k = sum(scintillatorThicknesses[k+1:end])
        D_sl_k = sum(deielctricThicknesses[k+1:end])
        N += exp(-(μ.scint.l * D_scint_k + μ.sl.l * D_sl_k)) * cur_n_l
        
        prev_n_e = cur_n_e 
    end
    return N
end

function get_μ(μ_scint_γ, μ_scint_e, μ_scint_l, μ_sl_γ, μ_sl_e, μ_sl_l)::GeneralAbsorptionCoeff
    μ_scint = AbsorptionCoeff(μ_scint_γ, μ_scint_e, μ_scint_l)
    μ_sl = AbsorptionCoeff(μ_sl_γ, μ_sl_e, μ_sl_l)
    μ = GeneralAbsorptionCoeff(μ_scint, μ_sl)
    return μ
end

function N_l_total(
    scintillatorThicknesses::Vector{Float64},
    deielctricThicknesses::Vector{Float64},
    μ_scint_γ::Float64, μ_scint_e::Float64, μ_scint_l::Float64, μ_sl_γ::Float64, μ_sl_e::Float64, μ_sl_l::Float64,
    C_sl::Float64, C_scint::Float64
    )::Float64
    Y = 1.
    return total_n_l(
        scintillatorThicknesses,
        deielctricThicknesses,
        get_μ(μ_scint_γ, μ_scint_e, μ_scint_l, μ_sl_γ, μ_sl_e, μ_sl_l),
        ConversionRates(C_sl, C_scint, Y))
end