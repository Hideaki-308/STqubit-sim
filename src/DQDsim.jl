using LinearAlgebra
using Plots

# %% Constants and basic matrices
const N = 2  # Number of sites

ann = [0.0 0.0;
    1.0 0.0]                   # Annihilation operator
cre = transpose(ann)             # Creation operator (transpose)
I2 = Matrix{Float64}(I, 2, 2)     # 2x2 Identity matrix
F = [1.0 0.0;
    0.0 -1.0]                  # Fermionic sign matrix

# Spin operators for each site (up and down)
c_up = kron(ann, I2)
c_down = kron(F, ann)
cdag_up = transpose(c_up)
cdag_down = transpose(c_down)

# %% Hamiltonian projection function
function proj_to_Nptcl(h::AbstractMatrix, num_particles::Int)
    states = collect(Iterators.product(ntuple(_ -> (0:1), 2 * N)...))
    indices = [i for (i, s) in enumerate(states) if sum(s) == num_particles]
    return h[indices, indices]
end

# %% Expand operator to full Hilbert space via tensor product
function make_matrix(opr, site_index::Int, N::Int)
    res = 1
    for i in N:-1:1
        if i > site_index
            res = kron(kron(F, F), res)
        elseif i == site_index
            res = kron(opr, res)
        else
            res = kron(kron(I2, I2), res)
        end
    end
    return res
end

# %% Operator dictionaries for each site and spin
c = Dict{String,Matrix{Float64}}()
cdag = Dict{String,Matrix{Float64}}()
for site in 1:N, spin in ["u", "d"]
    label = string(site, spin)
    mat = spin == "u" ? c_up : c_down
    matdag = spin == "u" ? cdag_up : cdag_down
    c[label] = make_matrix(mat, site, N)
    cdag[label] = make_matrix(matdag, site, N)
end

n = Dict{String,Matrix{Float64}}()
for label in keys(c)
    n[label] = cdag[label] * c[label]
end

function make_hamiltonian(mu, U, t, B, dB)
    # Initialize Hamiltonian matrix
    H = zeros(16, 16)

    # Chemical potential
    H += (mu / 2) * (n["1u"] + n["1d"]) - (mu / 2) * (n["2u"] + n["2d"])

    # On-site Coulomb interaction
    H += U * (n["1u"] * n["1d"]) + U * (n["2u"] * n["2d"])

    # Tunneling terms
    for (i1, i2) in [("1", "2"), ("2", "1")]
        H += t * (cdag["$(i1)u"] * c["$(i2)u"] + cdag["$(i1)d"] * c["$(i2)d"])
    end

    # Zeeman splitting
    H += -0.5(B + dB) * n["1u"] - 0.5(B - dB) * n["2u"] +
         0.5(B + dB) * n["1d"] + 0.5(B - dB) * n["2d"]

    # Project to subspace with fixed particle number
    H_proj = proj_to_Nptcl(H, 2)
    return H_proj
end

# %% Parameter sweep for Hubbard model
function plot_diagram()
    mu_list = range(-1.0, stop=1.0, length=61)
    U, t = 0.5, -0.05
    B, dB = 0.2, 0.1

    proj_dim = 6
    num_mu = length(mu_list)
    evals_mat = zeros(proj_dim, num_mu)

    for (mu_index, mu) in enumerate(mu_list)
        H = make_hamiltonian(mu, U, t, B, dB)
        evals_mat[:, mu_index] = eigen(H).values
    end

    # %% Plotting results
    plt = plot()
    for i in 1:proj_dim
        plot!(plt, mu_list, evals_mat[i, :], label="Branch $i")
    end
    xlabel!(plt, "μ")
    ylabel!(plt, "Eigenvalue")
    title!(plt, "Hubbard Hamiltonian Eigenvalues vs μ")
    display(plt)
end
