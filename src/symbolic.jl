
using Symbolics

include("DQDsim.jl")

@variables mu U t B dB
H = make_hamiltonian(mu, U, t, B, dB)


