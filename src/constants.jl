"""
Some physical constants
"""
module constants

export epsilon0, mu0
export electron_mass
export proton_charge, proton_mass

# https://physics.nist.gov/cgi-bin/cuu/Value?ep0
const epsilon0 = 8.8541878128e-12 # F m^-1

# https://physics.nist.gov/cgi-bin/cuu/Value?mu0
const mu0 = 1.25663706212e-6 # N A^-2

# https://physics.nist.gov/cgi-bin/cuu/Value?me
const electron_mass = 9.109383701e-31 # kg

# https://physics.nist.gov/cgi-bin/cuu/Value?e
const proton_charge = 1.602176634e-19 # C

# https://physics.nist.gov/cgi-bin/cuu/Value?mp
const proton_mass = 1.67262192369e-27 # kg

end
