import sympy as sp
 
# Define x, t, hx, ht
x, t, hx, ht = sp.symbols('x t hx ht', positive=True)
sp.init_printing()

# Bilinear basis on [0,hx]×[0,ht] trial shapes
phi1 = (hx - x)*(ht - t)/(hx*ht)
phi2 =     x *(ht - t)/(hx*ht)
phi3 =     x *    t /(hx*ht)
phi4 = (hx - x)*   t /(hx*ht)
phi = [phi1, phi2, phi3, phi4]


# Space‐stiffness: integral dx phi_i dx phi_j dx dt
dphi_dx = [sp.diff(f, x) for f in phi]
K_space = sp.Matrix(4,4, lambda i,j:
    sp.simplify(sp.integrate(dphi_dx[i]*dphi_dx[j], (x,0,hx),(t,0,ht)))
)

# Mass matrix: integral phi_i phi_j dx dt
K_mass = sp.Matrix(4,4, lambda i,j:
    sp.simplify(sp.integrate(phi[i]*phi[j], (x,0,hx),(t,0,ht)))
)

# Time matrix: integral phi_i partial_t phi_j
dphi_dt = [sp.diff(f, t) for f in phi]
K_time = sp.Matrix(4,4, lambda i,j:
    sp.simplify(sp.integrate(phi[i]*dphi_dt[j], (x,0,hx), (t,0,ht)))
    )

print("Space matrix:\n");  sp.pprint(K_space)
print("\n\nMass  matrix:\n");  sp.pprint(K_mass)
print("\n\nTime matrix:\n");  sp.pprint(K_time)