"""
CBCM Solver for Partially Compliant 4-Bar Mechanism (Example 4.2.1, Ma & Chen JMR 2016)
========================================================================================
Corrected geometry from Fig. 6:
  - A=(0,0), D=(0,-L_DA) directly below A
  - Beam DQ horizontal along +x, fixed-fixed
  - CQ perpendicular to beam at Q (vertical initially)
  - theta2 = 3pi/4 = 135° is the interior angle BCQ at vertex C
  - L_AB = L + L*cos(theta2), L_BC = L_DQ = L, L_CQ = L/20

10-variable coupled formulation with strict continuation.
"""

import numpy as np
from scipy.optimize import fsolve
import json

# === BCM Coefficients ===
G = np.array([[12.0, -6.0], [-6.0, 4.0]])
P_mat = np.array([[6.0/5, -1.0/10], [-1.0/10, 2.0/15]])
Q_mat = np.array([[-1.0/700, 1.0/1400], [1.0/1400, -11.0/6300]])
U_mat = P_mat.copy(); V_mat = Q_mat.copy()

# === Parameters ===
# Parameters from Ma & Chen (JMR 2016) Example 4.2.1
L = 0.1 # m
E = 1.4e9 # Pa (Polypropylene as specified in paper)
w_beam = 0.01 # m
t = 0.0017 # m (Selected to match 0.5 Nm torque peak in Fig 8)
I_beam = (w_beam * t**3) / 12
EI_beam = E * I_beam

# Geometry formulas:
# LAB = L + L*cos(h2), h2 = 3*pi/4
theta2 = 3 * np.pi / 4
L0 = L # Generic L scale
L_AB = L0 * (1.0 + np.cos(theta2)) # ~0.029289m
L_BC = L0 # 0.1m
L_DQ = L0 # 0.1m
L_CQ = L0 / 20.0 # 0.005m
N_elem = 3; L_e = L0/N_elem; theta0_base = 0.0

# === Assembly geometry ===
# CQ is perpendicular to beam at Q.
# At assembly: beam horizontal, Q = (L, -L_DA), C = Q + L_CQ*(0, 1) = (L, -L_DA + L_CQ)
# |C - B| = L_BC with B = (L_AB, 0) determines L_DA.
# LAD (L_DA) must satisfy assembly at theta1=0
def compute_assembly():
    dx = L0 - L_AB 
    # (L_DA - L_CQ)^2 = L_BC^2 - dx^2
    # L_DA = L_CQ + sqrt(L_BC^2 - dx^2)
    # Since L_BC = L0 and dx = L0*(1 + cos(135)), dx = L0*sin(45)?
    # dx = 0.1 * (1 - 0.7071) = 0.02929 ? No.
    # dx = L - L_AB = L - L(1+cos(h2)) = -L*cos(h2) = -L*cos(135) = 0.1 * 0.7071 = 0.07071
    # dy = sqrt(L^2 - (0.07071)^2) = 0.07071
    # L_DA = L_CQ + 0.07071 = 0.07571
    l_da = L_CQ + np.sqrt(L_BC**2 - (L0 - L_AB)**2)
    phi0 = np.arctan2(L_CQ - l_da, L0 - L_AB) # Direction B->C
    delta_cq = -np.pi/2 - phi0 # Makes CQ vertical (-90)
    
    l_bq = np.sqrt((L0-L_AB)**2 + l_da**2)
    beta = np.arctan2(-l_da, L0-L_AB)
    return l_da, phi0, delta_cq, l_bq, beta

L_DA, phi_BC_asm, delta_CQ, L_BQ, beta_BQ = compute_assembly()
D_pos = np.array([0.0, -L_DA])

print(f"Assembly geometry (Paper Ex 4.2.1):")
print(f"  L_AB = {L_AB*1000:.2f} mm")
print(f"  L_DA = {L_DA*1000:.2f} mm")
print(f"  L_BQ = {L_BQ*1000:.2f} mm")
print(f"  phi_BC_init = {np.degrees(phi_BC_asm):.2f}°")
print(f"  delta_CQ = {np.degrees(delta_CQ):.2f}°")
print(f"  beta_BQ = {np.degrees(beta_BQ):.2f}°")
print()

# === Helpers ===
def get_B(th1): return np.array([L_AB*np.cos(th1), L_AB*np.sin(th1)])

def get_Q_C_from_phi(B, phi):
    """Q and C positions from coupler angle phi_BC."""
    C = B + L_BC * np.array([np.cos(phi), np.sin(phi)])
    Q = C + L_CQ * np.array([np.cos(phi + delta_CQ), np.sin(phi + delta_CQ)])
    return Q, C

def get_tip_from_phi(phi):
    """Beam tip angle from coupler angle.
    CQ is always perpendicular to beam: direction Q->C = theta_tip + pi/2
    Direction C->Q = phi + delta_CQ, so Q->C = phi + delta_CQ + pi
    Therefore: theta_tip + pi/2 = phi + delta_CQ + pi
    theta_tip = phi + delta_CQ + pi/2
    """
    return phi + delta_CQ + np.pi/2

def bcm_element(dy_i, alpha_i, p_i):
    d = np.array([dy_i, alpha_i])
    K = G + p_i*P_mat + p_i**2*Q_mat
    fm = K @ d
    dx = (t**2*p_i)/(12.0*L_e**2) - 0.5*d@U_mat@d - p_i*d@V_mat@d
    return fm[0], fm[1], dx

# === 10-variable residual ===
def residual(x, theta1):
    dy, alpha, p = x[0:3], x[3:6], x[6:9]
    phi_BC = x[9]
    
    f = np.zeros(3); m = np.zeros(3); dx = np.zeros(3)
    for i in range(3):
        f[i], m[i], dx[i] = bcm_element(dy[i], alpha[i], p[i])
    h = np.array([0.0, alpha[0], alpha[0]+alpha[1]])
    res = np.zeros(10)
    
    # Eqs 1-6: Load equilibrium
    idx = 0
    for i in range(1, 3):
        ch, sh = np.cos(h[i]), np.sin(h[i])
        res[idx]   =  ch*f[i] + sh*p[i] - f[0]
        res[idx+1] = -sh*f[i] + ch*p[i] - p[0]
        res[idx+2] = (1+dx[i])*f[i] - dy[i]*p[i] + m[i] - m[i-1]
        idx += 3
    
    # Beam tip from CBCM
    Qx_b, Qy_b = D_pos[0], D_pos[1]
    for i in range(3):
        a = theta0_base + h[i]
        ca, sa = np.cos(a), np.sin(a)
        Qx_b += ca*L_e*(1+dx[i]) - sa*L_e*dy[i]
        Qy_b += sa*L_e*(1+dx[i]) + ca*L_e*dy[i]
    tip_beam = theta0_base + h[2] + alpha[2]
    
    # Coupler
    B = get_B(theta1)
    Q_c, C = get_Q_C_from_phi(B, phi_BC)
    tip_coupler = get_tip_from_phi(phi_BC)
    
    # Eqs 7-8: Position match
    res[6] = (Qx_b - Q_c[0]) / L
    res[7] = (Qy_b - Q_c[1]) / L
    
    # Eq 9: Tip angle match
    res[8] = tip_beam - tip_coupler
    
    # Eq 10: Moment equilibrium of coupler about B
    # Forces from beam at Q in global frame (base horizontal):
    #   Fy = f[0]*EI/Le^2 (transverse=+y), Fx = p[0]*EI/Le^2 (axial=+x)
    # Reaction on coupler: (-Fx, -Fy), moment (-m[2]*EI/Le)
    # Moment about B: (Qx-Bx)*(-Fy) - (Qy-By)*(-Fx) + (-m[2]*EI/Le) = 0
    # Normalized (divide by EI/Le^2):
    # -(Qx-Bx)*f[0] + (Qy-By)*p[0] - m[2]*Le = 0
    res[9] = (-(Q_c[0]-B[0])*f[0] + (Q_c[1]-B[1])*p[0] - m[2]*L_e) / L
    
    return res

def is_physical(sol):
    # Normalized dy, alpha, and p should be reasonable
    dy, alpha, p = sol[0:3], sol[3:6], sol[6:9]
    if np.any(np.abs(dy) > 5.0) or np.any(np.abs(alpha) > 5.0) or np.any(np.abs(p) > 20.0):
        return False
        
    # Also check actual beam tip position calculation match
    # Qx_b, Qy_b calculation from residual logic
    qx, qy = D_pos[0], D_pos[1]
    h = [0.0, alpha[0], alpha[0]+alpha[1]]
    for i in range(3):
        f_i, m_i, dx_i = bcm_element(dy[i], alpha[i], p[i])
        ang = theta0_base + h[i]
        ca, sa = np.cos(ang), np.sin(ang)
        qx += ca*L_e*(1+dx_i) - sa*L_e*dy[i]
        qy += sa*L_e*(1+dx_i) + ca*L_e*dy[i]
    
    # Mechanism size is ~0.1m. If tip is > 0.3m away, it's non-physical
    dist = np.sqrt((qx - D_pos[0])**2 + (qy - D_pos[1])**2)
    if dist > 0.3:
        print(f"DEBUG: Non-physical solution! dist={dist:.2f} q=({qx:.2f},{qy:.2f})")
        return False
    return True

# === Solver ===
def solve_at_angle(theta1, x_prev):
    def f(x): return residual(x, theta1)
    # Attempt 1: Strict continuation from previous
    sol, info, ier, msg = fsolve(f, x_prev, full_output=True, maxfev=10000)
    rn = np.linalg.norm(f(sol))
    
    if rn < 1e-4 and is_physical(sol):
        return sol, rn

    # Attempt 2: Random restarts if failed or non-physical
    best_physical_sol, best_physical_rn = None, 1e10
    
    for trial in range(50):
        # Wider perturbation
        x0_trial = x_prev + np.random.randn(10) * 0.5
        s, _, ie, _ = fsolve(f, x0_trial, full_output=True, maxfev=10000)
        r = np.linalg.norm(f(s))
        
        if r < 1e-4 and is_physical(s):
            # Check jump distance to stay on the same branch
            if np.linalg.norm(s - x_prev) < 2.0 or rn > 1e-2:
                return s, r
            if r < best_physical_rn:
                best_physical_sol, best_physical_rn = s, r
                
    if best_physical_sol is not None:
        return best_physical_sol, best_physical_rn
        
    # Final fallback: return previous solution with HIGH residual to mark failure
    return x_prev, 1e5

def solve_with_substeps(th_start, th_end, x_start, n_sub):
    thetas = np.linspace(th_start, th_end, n_sub + 1)
    x = x_start.copy()
    for k in range(1, n_sub + 1):
        x, rn = solve_at_angle(thetas[k], x)
        if rn > 1e-3: return x, rn # Abort if sub-stepping fails
    return x, rn

def beam_shape_points(x_sol, n_per=25):
    dy, alpha, p_arr = x_sol[0:3], x_sol[3:6], x_sol[6:9]
    fa, ma, dxa = np.zeros(3), np.zeros(3), np.zeros(3)
    for i in range(3):
        fa[i], ma[i], dxa[i] = bcm_element(dy[i], alpha[i], p_arr[i])
    h = np.array([0.0, alpha[0], alpha[0]+alpha[1]])
    px, py = [D_pos[0]], [D_pos[1]]
    cx, cy = D_pos[0], D_pos[1]
    for i in range(3):
        ang = theta0_base + h[i]
        ca, sa = np.cos(ang), np.sin(ang)
        dx_val = dxa[i]
        for j in range(1, n_per+1):
            xi = j/n_per
            # Linear distribution of axial deflection dx
            lx = L_e * xi * (1.0 + dx_val)
            
            # Interpolation for transverse deflection
            uy = (3*xi**2 - 2*xi**3)*dy[i] + (xi**3 - xi**2)*alpha[i]
            pi, fi, mi = p_arr[i], fa[i], ma[i]
            if abs(pi) > 1e-8:
                r = np.sqrt(abs(pi))
                try:
                    if pi > 0:
                        k1=(np.tanh(r)*(np.cosh(r*xi)-1)-np.sinh(r*xi)+r*xi)/r**3
                        k2=(np.cosh(r*xi)-1)/(r**2*np.cosh(r))
                    else:
                        k1=(np.sin(r*xi)-r*xi-np.tan(r)*(np.cos(r*xi)-1))/r**3
                        k2=(1-np.cos(r*xi))/(r**2*np.cos(r))
                    uy = k1*fi + k2*mi
                except: pass
            ly = L_e * uy
            px.append(cx + ca*lx - sa*ly)
            py.append(cy + sa*lx + ca*ly)
        
        # Advance to NEXT base exactly as in residual logic
        ex, ey = L_e*(1+dxa[i]), L_e*dy[i]
        cx += ca*ex - sa*ey
        cy += sa*ex + ca*ey
    return px, py

def compute_torque(x_sol, theta1):
    f1, _, _ = bcm_element(x_sol[0], x_sol[3], x_sol[6])
    B = get_B(theta1)
    return (B[0]*f1 - B[1]*x_sol[6]) * E*I_beam/L_e**2

def main():
    n_angles = 360
    theta1_arr = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    d_theta = theta1_arr[1] - theta1_arr[0]
    TOL = 1e-4
    
    solutions = [None]*n_angles
    residuals_arr = np.full(n_angles, 1e10)
    
    x0 = np.zeros(10)
    x0[9] = phi_BC_asm
    
    # Verify assembly solution
    r0 = residual(x0, 0.0)
    print(f"Assembly residual: {np.linalg.norm(r0):.2e}")
    
    # === Forward sweep ===
    print(f"Forward sweep ({n_angles} positions, d_theta={np.degrees(d_theta):.2f}°)...")
    x_prev = x0.copy()
    x_last_good = x0.copy()
    th_last_good = 0.0
    
    for i in range(n_angles):
        sol, rn = solve_at_angle(theta1_arr[i], x_prev)
        
        if rn >= TOL:
            for n_sub in [4, 16, 64, 256, 1024]:
                sol, rn = solve_with_substeps(th_last_good, theta1_arr[i], x_last_good, n_sub)
                if rn < TOL: break
        
        solutions[i] = sol.copy()
        residuals_arr[i] = rn
        
        if rn < TOL:
            x_prev = sol.copy()
            x_last_good = sol.copy()
            th_last_good = theta1_arr[i]
        else:
            x_prev = x_last_good.copy()
        
        if i % 90 == 0:
            print(f"  [{i+1}/{n_angles}] theta1={np.degrees(theta1_arr[i]):.1f}°  res={rn:.2e}")
    
    n1 = np.sum(residuals_arr < TOL)
    print(f"  Forward: {n1}/{n_angles}")
    
    # === Backward sweep ===
    failed = [i for i in range(n_angles) if residuals_arr[i] >= TOL]
    if failed:
        print(f"Backward sweep for {len(failed)} failures...")
        x_prev = None
        for i in range(n_angles-1, -1, -1):
            if residuals_arr[i] < TOL:
                x_prev = solutions[i].copy()
                th_last_good = theta1_arr[i]
                continue
            if x_prev is None: continue
            sol, rn = solve_at_angle(theta1_arr[i], x_prev)
            if rn >= TOL:
                for n_sub in [4, 16, 64, 256]:
                    sol, rn = solve_with_substeps(th_last_good, theta1_arr[i], x_prev, n_sub)
                    if rn < TOL: break
            if rn < residuals_arr[i]:
                solutions[i] = sol.copy()
                residuals_arr[i] = rn
            x_prev = solutions[i].copy()
            th_last_good = theta1_arr[i]
    
    nf = np.sum(residuals_arr < TOL)
    print(f"Final: {nf}/{n_angles} converged")
    
    # Smoothness
    mj = max(np.max(np.abs(solutions[i]-solutions[i-1])) for i in range(1, n_angles))
    print(f"Max jump: {mj:.6f}")
    
    failed = [i for i in range(n_angles) if residuals_arr[i] >= TOL]
    if failed:
        print(f"Failed angles: {[f'{np.degrees(theta1_arr[i]):.0f}' for i in failed[:20]]}...")
    
    # === Final Sanitation Pass ===
    # Even if solver thought it converged, if any point is way out of bounds, it's garbage.
    print("Sanitizing results...")
    n_sanitized = 0
    for i in range(1, n_angles):
        bx, by = beam_shape_points(solutions[i])
        outlier = False
        if any(abs(x) > 0.3 for x in bx) or any(abs(y + L_DA) > 0.3 for y in by):
            outlier = True
        
        if outlier:
            solutions[i] = solutions[i-1].copy()
            residuals_arr[i] = 1e6
            n_sanitized += 1
    if n_sanitized > 0:
        print(f"Sanitized {n_sanitized} outlier frames.")

    # === Output ===
    results = {
        'mechanism_params': {
            'L': L, 't': t, 'w': w_beam, 'E': E,
            'L_AB': L_AB, 'L_BC': L_BC, 'L_CQ': L_CQ,
            'L_DA': L_DA, 'L_DQ': L_DQ,
            'theta2_deg': float(np.degrees(theta2)),
            'N_elem': N_elem
        },
        'frames': []
    }
    for i in range(n_angles):
        th1 = theta1_arr[i]
        sol = solutions[i]
        B = get_B(th1)
        Q_c, C = get_Q_C_from_phi(B, sol[9])
        bx, by = beam_shape_points(sol)
        tp = compute_torque(sol, th1)
        results['frames'].append({
            'theta1_deg': float(np.degrees(th1)),
            'residual': float(residuals_arr[i]),
            'converged': bool(residuals_arr[i] < TOL),
            'joints': {
                'A': [0.0, 0.0],
                'B': [float(B[0]), float(B[1])],
                'C': [float(C[0]), float(C[1])],
                'Q': [float(Q_c[0]), float(Q_c[1])],
                'D': [float(D_pos[0]), float(D_pos[1])]
            },
            'beam_shape': {'x':[float(v) for v in bx], 'y':[float(v) for v in by]},
            'torque_Nm': float(tp),
        })
    
    with open('mechanism_data.json', 'w') as fp:
        json.dump(results, fp, indent=2)
    print("Saved mechanism_data.json")

if __name__ == '__main__':
    main()
