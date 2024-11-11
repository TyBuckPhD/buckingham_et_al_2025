import numpy as np
import matplotlib.pyplot as plt
from phd.core.helmholtz_decomposition import HelmholtzDecomposition

if __name__ == "__main__":
    hhd = HelmholtzDecomposition(config_path='inputs/type1_front_config.json')
    non_div, non_rot, harmonic = hhd.run()

stretching = []
for harm in harmonic:

    u_harmonic = harm[..., 0]
    v_harmonic = harm[..., 1]

    dy = 1000
    dx = 1000

    # Compute spatial derivatives
    dv_dy = np.gradient(v_harmonic, dy, axis=0) # WAIT. DOES IT NOT NEED TO BE THE WHOLE STRAIN?!
    stretching_deformation = dv_dy.mean()
    stretching.append(stretching_deformation)

    # # Plot the stretching deformation
    # plt.figure(figsize=(10, 8))
    # plt.contourf(dv_dy) 
    # plt.axis('equal')
    # plt.show()
    
plt.plot(stretching)
plt.ylim(0, 2e-5)
plt.show()