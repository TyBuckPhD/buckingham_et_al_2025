import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
from wrf import getvar, to_np, get_cartopy, ALL_TIMES
from netCDF4 import Dataset
from pynhhd import nHHD
from scipy.ndimage import rotate
import matplotlib as mpl
from shapely import geometry

data_type = '1'
event = {'1': 'wrfout_d03_2011-11-29_08-14_00_00.nc',
         '2': 'wrfout_d03_2005-11-24_12_00_00'}
event = event[data_type]
fp = '/Users/tybuckingham/Desktop/phd/data'

def load_wrf(event, fp):    
    
    files = os.path.join(fp, 'type'+data_type, event)
    ds_nc = Dataset(files)

    # Load in temp var for metadata and np.arrays of calc vars.
    wrf_var = getvar(ds_nc, 'ua', timeidx = ALL_TIMES) # Temp var.
    u = to_np(wrf_var)
    # u = to_np(getvar(ds_nc, 'ua', timeidx = ALL_TIMES))
    v = to_np(getvar(ds_nc, 'va', timeidx = ALL_TIMES))
    w = to_np(getvar(ds_nc, 'wa', timeidx = ALL_TIMES))
    hgts = to_np(getvar(ds_nc, 'height_agl'))
    
    lats = to_np(getvar(ds_nc, 'lat'))
    lons = to_np(getvar(ds_nc, 'lon'))
    
    return ds_nc, wrf_var, u, v, w, hgts, lats, lons

ds_nc, wrf_var, u, v, w, hgts, lats, lons = load_wrf(event, fp)

##############################################################################

# Frontal zones (for 8-11 file).
ymin = [250] * len(v)
ymax = [850] * len(v)

xmin_dev = list(np.linspace(280, 385, 24))
xmax_dev = list(np.linspace(450, 555, 24))
angle_dev = list(np.linspace(-14, -8, 24))

xmin_dec = list(np.linspace(390, 540, 18))
xmax_dec = list(np.linspace(560, 710, 18))
angle_dec = list(np.linspace(-7.8, -4.5, 18))

xmin = xmin_dev + xmin_dec 
xmax = xmax_dev + xmax_dec
angle = angle_dev + angle_dec

##############################################################################

# Define a new domain that centralised on the front.
def focus_domain(u, v, ymin, ymax, xmin, xmax, lats, lons, angle):

    def plot_rectangle(xmin, xmax, ymin, ymax):
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xs, ys, color = 'black')
            
    u_domain = []
    v_domain = []
    lats_domain = []
    lons_domain = []
        
    for i in range(len(v)):
        v_ref = v[i][5] * (np.cos(np.radians(angle[i])))
        u_ref = u[i][5] * (np.cos(np.radians(angle[i])))
        
        v_rot = rotate(v_ref, 
                       angle = angle[i], 
                       reshape = True)
        u_rot = rotate(u_ref, 
                       angle = angle[i], 
                       reshape = True)

        lats_rot = rotate(lats, 
                          angle = angle[i], 
                          reshape = True)
        lons_rot = rotate(lons, 
                          angle = angle[i], 
                          reshape = True)
        
        border = 140
        v_rot = v_rot[int(ymin[i] - border):int(ymax[i] + border), 
                      int(xmin[i] - border):int(xmax[i] + border)]
        u_rot = u_rot[int(ymin[i] - border):int(ymax[i] + border), 
                      int(xmin[i] - border):int(xmax[i] + border)]        

        lats_rot = lats_rot[int(ymin[i] - border):int(ymax[i] + border), 
                            int(xmin[i] - border):int(xmax[i] + border)]
        lons_rot = lons_rot[int(ymin[i] - border):int(ymax[i] + border), 
                            int(xmin[i] - border):int(xmax[i] + border)]  
        
        u_domain.append(u_rot)
        v_domain.append(v_rot)
        lats_domain.append(lats_rot)
        lons_domain.append(lons_rot)
        
        plt.figure(figsize = (5, 9))
        plt.contourf(v_rot)
        plot_rectangle(border, 
                       border + int(xmax[i] - xmin[i]), 
                       border, 
                       border + int(ymax[i] - ymin[i]))
        plt.show() 

    return u_domain, v_domain, lats_domain, lons_domain

def frontal_domain(u, v, ymin, ymax, xmin, xmax, lats, lons, angle):

    def plot_rectangle(xmin, xmax, ymin, ymax):
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xs, ys, color = 'black')
            
    u_front = []
    v_front = []
    lats_front = []
    lons_front = []

    for i in range(len(v)):
        v_ref = v[i][5] * (np.cos(np.radians(angle[i])))
        u_ref = u[i][5] * (np.cos(np.radians(angle[i])))
        
        v_rot = rotate(v_ref, 
                       angle = angle[i], 
                       reshape = True)
        u_rot = rotate(u_ref, 
                       angle = angle[i], 
                       reshape = True)

        lats_rot = rotate(lats, 
                          angle = angle[i], 
                          reshape = True)
        lons_rot = rotate(lons, 
                          angle = angle[i], 
                          reshape = True)
        
        v_rot = v_rot[int(ymin[i]):int(ymax[i]), 
                      int(xmin[i]):int(xmax[i])]
        u_rot = u_rot[int(ymin[i]):int(ymax[i]), 
                      int(xmin[i]):int(xmax[i])]        

        lats_rot = lats_rot[int(ymin[i]):int(ymax[i]), 
                            int(xmin[i]):int(xmax[i])]
        lons_rot = lons_rot[int(ymin[i]):int(ymax[i]), 
                            int(xmin[i]):int(xmax[i])] 
        
        u_front.append(u_rot)
        v_front.append(v_rot)
        lats_front.append(lats_rot)
        lons_front.append(lons_rot)
        
        plt.figure(figsize = (2, 9))
        plt.contourf(v_rot)
        plt.show() 

    return u_front, v_front, lats_front, lons_front

def hhd_full(u, v):
    
    div_full = []
    rot_full = []
    env_full = []
    
    for i in range(len(u)):
        # Structure the arrays correctly for decomposition.
        windfield = np.stack((u[i][5], v[i][5]))
        windfield = np.transpose(windfield, (1, 2, 0))       

        # Calculate the Helmholtz-Hodge decomposition on a limited domain.
        # Transformed grid spacing.
        dims = (u[i][5].shape[0], u[i][5].shape[1])
        dx = (1000, 1000)

        # Perform the decomposition.
        nhhd = nHHD(grid = dims, spacings = dx)
        nhhd.decompose(windfield)
    
        # Extract the relevant components.
        rot = nhhd.r
        div = nhhd.d
        harmonic = nhhd.h

        # Transpose to correct plotting dimension order.
        div =  np.transpose(div, (2, 0, 1))  
        rot = np.transpose(rot, (2, 0, 1))  
        harmonic =  np.transpose(harmonic, (2, 0, 1))        

        div_full.append(div)
        rot_full.append(rot)
        env_full.append(harmonic)
    
    return div_full, rot_full, env_full

# Decompose the front-centric domain!
def hhd_domain(u_domain, v_domain, angle):
    
    div_domain = []
    rot_domain = []
    env_domain = []
    
    for i in range(len(u_domain)):
        # Structure the arrays correctly for decomposition.
        windfield = np.stack((u_domain[i], v_domain[i]))
        windfield = np.transpose(windfield, (1, 2, 0))       

        # Calculate the Helmholtz-Hodge decomposition on a limited domain.
        # Transformed grid spacing.
        dims = (u_domain[i].shape[0], u_domain[i].shape[1])
        dx = (1000 * (np.cos(np.radians(angle[i]))), 
              1000 * (np.cos(np.radians(angle[i]))))

        # Perform the decomposition.
        nhhd = nHHD(grid = dims, spacings = dx)
        nhhd.decompose(windfield)
    
        # Extract the relevant components.
        rot = nhhd.r
        div = nhhd.d
        harmonic = nhhd.h

        # Transpose to correct plotting dimension order.
        div =  np.transpose(div, (2, 0, 1))  
        rot = np.transpose(rot, (2, 0, 1))  
        harmonic =  np.transpose(harmonic, (2, 0, 1))        

        div_domain.append(div)
        rot_domain.append(rot)
        env_domain.append(harmonic)
    
    return div_domain, rot_domain, env_domain

def hhd_front(u_front, v_front, angle):
    
    div_front = []
    rot_front = []
    
    for i in range(len(u_domain)):
        # Structure the arrays correctly for decomposition.
        windfield = np.stack((u_front[i], v_front[i]))
        windfield = np.transpose(windfield, (1, 2, 0))       

        # Calculate the Helmholtz-Hodge decomposition on a limited domain.
        # Transformed grid spacing.
        dims = (u_front[i].shape[0], u_front[i].shape[1])
        dx = (1000 * (np.cos(np.radians(angle[i]))), 
              1000 * (np.cos(np.radians(angle[i]))))

        # Perform the decomposition.
        nhhd = nHHD(grid = dims, spacings = dx)
        nhhd.decompose(windfield)
    
        # Extract the relevant components.
        rot = nhhd.r
        div = nhhd.d

        # Transpose to correct plotting dimension order.
        div =  np.transpose(div, (2, 0, 1))  
        rot = np.transpose(rot, (2, 0, 1))        

        div_front.append(div)
        rot_front.append(rot)
  
    return div_front, rot_front

def subtract_front(u, v, env_domain, angle):
    
    env_front = []
    flow_nofront = []
      
    for i in range(len(v)):
        v_ref = v[i][5] * (np.cos(np.radians(angle[i])))
        u_ref = u[i][5] * (np.cos(np.radians(angle[i])))
        
        v_rot = rotate(v_ref, 
                       angle = angle[i], 
                       reshape = True)
        u_rot = rotate(u_ref, 
                       angle = angle[i], 
                       reshape = True)
        
        border = 140
        v_rot = v_rot[int(ymin[i] - border):int(ymax[i] + border), 
                      int(xmin[i] - border):int(xmax[i] + border)]
        u_rot = u_rot[int(ymin[i] - border):int(ymax[i] + border), 
                      int(xmin[i] - border):int(xmax[i] + border)]

        xcorner = ycorner = border

        env_u = env_domain[i][0]
        env_v = env_domain[i][1]

        env_u_front = env_u[border:border + int(ymax[i] - ymin[i]), 
                            border:border + int(xmax[i] - xmin[i])]
        env_v_front = env_v[border:border + int(ymax[i] - ymin[i]), 
                            border:border + int(xmax[i] - xmin[i])]

        u_rot[ycorner : ycorner + env_u_front.shape[0],
              xcorner : xcorner + env_u_front.shape[1]] = env_u_front
        v_rot[ycorner : ycorner + env_v_front.shape[0],
              xcorner : xcorner + env_v_front.shape[1]] = env_v_front
        
        uv_env_front = np.stack((env_u_front, env_v_front))
        uv_nofront = np.stack((u_rot, v_rot))
        
        env_front.append(uv_env_front)
        flow_nofront.append(uv_nofront)
        
    return env_front, flow_nofront

def calculate_strain_full(env_full):
    
    strain_full = []
    strain_full_mean = []
    for i in range(len(env_full)):
        v_env = env_full[i][1]
        delta = 1000
    
        dv_envdy = np.gradient(v_env, delta, axis = 0)
        mean = np.mean(dv_envdy)
        
        strain_full.append(dv_envdy)
        strain_full_mean.append(mean)
    
    return strain_full, strain_full_mean

def calculate_strain_front(env_front, angle):
    
    strain_front = []
    strain_front_mean = []
    for i in range(len(env_front)):
        v_env = env_front[i][1]
        delta = 1000 * (np.cos(np.radians(angle[i])))
    
        dv_envdy = np.gradient(v_env, delta, axis = 0)
        mean = np.mean(dv_envdy)
        
        strain_front.append(dv_envdy)
        strain_front_mean.append(mean)
    
    return strain_front, strain_front_mean

def calculate_strain_domain(env_domain, angle):
    
    strain_domain = []
    strain_domain_mean = []
    for i in range(len(env_domain)):
        v_env = env_domain[i][1]
        delta = 1000 * (np.cos(np.radians(angle[i])))
    
        dv_envdy = np.gradient(v_env, delta, axis = 0)
        mean = np.mean(dv_envdy)
        
        strain_domain.append(dv_envdy)
        strain_domain_mean.append(mean)
    
    return strain_domain, strain_domain_mean

def theoretical_strain_front(u, v, ymin, ymax, xmin, xmax, lats, lons, angle):

    def plot_rectangle(xmin, xmax, ymin, ymax):
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xs, ys, color = 'black')
            
    u_front = []
    v_front = []
    lats_front = []
    lons_front = []
    front_theoretical_strain = []

    for i in range(len(v)):
        v_ref = v[i][5] * (np.cos(np.radians(angle[i])))
        u_ref = u[i][5] * (np.cos(np.radians(angle[i])))
        
        v_rot = rotate(v_ref, 
                       angle = angle[i], 
                       reshape = True)
        u_rot = rotate(u_ref, 
                       angle = angle[i], 
                       reshape = True)

        lats_rot = rotate(lats, 
                          angle = angle[i], 
                          reshape = True)
        lons_rot = rotate(lons, 
                          angle = angle[i], 
                          reshape = True)
        
        v_rot = v_rot[int(ymin[i]):int(ymax[i]), 
                      int(xmin[i]):int(xmax[i])]
        u_rot = u_rot[int(ymin[i]):int(ymax[i]), 
                      int(xmin[i]):int(xmax[i])]        

        dudy = np.gradient(u_rot, 1000, axis = 0)
        dvdx = np.gradient(v_rot, 1000, axis = 1)
    
        vortz = dvdx - dudy

        lats_rot = lats_rot[int(ymin[i]):int(ymax[i]), 
                            int(xmin[i]):int(xmax[i])]
        lons_rot = lons_rot[int(ymin[i]):int(ymax[i]), 
                            int(xmin[i]):int(xmax[i])] 
        
        u_front.append(u_rot)
        v_front.append(v_rot)
        lats_front.append(lats_rot)
        lons_front.append(lons_rot)
        
        plt.figure(figsize = (2, 9))
        levels_pos = [3e-3, 6e-3, 9e-3, 12e-3]
        plt.contour(vortz,
                    levels = levels_pos,
                    linewidths = 0.5,
                    colors = 'black')
        levels_neg = [-2e-3, -1.5e-3, -1e-3, -0.5e-3]
        plt.contour(vortz,
                    levels = levels_neg,
                    linewidths = 0.5,
                    colors = 'black')
        
        plt.savefig('/Users/tybuckingham/Desktop/vort_domain.png',
                    format = 'png',
                    dpi = 600,
                    bbox_inches = 'tight')    
        plt.show() 
        
        ##############
        
        vort_front = vortz[vortz > 0.003].mean()
        vort_neg = vortz[vortz < -0.001].mean()
        
        f = 1e-4
        wavenumber = np.e**((-2)*(2.5/17.5))
        strain_min = (f/4) * (vort_front + vort_neg)/(vort_front - vort_neg) * wavenumber
        
        # print(vortz.min(), ' | ', vortz.max())
        # print(strain_min)
        
        front_theoretical_strain.append(strain_min)
        
    return front_theoretical_strain

def theoretical_strain_focus(u, v, ymin, ymax, xmin, xmax, lats, lons, angle):

    def plot_rectangle(xmin, xmax, ymin, ymax):
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xs, ys, color = 'black')
            
    u_domain = []
    v_domain = []
    lats_domain = []
    lons_domain = []
    focus_theoretical_strain = []
        
    for i in range(len(v)):
        v_ref = v[i][5] * (np.cos(np.radians(angle[i])))
        u_ref = u[i][5] * (np.cos(np.radians(angle[i])))
        
        v_rot = rotate(v_ref, 
                       angle = angle[i], 
                       reshape = True)
        u_rot = rotate(u_ref, 
                       angle = angle[i], 
                       reshape = True)

        lats_rot = rotate(lats, 
                          angle = angle[i], 
                          reshape = True)
        lons_rot = rotate(lons, 
                          angle = angle[i], 
                          reshape = True)
        
        border = 140
        v_rot = v_rot[int(ymin[i] - border):int(ymax[i] + border), 
                      int(xmin[i] - border):int(xmax[i] + border)]
        u_rot = u_rot[int(ymin[i] - border):int(ymax[i] + border), 
                      int(xmin[i] - border):int(xmax[i] + border)]   
        
        dudy = np.gradient(u_rot, 1000, axis = 0)
        dvdx = np.gradient(v_rot, 1000, axis = 1)
    
        vortz = dvdx - dudy

        lats_rot = lats_rot[int(ymin[i] - border):int(ymax[i] + border), 
                            int(xmin[i] - border):int(xmax[i] + border)]
        lons_rot = lons_rot[int(ymin[i] - border):int(ymax[i] + border), 
                            int(xmin[i] - border):int(xmax[i] + border)]  
        
        u_domain.append(u_rot)
        v_domain.append(v_rot)
        lats_domain.append(lats_rot)
        lons_domain.append(lons_rot)

        plt.figure(figsize = (5, 9))
        levels_pos = [3e-3, 6e-3, 9e-3, 12e-3]
        plt.contour(vortz,
                    levels = levels_pos,
                    linewidths = 0.5,
                    colors = 'black')
        levels_neg = [-2e-3, -1.5e-3, -1e-3, -0.5e-3]
        plt.contour(vortz,
                    levels = levels_neg,
                    linewidths = 0.5,
                    colors = 'black')
        plot_rectangle(border, 
                       border + int(xmax[i] - xmin[i]), 
                       border, 
                       border + int(ymax[i] - ymin[i]))        
        
        plt.savefig('/Users/tybuckingham/Desktop/vort_domain.png',
                    format = 'png',
                    dpi = 600,
                    bbox_inches = 'tight')    
        plt.show() 
        
        ##############
        
        vort_front = vortz[vortz > 0.003].mean()
        vort_neg = vortz[vortz < -0.001].mean()
        
        f = 1e-4
        wavenumber = np.e**((-2)*(2.5/17.5))
        strain_min = (f/4) * (vort_front + vort_neg)/(vort_front - vort_neg) * wavenumber
        
        # print(vortz.min(), ' | ', vortz.max())
        # print(strain_min)
        
        focus_theoretical_strain.append(strain_min)

    return focus_theoretical_strain

# Plot the front-centric domain decomposition components.
def plot_hhd(wrf_var, hhd_var, lats, lons):
    
    cart_proj = get_cartopy(wrf_var)
    
    for i in range(len(hhd_var)):
        plt.subplots(1, 1, figsize = (10, 10), 
                     subplot_kw = {'projection': cart_proj})
    
        dens = 15
        plt.quiver(lons[i][::dens, ::dens],
                   lats[i][::dens, ::dens],
                   hhd_var[i][0][::dens, ::dens], 
                   hhd_var[i][1][::dens, ::dens], 
                   scale = 400, # 40 for div, 150 for rot/env.
                   width = 0.004,
                   transform = crs.PlateCarree())
        plt.show()

def plot_strain_mean(strain_domain_mean, strain_front_mean, strain_full_mean, front_theoretical_strain, focus_theoretical_strain):
    
    strain_domain_scaled = [domain * 100000 for domain in strain_domain_mean]
    strain_front_scaled = [front * 100000 for front in strain_front_mean]
    strain_full_scaled = [full * 100000 for full in strain_full_mean]
    focus_strain_scaled = [strain * 100000 for strain in focus_theoretical_strain]
    front_strain_scaled = [strain * 100000 for strain in front_theoretical_strain]
    
    plt.figure(figsize = (6, 6))
    plt.margins(0)
    
    ax = plt.gca()
    ax.set_ylim(-0.2, 2.2)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2])
    xlabels = ['8', '9', '10', '11', '12', '13', '14']
    ylabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2]
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    plt.xticks(rotation = 0)
    
    plt.xlabel('Time (UTC)', fontsize = 13, labelpad = 15)
    plt.ylabel('Stretching-deformation rate (s$^{-1}$ x 10$^{-5}$)', fontsize = 13)
    
    # plt.axhline(0.6, color = 'black', linestyle = 'dashed', linewidth = 0.9)
    # plt.axvspan(0, 11, facecolor = 'lightyellow', alpha = 0.5)
    # plt.axvspan(11, 21, facecolor = 'yellow', alpha = 0.5)
    # plt.axvspan(21, 34, facecolor = 'orange', alpha = 0.5)
    # plt.axvspan(34, 41, facecolor = 'orangered', alpha = 0.5)
    
    # plt.text(0.6, 1.64, '(a) Vortex sheet\n      roll up', fontsize = 9)
    # plt.text(11.4, 1.64, '(b) Subharmonic\n      interaction', fontsize = 9)
    # plt.text(21.4, 1.705, '(c) Consolidation', fontsize = 9)
    # plt.text(34.4, 1.705, '(d) Decay', fontsize = 9)

    plt.plot(strain_front_scaled, color = 'black', linewidth = 2.5, linestyle = (0,(3,1,1,1)), label = 'Inner frontal domain')    
    plt.plot(strain_domain_scaled, color = 'black', linewidth = 2.5, label = 'Outer fontal domain')
    plt.plot(front_strain_scaled, color = 'darkred', linewidth = 2.5, linestyle = (0,(3,1,1,1)), label = 'Inner frontal\ntheoretical minimum')
    plt.plot(focus_strain_scaled, color = 'darkred', linewidth = 2.5, label = 'Outer frontal\ntheoretical minimum')
    ax.legend(loc = 'lower right')
    
    
    plt.savefig('/Users/tybuckingham/Desktop/strain_timeseries_type1.png',
                format = 'png',
                dpi = 250,
                bbox_inches = 'tight')    
    
    plt.show()

def plot_domains(wrf_var, u, v, lats_full, lons_full, lats_domain, lons_domain, lats_front, lons_front):   
    # Indices for the snapshot.
    time = 14
    
    # Grid spacing in m.
    delta = 1000
    
    # Gradients.
    dudy = np.gradient(u[time][5], delta, axis = 0)
    dvdx = np.gradient(v[time][5], delta, axis = 1)

    # Vertical vorticity.
    vortz = dvdx - dudy
    
    # Domain boxes.
    lats_full_time = lats_full[time]
    lons_full_time = lons_full[time]
    lats_domain_time = lats_domain[time]
    lons_domain_time = lons_domain[time]
    lats_front_time = lats_front[time]
    lons_front_time = lons_front[time]
    
    lats_domain_corners = lats_domain_time[[0, 0, -1, -1], [0, -1, -1, 0]]
    lons_domain_corners = lons_domain_time[[0, 0, -1, -1], [0, -1, -1, 0]]
    lats_front_corners = lats_front_time[[0, 0, -1, -1], [0, -1, -1, 0]]
    lons_front_corners = lons_front_time[[0, 0, -1, -1], [0, -1, -1, 0]]
        
    domain_poly = geometry.Polygon([[lons_domain_corners[0], lats_domain_corners[0]],
                                    [lons_domain_corners[1], lats_domain_corners[1]],
                                    [lons_domain_corners[2], lats_domain_corners[2]],
                                    [lons_domain_corners[3], lats_domain_corners[3]]])   
    front_poly = geometry.Polygon([[lons_front_corners[0], lats_front_corners[0]],
                                   [lons_front_corners[1], lats_front_corners[1]],
                                   [lons_front_corners[2], lats_front_corners[2]],
                                   [lons_front_corners[3], lats_front_corners[3]]])      
    
    # Figure arguments.
    cart_proj = get_cartopy(wrf_var)
    plt.subplots(1, 1, figsize = (10, 10), 
                 subplot_kw = {'projection': cart_proj})

    mpl.rcParams['axes.linewidth'] = 4

    levels = [-12e-3, -9e-3, -6e-3, -3e-3, 3e-3, 6e-3, 9e-3, 12e-3]
    plt.contour(lons_full_time,
                lats_full_time,
                vortz, 
                levels = levels,
                colors = 'black',
                transform = crs.PlateCarree(), 
                zorder = 2)

    dens = 50
    plt.quiver(lons_full_time[::dens, ::dens],
               lats_full_time[::dens, ::dens],
               u[time][5][::dens, ::dens], 
               v[time][5][::dens, ::dens], 
               scale = 600, # 40 for div, 150 for rot/env.
               width = 0.003,
               transform = crs.PlateCarree())

    
    ax = plt.gca()
    ax.coastlines(resolution='10m', zorder = 1, color = 'grey')
    ax.add_geometries([domain_poly], 
                      crs = crs.PlateCarree(), 
                      edgecolor = 'blue',
                      facecolor = 'None',
                      linewidth = 4,
                      zorder = 3)
    
    ax.add_geometries([front_poly], 
                      crs = crs.PlateCarree(), 
                      edgecolor = 'purple',
                      facecolor = 'None',
                      linewidth = 4,
                      zorder = 3)    

    plt.savefig('/Users/tybuckingham/Desktop/domains.png',
                format = 'png',
                dpi = 250,
                bbox_inches = 'tight')     
    plt.show()

# Plot the front-centric domain decomposition components.
def plot_hhd_front(wrf_var, rot_front, div_front, env_front, lats_front, lons_front):
    
    time = 14
    
    lats_front_corners = lats_front[time][[0, 0, -1, -1], [0, -1, -1, 0]]
    lons_front_corners = lons_front[time][[0, 0, -1, -1], [0, -1, -1, 0]]

    front_poly = geometry.Polygon([[lons_front_corners[0], lats_front_corners[0]],
                                   [lons_front_corners[1], lats_front_corners[1]],
                                   [lons_front_corners[2], lats_front_corners[2]],
                                   [lons_front_corners[3], lats_front_corners[3]]]) 
    
    mpl.rcParams['axes.linewidth'] = 1
    cart_proj = get_cartopy(wrf_var)
    fig, ax = plt.subplots(1, 3, figsize = (10, 8), 
                           subplot_kw = {'projection': cart_proj})
    axs = ax.ravel()
    
    plt.subplots_adjust(wspace = 0.04)
    
    dens = 15
    axs[0].quiver(lons_front[time][::dens, ::dens],
                  lats_front[time][::dens, ::dens],
                  rot_front[time][0][::dens, ::dens], 
                  rot_front[time][1][::dens, ::dens], 
                  scale = 400, # 40 for div, 150 for rot/env.
                  width = 0.005,
                  zorder = 2,
                  transform = crs.PlateCarree())

    axs[0].text(lons_front[time][-1][0] - 1.8,
                lats_front[time][-1][0] - 0.2, 
                '(a)',
                fontsize = 22,
                fontweight = 'bold',
                transform = crs.PlateCarree())

    axs[1].quiver(lons_front[time][::dens, ::dens],
                  lats_front[time][::dens, ::dens],
                  div_front[time][0][::dens, ::dens], 
                  div_front[time][1][::dens, ::dens], 
                  scale = 100, # 40 for div, 150 for rot/env.
                  width = 0.005,
                  zorder = 2,
                  transform = crs.PlateCarree())

    axs[1].text(lons_front[time][-1][0] - 1.8,
                lats_front[time][-1][0] - 0.2, 
                '(b)',
                fontsize = 22,
                fontweight = 'bold',
                transform = crs.PlateCarree())
    
    axs[2].quiver(lons_front[time][::dens, ::dens],
                  lats_front[time][::dens, ::dens],
                  env_front[time][0][::dens, ::dens], 
                  env_front[time][1][::dens, ::dens], 
                  scale = 300, # 40 for div, 150 for rot/env.
                  width = 0.005,
                  zorder = 2,
                  transform = crs.PlateCarree())

    axs[2].text(lons_front[time][-1][0] - 1.8,
                lats_front[time][-1][0] - 0.2, 
                '(c)',
                fontsize = 22,
                fontweight = 'bold',
                transform = crs.PlateCarree())

    for ax in axs:
        ax.add_geometries([front_poly], 
                          crs = crs.PlateCarree(), 
                          edgecolor = 'purple',
                          facecolor = 'None',
                          linewidth = 2,
                          zorder = 1)  

    plt.savefig('/Users/tybuckingham/Desktop/hhd_trio.png',
                format = 'png',
                dpi = 250,
                bbox_inches = 'tight')     
    plt.show()

u_domain, v_domain, lats_domain, lons_domain = focus_domain(u, v, ymin, ymax, xmin, xmax, lats, lons, angle)
u_front, v_front, lats_front, lons_front = frontal_domain(u, v, ymin, ymax, xmin, xmax, lats, lons, angle)

lons_full = [lons] * len(wrf_var)
lats_full = [lats] * len(wrf_var)
div_full, rot_full, env_full = hhd_full(u, v)
div_domain, rot_domain, env_domain = hhd_domain(u_domain, v_domain, angle)
div_front, rot_front = hhd_front(u_front, v_front, angle) # Disregard the harmonic.
env_front, flow_nofront = subtract_front(u, v, env_domain, angle)

strain_front, strain_front_mean = calculate_strain_front(env_front, angle)
strain_domain, strain_domain_mean = calculate_strain_domain(env_domain, angle)
strain_full, strain_full_mean = calculate_strain_full(env_full)
front_theoretical_strain = theoretical_strain_front(u, v, ymin, ymax, xmin, xmax, lats, lons, angle)
focus_theoretical_strain = theoretical_strain_focus(u, v, ymin, ymax, xmin, xmax, lats, lons, angle)

plot_hhd(wrf_var, env_front, lats_front, lons_front)
plot_hhd(wrf_var, env_domain, lats_domain, lons_domain)
plot_hhd(wrf_var, div_full, lats_full, lons_full)
plot_strain_mean(strain_domain_mean, strain_front_mean, strain_full_mean, front_theoretical_strain, focus_theoretical_strain)
plot_domains(wrf_var, u, v, lats_full, lons_full, lats_domain, lons_domain, lats_front, lons_front)
plot_hhd_front(wrf_var, rot_front, div_front, env_front, lats_front, lons_front)

# for domain in strain_full:
#     plt.figure(figsize = (4, 8))
#     plt.contourf(domain)











def calc_vorticity(u, v, lons, lats):
    
    time = 8
    
    # Grid spacing in m.
    delta = 1000
    
    # Gradients.
    dudy = np.gradient(u[time][5], delta, axis = 0)
    dvdx = np.gradient(v[time][5], delta, axis = 1)
    
    vortz = dvdx - dudy

    cart_proj = get_cartopy(wrf_var)
    plt.subplots(1, 1, figsize = (10, 10), 
                 subplot_kw = {'projection': cart_proj})
    
    # levels = [-12e-3, -9e-3, -6e-3, -3e-3, 3e-3, 6e-3, 9e-3, 12e-3]
    levels_pos = [3e-3, 6e-3, 9e-3, 12e-3]
    plt.contour(lons,
                lats,
                vortz, 
                levels = levels_pos,
                colors = 'black',
                transform = crs.PlateCarree(), 
                linewidths = 0.1,
                zorder = 2)

    levels_neg = [-4e-3, -3e-3, -2e-3, -1e-3]
    plt.contour(lons,
                lats,
                vortz, 
                levels = levels_neg,
                colors = 'black',
                transform = crs.PlateCarree(), 
                linewidths = 0.1,
                linestyle = 'dashed',
                zorder = 2)   
    
    plt.savefig('/Users/tybuckingham/Desktop/vort_test.png',
                format = 'png',
                dpi = 600,
                bbox_inches = 'tight')     
    plt.show()