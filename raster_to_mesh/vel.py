import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import linregress
# takes raster thickness data, loads it, and plots the first time step
# The first dimension of H is a time variable, and the next 2
# dimensions are x, y spatial variables
def load_data():
    data = xr.load_dataset('/home/katie/Documents/graph_neural/grate_stuff/graph_neural/raster/0.nc')
    H = data.Thickness.data
    S = data.Surface.data
    S_index= data.bed.data + H
    # xl=data.x.data
    # yl = data.y.data
    accumulation = data.SmbMassBalance.data
    # Below can be uncommented to see plots of data
    
    # plot first time step
    # plt.imshow(accumulation[0])
    # plt.show()
    
    return data, H, S, accumulation


def print_vars(data):
    for var_name, var in data.variables.items():
        print(f"Variable: {var_name}")
        
        # cdata = var.data
        # print(cdata.shape)
        # if cdata.shape ==  (161,178,206):
        #      plt.imshow(cdata[0])
        #      plt.title(f"{var_name}")
        #      plt.show()



# Sort elevations
# Remove duplicates
def sort_elev(S_index, H):
    # Set elevation to infinity everywhere that the glacier doesn't exist
    snew = np.where(H == 0, np.inf, S_index)

    # Mask this new elevation, sort it from high to low. 
    mask = (snew != np.inf)
    # Take out duplicate values- those are dealt with in projection2
    s_small = np.array(list(set(snew[mask])))
    e_sorted = -np.sort(-s_small)

    # Uncomment this if nervous that elevation isn't sorted ¯\_(ツ)_/¯
    # if check_set(e_sorted) == False:
    #     raise ValueError("Not sorted")
    return e_sorted, snew

def check_set(elev):
    #sorted_elev_set= list(sorted_elev_set)
    for i in range(1,len(elev)):
        if elev[i] >= elev[i-1]:
            print(i)
            return False
    return True


# Scalar flux out of a grid cell in x and y directions given flow direction theta and accumulation
def flux(theta, accumulation):

    flux_x = accumulation*np.abs(np.cos(theta))/(np.abs(np.sin(theta)) + np.abs(np.cos(theta)))
 
    flux_y = accumulation*np.abs(np.sin(theta))/(np.abs(np.sin(theta)) + np.abs(np.cos(theta)))

    return flux_x, flux_y


# Computes flow in x and y directions and gives a theta for flow direction
def flow_2(curr_highest, s_copy):
    x_diff_left  = s_copy[curr_highest] - s_copy[curr_highest[0], curr_highest[1] -1 ]
    x_diff_right = s_copy[curr_highest] - s_copy[curr_highest[0], curr_highest[1] + 1]

    y_diff_down = s_copy[curr_highest] - s_copy[curr_highest[0]+1, curr_highest[1]]
    y_diff_up = s_copy[curr_highest] - s_copy[curr_highest[0]-1, curr_highest[1]]
    
    # NOT MOVING IN X DIRECTION
    if x_diff_right < 0 and x_diff_left < 0:
       x_comp = 0

   # MOVING LEFT
    elif (x_diff_left) > (x_diff_right):
       x_comp = -1* x_diff_left
       
    # MOVING RIGHT
    else:
        x_comp = x_diff_right

    # NOT MOVING IN Y DIRECTION
    if y_diff_down < 0 and y_diff_up < 0:
        y_comp = 0

    # MOVING DOWN    
    elif y_diff_down > y_diff_up:
        y_comp = -y_diff_down

    # MOVING UP
    else:
        y_comp = y_diff_up

    theta = np.arctan2(y_comp,x_comp)

   
    return theta, x_comp, y_comp
    
# Project change in accumulation (adot) from higher elevations onto lower elevations at a specific
# time (Right now, all projection takes place in 1 time frame)
def projection2(S_index, adot, H):
    e_sorted, snew = sort_elev(S_index, H)

    # Adding a frame around the elevation matrix- otherwise will have boundary  problems
    # This is realy only necessary if the glacier fills up the entire grid (ie, doing part of a glacier)
    # s_copy = np.pad(S_index, (1,1), "constant", constant_values=(np.inf,np.inf))
    #return_flow = np.pad(adot, (1,1), "constant", constant_values =(0,0))
    # orig_accum = np.pad(adot, (1,1), "constant", constant_values =(0,0))


    s_copy = snew

    return_flow = np.where(H == 0, np.inf, adot)
    inflow = np.zeros_like(return_flow)
    outflow = np.zeros_like(return_flow)
  
    for e in e_sorted:
        # print("Index, elevation:")
        # print(np.where(e_sorted == e), e)
        curr_highest = np.where(s_copy == e)
        c = (curr_highest[1], curr_highest[0])
        # Below is for if you pad the matrix
        # curr_highest = tuple(np.array([ch[0] + 1, ch[1] + 1]))

        # print("curr highest", curr_highest, 'length', len(curr_highest))
        
        # theta, x_comp, y_comp = flow_2(curr_highest, s_copy)
        pairs = []
        #if len(curr_highest[0]) > 1:
        # Need to account for elevations that might be the same
        for i in range(len(curr_highest[0])):
            pairs.append((np.array(curr_highest[0][i]),np.array(curr_highest[1][i])))
        # print("Pairs! Length of pairs", len(pairs))
        # print(pairs)
        # else:
        #     pairs.append(np.array())
        #theta, x_comp, y_comp = flow_2(curr_highest, s_copy)
        # print("PAIRS", pairs)
        
        for pair in pairs:
            
            theta, x_comp, y_comp = flow_2(pair, s_copy)
            flux_x, flux_y = np.abs(flux(theta, return_flow[pair]))

        
            outflow[pair] += flux_y
            outflow[pair] += flux_x

            if np.round(theta,5) == np.round(-np.pi/2,5):
                # Only down
                return_flow[pair[0] + 1, pair[1]] += flux_y
                inflow[pair[0] + 1, pair[1]] += flux_y
                
            elif np.round(theta,5) == np.round(np.pi/2,5):
                # Only up 
                return_flow[pair[0] - 1, pair[1]] += flux_y
                inflow[pair[0] - 1, pair[1]] += flux_y

            elif theta == 0.0 and x_comp == 0 and y_comp == 0:
                # print("Here")
                pass

            elif theta >=0 and theta < np.pi/2:
                # Quad 1, x right, y up
                # print("Q1, x right, y up")
                
                # print(return_flow[curr_highest[0], curr_highest[1] + 1])
                
                return_flow[pair[0], pair[1] + 1] += flux_x
                
                inflow[pair[0], pair[1] + 1] += flux_x

                return_flow[pair[0] - 1, pair[1]] += flux_y

                inflow[pair[0] - 1, pair[1]] += flux_y

            elif theta >= np.pi/2 and np.round(theta,5) <= np.round(np.pi,5):
                #Quad 2, x left, y up
                # print("q2, x left, y up")
                return_flow[pair[0], pair[1] - 1] += flux_x
                return_flow[pair[0] - 1, pair[1]] += flux_y

                inflow[pair[0], pair[1] - 1] += flux_x
                inflow[pair[0] - 1, pair[1]] += flux_y

            elif theta >= -np.pi and theta <= -np.pi/2:
                #  Quad 3, x left, y down
                # print("q3, x left, y down")
                return_flow[pair[0], pair[1] - 1] += flux_x
                return_flow[pair[0] + 1, pair[1]] += flux_y
                
                inflow[pair[0], pair[1] - 1] += flux_x
                inflow[pair[0] + 1, pair[1]] += flux_y        

            elif theta < 0 and theta >= -np.pi/2:
                # quad 4, x right, y down
                # print("q4, x right, y down")
                return_flow[pair[0], pair[1] + 1] += flux_x
                return_flow[pair[0] + 1, pair[1]] += flux_y

                inflow[pair[0], pair[1] + 1] += flux_x
                inflow[pair[0] + 1, pair[1]] += flux_y

            else:
                print("PROBLEM")
                print("Elevation", S_index[curr_highest])
                print("theta", theta)
      
        # plot_changes(s_copy,return_flow, adot,rf1,c)

        # return_flow starts with accumulation, only gets added to (Total influx)
        # inflow starts with 0s, gets added to
        # Outflow starts with accumulation, gets added to from

    change = return_flow - outflow
    vel = return_flow/H
      
    return return_flow, vel



# For use with projection2
def plot_changes(s_copy,return_flow, orig_accum,rf1,c):
    fig, axs = plt.subplots(1,5, figsize=(12,4))
    axs[0].imshow(return_flow)
    axs[0].set_title("Changed Flow, Total")
    axs[0].scatter(*c,  color='blue')

    axs[1].imshow(orig_accum)
    axs[1].set_title("Original Accumulation")
    axs[1].scatter(*c,  color='blue')

    axs[2].imshow(return_flow - orig_accum)
    axs[2].set_title("Difference")
    axs[2].scatter(*c,  color='blue')

    axs[3].set_title("Changed Flow, this time step")
    axs[3].imshow(return_flow - rf1)
    axs[3].scatter(*c,  color='blue')

    axs[4].imshow(s_copy)
    axs[4].set_title("Elevation")
    axs[4].scatter(*c,  color='blue')
    plt.tight_layout
    plt.show()
    

#Visualize outputs
def plot_final(s_copy, return_flow, orig_adot, inflow, outflow, H, snew, vel):
    fig, axs = plt.subplots(3,3, figsize=(12,4))
    # orig_adot = np.pad(orig_adot, (1,1), "constant", constant_values =(0,0))
    masked_a = np.where(H == 0, np.inf, orig_adot)
    axs[0][0].imshow(orig_adot)
    axs[0][0].set_title("Original Accumulation")

    axs[0][1].imshow(s_copy)
    axs[0][1].set_title("Total Elevation")

    axs[0][2].imshow(H)
    axs[0][2].set_title("Thickness")

    axs[1][1].imshow(snew)
    axs[1][1].set_title("Masked Elevation")

    axs[1][0].imshow(masked_a)
    axs[1][0].set_title("Masked Accumulation")

    axs[1][2].imshow(vel)
    axs[1][2].set_title("Velocity")
    
    axs[2][0].imshow(return_flow)
    axs[2][0].set_title("Calculated Flow")

    axs[2][1].imshow(inflow)
    axs[2][1].set_title("Inflow")

    axs[2][2].imshow(outflow)
    axs[2][2].set_title("Outflow")


    plt.tight_layout
    
    plt.show()

    
def plotter(rf, snew, H):
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(rf)
    axs[1].imshow(snew)
    axs[2].imshow(H)

    ch = np.where(rf == rf.min())

    highlight_color = 'red'  # Color for highlighting (change as desired)
    highlight_marker = 'o'  # Marker shape for highlighting (change as desired)
    highlight_size = 10  # Size of the marker (change as desired)
    axs[0].scatter(*ch, marker=highlight_marker, color=highlight_color, s=highlight_size)
    axs[1].scatter(*ch, marker=highlight_marker, color=highlight_color, s=highlight_size)
    axs[2].scatter(*ch, marker=highlight_marker, color=highlight_color, s=highlight_size)

    ch2 = (ch[1], ch[0])
    axs[0].scatter(*ch2, marker=highlight_marker, color='blue', s=highlight_size)
    axs[1].scatter(*ch2, marker=highlight_marker, color='blue', s=highlight_size)
    axs[2].scatter(*ch2, marker=highlight_marker, color='blue', s=highlight_size)


    plt.show()


# This is Jake's script to generate random noise
def gen_field(nx, ny, correlation_scale):

    # Create the smoothing kernel
    x = np.arange(-correlation_scale, correlation_scale)
    y = np.arange(-correlation_scale, correlation_scale)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X*X + Y*Y)
    filter_kernel = np.exp(-dist**2/(2*correlation_scale))

    # Generate random noise and smooth it
    noise = np.random.randn(nx, ny) 
    z = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
        
    # Normalize so its in 0-1 range
    z -= z.min()
    z /= z.max()

    return z


# beta is basal friction coefficient = alpha_0
def beta(thickness, elev, balance_v ,density=911, gravity=9.8) :
    
    gradient_x, gradient_y = np.gradient(elev)
    gradient_norm = np.linalg.norm([gradient_x, gradient_y], axis=0)
   
    beta_squared = (density*gravity*thickness*gradient_norm)/(balance_v + .1)
    
    # Some funky stuff happens with beta with all the infinities- mask it to 0 for plotting
    # mask = np.isclose(beta_squared, 0.0, atol=1e-10)
    # bs = np.ma.masked_array(beta_squared, mask)

    # fig, axs = plt.subplots(1,3)
    # axs[0].imshow(beta_squared)
    # axs[0].set_title("Beta squared")
   

    # axs[1].imshow(bs)
    # axs[1].set_title("Masked")

    # axs[2].imshow(balance_v)
    # axs[2].set_title("Velocity")
    
    # plt.show()
    
    return  beta_squared


# New final equation
def traction(beta_squared, gamma = .7):
    # Noise is already normalized
    # Eqn: b_squared = b_squared + noise/noise_max * |b_squared| * gamma
    noise = gen_field(len(beta_squared), len(beta_squared[1]), 100)

    # Want to find the average of beta_squared, but don't want to include any of the zeros
    zero_mask = np.ma.masked_equal(beta_squared, 0) 
    beta_avg = np.abs(np.mean(zero_mask))

    traction_squared = beta_squared + noise*beta_avg*gamma

    # Not sure if it should be 0 everywhere the glacier doesn't exist
    tnew = np.where(beta_squared == 0, 0, traction_squared)

   
    return traction_squared, tnew


# Runs it all- gives out results of all previous functions
def finale():
    data, H, S, accumulation = load_data()
    return_flow, vel = projection2(S[0], accumulation[0], H[0])
    beta_squared = beta(H[0], S[0], vel)
    
    traction_squared, tnew = traction(beta_squared)
    finale_plots(accumulation[0],vel,beta_squared,tnew,H[0])
    return return_flow, vel, beta_squared, traction_squared, tnew

def compute_vel_from_mesh(s, adot,h):
    rf, vel = projection2(s,adot,h)
    beta_s = beta(h,s,vel)
    ts, tnew = traction(beta_s)
    return vel, tnew
    

def finale_plots(orig_acc,vel, beta_squared,tnew,H):
    fig, axs = plt.subplots(2,2)
    masked_a = np.where(H == 0, np.inf, orig_acc)
    axs[0][0].imshow(masked_a, origin='lower')
    axs[0][0].set_title("Original Accumulation, Masked")

    axs[0][1].imshow(vel,origin='lower')
    axs[0][1].set_title("Calculated Velocity")

    masked_beta = np.where(beta_squared == 0, np.inf, beta_squared)
    axs[1][0].imshow(masked_beta, origin = 'lower')
    axs[1][0].set_title('Beta Squared')

    masked_t = np.where(tnew == 0,np.inf, tnew)
    axs[1][1].imshow(np.sqrt(masked_t), origin = 'lower')
    axs[1][1].set_title('Basal Traction')

    fig.suptitle("Output for Glacier 0")
    plt.show()

def fit_smb(so,adot):
    adot_flat = adot.ravel()
    so_flat = so.ravel()

    slope, intercept, _, _, _ = linregress(adot_flat, so_flat)

    x_fit = np.linspace(min(adot_flat), max(adot_flat), 100)
    y_fit = slope * x_fit + intercept

    plt.scatter(adot_flat, so_flat)
    plt.plot(x_fit, y_fit, color='red', label='Fitted Line')

    plt.xlabel("Surface Mass Balance")
    plt.ylabel("Elevation")
    plt.title("Elevation vs SMB with Line of Best Fit")
    plt.show()

    return slope, intercept

def est_adot(slope,intercept,elev_raster):
    SMB_estimates = slope * elev_raster + intercept
    return SMB_estimates

def main():
    
    data, H, S, xl, yl, accumulation = load_data()
 
def practice(vel):
    neg = vel < 0
    y,x  = np.where(neg)
    plt.imshow(vel)
    plt.scatter(x,y,marker='x',color='red')
    plt.colorbar()
    plt.show()
if __name__ == '__main__':
   main()