import numpy as np
import matplotlib.pyplot as plt


### Function for legend
## https://stackoverflow.com/questions/31967472/smooth-interpolated-tertiary-or-quaternary-colour-scales-e-g-r-g-b-triangle
def abc_to_rgb(A=0.0,B=0.0,C=0.0, sat=1):
    ''' Map values A, B, C (all in domain [0,1]) to
    suitable red, green, blue values.'''
    #return (min(B+C,1.0),min(A+C,1.0),min(A+B,1.0)) #CMY
    maxr = max(abs(A),abs(B),abs(C))/sat#
    return (A/maxr,B/maxr,C/maxr)#RGB

def plot_legend(ax, sat):
    ''' Plots a legend for the colour scheme
    given by abc_to_rgb. Includes some code adapted
    from http://stackoverflow.com/a/6076050/637562'''

    # Basis vectors for triangle
    basis = np.array([[0.0, 1.0], [-1.5/np.sqrt(3), -0.5],[1.5/np.sqrt(3), -0.5]])
    


    # Plot points
    a, b, c = np.mgrid[0.0:1:50j, 0.0:1:50j, 0.0:1:50j]
    a, b, c = a.flatten(), b.flatten(), c.flatten()

    abc = np.dstack((a,b,c))[0]
    #abc = filter(lambda x: x[0]+x[1]+x[2]==1, abc) # remove points outside triangle
    abc = np.array(list(map(lambda x: x/sum(x), abc))) # or just make sure points lie inside triangle ...

    data = np.dot(abc, basis)
    colours = [abc_to_rgb(A=point[0],B=point[1],C=point[2], sat=sat) for point in abc]

    ax.scatter(data[:,0], data[:,1],marker=',',edgecolors='none',facecolors=colours)

    # Plot triangle
    #ax.plot([basis[_,0] for _ in range(3) + [0,]],[basis[_,1] for _ in range(3) + [0,]],**{'color':'black','linewidth':3})

    # Plot labels at vertices
    offset = 0.4
    fontsize = 12
    ax.text(basis[0,0]*(1+offset), basis[0,1]*(1+offset), '$U_1$', horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)
    ax.text(basis[1,0]*(1+offset), basis[1,1]*(1+offset), '$U_2$', horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)
    ax.text(basis[2,0]*(1+offset), basis[2,1]*(1+offset), '$U_3$', horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)    

    ax.set_frame_on(False)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect('equal')

    return(ax)

## list of data point in 2D space with function as R^2 --> R^3

points = np.loadtxt('omega_k_PSV.csv', delimiter=",", dtype=complex)

### get 3 values by point

u = np.loadtxt('u.txt')
u = u/np.max(u, axis=1)[:, None]
mmaxu = np.max(u)



## Scatter plot

def get_rgbcolor(u, color_saturation = 1):
    """
    u : array N, 3 dtype real
    colorsaturation : float [0-1] 

    return list of rgb
    """
    norm = np.max(u)/(color_saturation)
    return [(u1/norm, u2/norm, u3/norm) for u1, u2, u3 in zip(u[:,0],u[:,1],u[:,2])] ## RGB
    #return [(min((B+C)/norm,1.0),min((A+C)/norm,1.0),min((A+B)/norm,1.0)) for A, B, C in zip(u[:,0],u[:,1],u[:,2])] ### CMY


om = points[:, 0]
k =  points[:, 1]

plt.figure()
ax = plt.subplot(111)

idxR = np.abs(np.imag(k))< 1e-5 # get only real wavenumber

u = u[idxR,:]
ax.scatter(np.real(k[idxR]), np.real(om[idxR]), s = 1, c=get_rgbcolor(u, 0.8))

ax.set_xlabel("k")
ax.set_xlim(0, 15)
ax.set_ylabel(r"$\omega$")
axins = ax.inset_axes([0.7, 0.7, 0.2, 0.2])
plot_legend(axins, 0.8)
plt.savefig('ex.png', dpi = 800)
plt.show()