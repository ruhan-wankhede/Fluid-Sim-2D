import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 0.1
diff = 0.00001
N = 64
CELL_SIZE = 8
WIDTH = N*CELL_SIZE


velx = np.ones((N+2, N+2), dtype=np.float64)
velx_prev = np.zeros((N+2, N+2), dtype=np.float64)
vely = np.ones((N+2, N+2), dtype=np.float64)
vely_prev = np.zeros((N+2, N+2), dtype=np.float64)

density = np.zeros((N+2, N+2), dtype=np.float64)
density_prev = np.zeros((N+2, N+2), dtype=np.float64)

sources = np.zeros((N+2, N+2), dtype=np.float64)
for i in range(1, 3):
    for j in range(1, 3):
        sources[i, j] = 1


fig, ax = plt.subplots()
img = ax.imshow(density, cmap='inferno', origin='lower', vmin=0, vmax=1)
quiver = ax.quiver(velx, vely, scale=50, color='white')

"""sources[N//2, N//2] = 1
sources[N//2, N//2+1] = 1
sources[N//2+1, N//2] = 1
sources[N//2+1, N//2+1] = 1"""

def set_bnd(b, x):
    x[0,:] = x[1,:] if b != 1 else -x[1,:]
    x[-1,:] = x[-2,:] if b != 1 else -x[-2,:]
    x[:,0] = x[:,1] if b != 2 else -x[:,1]
    x[:,-1] = x[:,-2] if b != 2 else -x[:,-2]

    x[0,0] = 0.5 * (x[1,0] + x[0,1])
    x[0,-1] = 0.5 * (x[1,-1] + x[0,-2])
    x[-1,0] = 0.5 * (x[-2,0] + x[-1,1])
    x[-1,-1] = 0.5 * (x[-2,-1] + x[-1,-2])

    return x

def diffuse() -> None:
    global density
    # Gauss-Seidel relaxation to approximate next density value as average of its next four cells
    a: float = dt * diff * N * N
    for k in range(20):
        for i in range(N):
            for j in range(N):
                density[i, j] = density_prev[i, j] + a * (density[i-1, j] + density[i, j-1] + density[i+1, j] + density[i, j+1]) / (1 + 4 * a)

    density = set_bnd(0, density)


def advect() -> None:
    global density
    # Advection calculation
    dt0 = dt * N
    for i in range(N):
        for j in range(N):
            x = i - dt0*velx[i, j]
            y = j - dt0*vely[i, j]

            x = np.clip(x, 0.5, N + 0.5)
            y = np.clip(y, 0.5, N + 0.5)

            i0 = int(x)
            i1 = i0 + 1
            j0 = int(y)
            j1 = j0 + 1

            s1 = x - i0
            s0 = 1 - s1
            t1 = y - j0
            t0 = 1 - t1

            density[i, j] = s0 * (t0 * density_prev[i0, j0] + t1 * density_prev[i0, j1]) + s1 * (t0 * density_prev[i1, j0] + t1 * density_prev[i1, j1])

    density = set_bnd(0, density)


def add_source() -> None:
    for i in range(N):
        for j in range(N):
            density[i, j] += dt * sources[i, j]


def step() -> None:
    global density_prev
    add_source()
    density_prev = density.copy()
    diffuse()
    density_prev = density.copy()
    advect()


def simulate(frame: int):
    step()
    img.set_data(density)
    return [img]

if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, simulate, frames=144, interval=50, blit=True)
    plt.show()


