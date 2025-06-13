import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 0.1
diff = 0.001
N = 64
CELL_SIZE = 8
WIDTH = N*CELL_SIZE

def create_wave_field(N):
    velx = np.zeros((N+2, N+2), dtype=np.float32)
    vely = np.zeros((N+2, N+2), dtype=np.float32)

    for i in range(1, N+1):
        for j in range(1, N+1):
            velx[i, j] = np.sin(j / N * 2 * np.pi)
            vely[i, j] = np.cos(i / N * 2 * np.pi)

    return velx, vely

velx, vely = create_wave_field(N)
velx_prev = velx.copy()
vely_prev = vely.copy()

"""velx = np.zeros((N+2, N+2), dtype=np.float32)
velx_prev = np.zeros((N+2, N+2), dtype=np.float32)
vely = np.zeros((N+2, N+2), dtype=np.float32)
vely_prev = np.zeros((N+2, N+2), dtype=np.float32)"""

density = np.zeros((N+2, N+2), dtype=np.float32)
density_prev = np.zeros((N+2, N+2), dtype=np.float32)

sources = np.zeros((N+2, N+2), dtype=np.float32)
for i in range(30, 35):
    for j in range(30, 35):
        sources[i, j] = 1

density_prev = sources.copy()

fig, ax = plt.subplots()
img = ax.imshow(density, cmap='inferno', origin='lower', vmin=0, vmax=1)
X, Y = np.meshgrid(np.arange(1, N+1), np.arange(1, N+1))
quiver = ax.quiver(X, Y, velx[1:-1, 1:-1], vely[1:-1, 1:-1], scale=50, color='white')


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

def diffuse(b: int, x: np.ndarray, x0: np.ndarray) -> np.ndarray:
    # Gauss-Seidel relaxation to approximate next density value as average of its next four cells
    a: float = dt * diff * N * N
    for k in range(20):
        for i in range(1, N):
            for j in range(1, N):
                x[i, j] = x0[i, j] + a * (x[i-1, j] + x[i, j-1] + x[i+1, j] + x[i, j+1]) / (1 + 4 * a)

    x = set_bnd(b, x)
    return x


def advect(b: int, dens: np.ndarray, dens_prev: np.ndarray) -> np.ndarray:
    # Advection calculation
    dt0 = dt * N
    for i in range(1, N):
        for j in range(1, N):
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

            dens[i, j] = s0 * (t0 * dens_prev[i0, j0] + t1 * dens_prev[i0, j1]) + s1 * (t0 * dens_prev[i1, j0] + t1 * dens_prev[i1, j1])

    dens = set_bnd(b, dens)
    return dens

def project(u: np.ndarray, v: np.ndarray, p: np.ndarray, div: np.ndarray) -> tuple[np.ndarray]:
    h = 1/N
    for i in range(1, N):
        for j in range(1, N):
            div[i, j] = -0.5 * h * (u[i+1, j] - u[i-1, j] + v[i, j+1] - v[i, j-1])
            p[i, j] = 0

    div = set_bnd(0, div)
    p = set_bnd(0, p)

    # Gauss-Seidel relaxation
    for k in range(20):
        for i in range(1, N):
            for j in range(1, N):
                p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1]) / 4

        p = set_bnd(0, p)

    for i in range(1, N):
        for j in range(1, N):
            u[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j]) / h
            v[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1]) / h

    u = set_bnd(1, u)
    v = set_bnd(2, v)

    return u, v, p, div


def add_source(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    for i in range(1, N):
        for j in range(1, N):
            x[i, j] += dt * s[i, j]
    return x



def step() -> None:
    global density, density_prev, velx, vely, velx_prev, vely_prev

    density = add_source(density, density_prev)

    density_prev = density.copy()
    density = diffuse(0, density, density_prev)

    density_prev = density.copy()
    density = advect(0, density, density_prev)

    density = set_bnd(0, density)

    velx = add_source(velx, velx_prev)
    vely = add_source(vely, vely_prev)

    velx_prev = velx.copy()
    velx = diffuse(1, velx, velx_prev)
    vely_prev = vely.copy()
    vely = diffuse(2, vely, vely_prev)

    velx, vely, velx_prev, vely_prev = project(velx, vely, velx_prev, vely_prev)

    velx_prev = velx.copy()
    vely_prev = vely.copy()

    velx = advect(1, velx, velx_prev)
    vely = advect(2, vely, vely_prev)
    velx, vely, velx_prev, vely_prev = project(velx, vely, velx_prev, vely_prev)

    velx *= 0.8
    vely *= 0.8




def simulate(frame: int):
    step()
    img.set_data(density)
    quiver.set_UVC(velx[1:-1, 1:-1], vely[1:-1, 1:-1])
    return [img, quiver]

if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, simulate, frames=200, interval=50, blit=True)
    plt.show()


