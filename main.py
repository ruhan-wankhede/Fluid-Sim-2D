import numpy as np
import pygame
import matplotlib

colormap = matplotlib.colormaps["inferno"]

pygame.init()


dt = 0.1
diff = 0.001
N = 32
CELL_SIZE = 8
WIDTH = N*CELL_SIZE


velx = np.zeros((N+2, N+2), dtype=np.float32)
velx_prev = np.zeros((N+2, N+2), dtype=np.float32)
vely = np.zeros((N+2, N+2), dtype=np.float32)
vely_prev = np.zeros((N+2, N+2), dtype=np.float32)

density = np.zeros((N+2, N+2), dtype=np.float32)
density_prev = np.zeros((N+2, N+2), dtype=np.float32)


def set_bnd(b, x):
    n = x.shape[0] - 2

    # Left/right edges
    for i in range(1, n + 1):
        x[0, i]     = -x[1, i]     if b == 1 else x[1, i]     # left
        x[n + 1, i]   = -x[n, i]     if b == 1 else x[n, i]     # right

        x[i, 0]     = -x[i, 1]     if b == 2 else x[i, 1]     # bottom
        x[i, n + 1]   = -x[i, n]     if b == 2 else x[i, n]     # top

    # Corners: average of adjacent edges
    x[0, 0]       = 0.5 * (x[1, 0] + x[0, 1])
    x[0, n + 1]     = 0.5 * (x[1, n + 1] + x[0, n])
    x[n + 1, 0]     = 0.5 * (x[n, 0] + x[n + 1, 1])
    x[n + 1, n + 1]   = 0.5 * (x[n, n + 1] + x[n + 1, n])

    return x

def diffuse(b: int, x: np.ndarray, x0: np.ndarray) -> np.ndarray:
    # Gauss-Seidel relaxation to approximate next density value as average of its next four cells
    a: float = dt * diff * N * N
    for k in range(20):
        for i in range(1, N):
            for j in range(1, N):
                x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i, j-1] + x[i+1, j] + x[i, j+1])) / (1 + 4 * a)

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


def add_density(dens: np.ndarray) -> np.ndarray:
    pos = pygame.mouse.get_pos()
    i = pos[0] // CELL_SIZE
    j = pos[1] // CELL_SIZE
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if 1 <= i + di < N + 1 and 1 <= j + dj < N + 1:
                dens[i + di, j + dj] += 2  # stronger injection
    return set_bnd(0, dens)


def add_velocity(vel_x: np.ndarray, vel_y: np.ndarray, vx: float, vy: float) -> np.ndarray:
    pos = pygame.mouse.get_pos()
    vel_x[pos[0] // CELL_SIZE, pos[1] // CELL_SIZE] += vx * 0.5
    vel_y[pos[0] // CELL_SIZE, pos[1] // CELL_SIZE] += vy * 0.5
    vel_x = set_bnd(1, vel_x)
    vel_y = set_bnd(2, vel_y)

    return vel_x, vel_y


def step() -> None:
    global density, density_prev, velx, vely, velx_prev, vely_prev

    density_prev = density.copy()
    density = diffuse(0, density, density_prev)

    density_prev = density.copy()
    density = advect(0, density, density_prev)

    density = set_bnd(0, density)
    density *= 0.99

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


def draw(win: pygame.Surface) -> None:
    for i in range(1,N):
        for j in range(1, N):
            x = CELL_SIZE * i
            y = CELL_SIZE * j
            val = np.clip(density[i, j], 0.0, 1.0)  # clamp between 0–1
            r, g, b, _ = colormap(val)
            color = (int(r * 255), int(g * 255), int(b * 255))
            pygame.draw.rect(win, color, (x-4, y-4, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()



def simulate():
    global density, velx, vely
    run = True

    WIN = pygame.display.set_mode((WIDTH, WIDTH))
    pygame.display.set_caption('Fluid Simulation')

    clock = pygame.time.Clock()
    prev_mouse_pos = pygame.mouse.get_pos()
    mouse_pos = prev_mouse_pos
    while run:
        delt = clock.tick(60) / 1000

        WIN.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()


        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:  # Left mouse button held
            mouse_pos = pygame.mouse.get_pos()
            dx = mouse_pos[0] - prev_mouse_pos[0]
            dy = mouse_pos[1] - prev_mouse_pos[1]
            vx = dx / delt
            vy = dy / delt
            density = add_density(density)
            velx, vely = add_velocity(velx, vely, vx, vy)

        prev_mouse_pos = mouse_pos

        step()
        draw(WIN)
        pygame.display.flip()


if __name__ == '__main__':
    simulate()


