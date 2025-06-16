# 💧 2D Fluid Simulation in Python

This project is a real-time 2D fluid simulation implemented in **Python** using **NumPy** and **Pygame**. It simulates fluid behavior such as smoke or dye using a grid-based velocity and density field.

---

## 📘 Reference

This simulation is based on the semi-Lagrangian method described in the paper:

**Real-Time Fluid Dynamics for Games**  
  by Jos Stam  
📄 [Read the paper here (PDF)](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf)

---

## 🎥 Inspirations

This implementation was inspired by these excellent visual and educational resources:

- 🌊 **The Coding Train - Fluid Simulation**  
  [Watch on YouTube](https://www.youtube.com/watch?v=alhpH6ECFvQ)

---

## 🧠 Features

- Real-time interactive simulation
- Fluid advection, diffusion, and incompressibility enforcement
- Mouse-driven density and velocity injection
- Density fading and clamping for stability

---

## 🔧 Requirements

- Python 3.7+
- Pygame
- NumPy
- Matplotlib for advanced coloring

Install dependencies with:

```bash
pip install pygame numpy matplotlib
```

## 🕹️ Controls

| Action                  | Input            |
|-------------------------|------------------|
| Add dye & velocity      | Left mouse drag  |
| Exit simulation         | Close window     |

## 🌀 Demo

![Fluid Simulation Demo](fluid_sim.gif)


