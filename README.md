# PixelWave

A realtime 2D wave simulation running entirely on the GPU via Metal compute and render shaders. Touch the screen to inject ripples and watch them propagate, reflect off boundaries, interfere with each other, and gradually decay all rendered with a stylised ocean-water aesthetic at up to 120 Hz.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Simulation Pipeline](#simulation-pipeline)
  - [Frame Loop](#frame-loop)
  - [Fixed-Timestep Accumulator](#fixed-timestep-accumulator)
  - [Texture Triple-Buffering](#texture-triple-buffering)
- [The Wave Equation](#the-wave-equation)
  - [Continuous Form](#continuous-form)
  - [Finite-Difference Discretisation](#finite-difference-discretisation)
  - [9-Point Laplacian Stencil](#9-point-laplacian-stencil)
  - [Dispersion Correction (Micro-Chop)](#dispersion-correction-micro-chop)
- [Viscosity Model](#viscosity-model)
- [Damping](#damping)
- [Boundary Conditions](#boundary-conditions)
- [Disturbance Injection](#disturbance-injection)
  - [Gaussian Impulse Kernel](#gaussian-impulse-kernel)
  - [Displacement–Velocity Split](#displacementvelocity-split)
- [Rendering Pipeline](#rendering-pipeline)
  - [Vertex Stage](#vertex-stage)
  - [Fragment Stage — Water Shading](#fragment-stage--water-shading)
  - [Activity-Based Transparency](#activity-based-transparency)
  - [Premultiplied Alpha Blending](#premultiplied-alpha-blending)
- [Configurable Parameters](#configurable-parameters)
- [File Reference](#file-reference)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  ContentView (SwiftUI)                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  WaveSimulationView (UIViewRepresentable)              │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  InteractiveMTKView (MTKView subclass)           │  │  │
│  │  │  ┌────────────────────────────────────────────┐  │  │  │
│  │  │  │  WaveRenderer (MTKViewDelegate)            │  │  │  │
│  │  │  │  ┌─────────────┐ ┌──────────────────────┐  │  │  │  │
│  │  │  │  │  Compute    │ │  Render              │  │  │  │  │
│  │  │  │  │  Pipelines  │ │  Pipeline            │  │  │  │  │
│  │  │  │  │  • waveStep │ │  • waveVertex        │  │  │  │  │
│  │  │  │  │  • disturb  │ │  • waveFragment      │  │  │  │  │
│  │  │  │  └─────────────┘ └──────────────────────┘  │  │  │  │
│  │  │  └────────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

| Layer | Responsibility |
|---|---|
| **ContentView** | SwiftUI host. Places `WaveSimulationView` over any background and overlays the control panel. |
| **WaveSimulationView** | Bridges SwiftUI ↔ UIKit. Creates the `MTKView`, wires up the renderer, and forwards touch events. |
| **InteractiveMTKView** | `MTKView` subclass that converts `UITouch` locations to normalised UV coordinates with pressure. |
| **WaveRenderer** | `MTKViewDelegate`. Owns the Metal device, command queue, compute/render pipelines, simulation textures, and the frame loop. |
| **WaveShaders.metal** | All GPU code: simulation kernels (`waveStep`, `applyDisturbances`) and rendering shaders (`waveVertex`, `waveFragment`). |

---

## Simulation Pipeline

### Frame Loop

On every display refresh (`draw(in:)`), the renderer:

1. **Measures frame delta** from `CACurrentMediaTime()`.
2. **Consumes pending disturbances** from the touch queue (thread-safe via `NSLock`).
3. **Encodes the disturbance compute pass** to inject impulses into the simulation textures.
4. **Runs N simulation steps** via the fixed-timestep accumulator.
5. **Encodes the render pass** — a full-screen quad textured with the current wave height field.
6. **Presents** the drawable.

### Fixed-Timestep Accumulator

The simulation runs at a locked **120 Hz** physics rate ($\Delta t = \frac{1}{120}\text{s}$), decoupled from the display refresh rate.

```
accumulator += frameDelta
while accumulator >= Δt and steps < maxSteps:
    encode waveStep
    rotate textures
    accumulator -= Δt
    steps += 1
```

- **`maxSimulationStepsPerFrame = 10`** — prevents spiral-of-death when the frame rate drops.
- If max steps are hit, the accumulator is reset to zero (drops time rather than accumulating debt).

### Texture Triple-Buffering

Three single-channel `R32Float` textures hold the wave height field:

| Texture | Role |
|---|---|
| `previousTexture` | Height at time $t - \Delta t$ |
| `currentTexture` | Height at time $t$ |
| `nextTexture` | Write target for time $t + \Delta t$ |

After each simulation step, the textures **rotate**:

$$
\text{previous} \leftarrow \text{current}, \quad
\text{current} \leftarrow \text{next}, \quad
\text{next} \leftarrow \text{previous}\ (\text{recycled})
$$

The grid resolution is derived from the drawable size divided by `pixelSize` (default 3.0), giving a chunky, pixel-art style grid.

---

## The Wave Equation

### Continuous Form

The simulation solves the **2D wave equation** with damping:

$$
\frac{\partial^2 u}{\partial t^2} + \gamma \frac{\partial u}{\partial t} = c^2 \nabla^2 u
$$

Where:

| Symbol | Meaning |
|---|---|
| $u(x, y, t)$ | Wave height (displacement) |
| $c$ | Wave propagation speed |
| $\gamma$ | Damping coefficient |
| $\nabla^2 u$ | Laplacian — spatial curvature of the surface |

### Finite-Difference Discretisation

Time is discretised using the **Verlet (leapfrog)** integration scheme. Given height values at two consecutive timesteps:

$$
v^{n} = u^{n} - u^{n-1}
$$

$$
a^{n} = \sigma \cdot \nabla^2 u^{n}
$$

$$
v^{n} \leftarrow v^{n} + a^{n}
$$

$$
v^{n} \leftarrow v^{n} \cdot e^{-\gamma \Delta t}
$$

$$
u^{n+1} = u^{n} + v^{n}
$$

Where the **propagation factor** $\sigma$ is the squared Courant number, clamped for stability:

$$
\sigma = \min\!\left((c \cdot \Delta t)^2,\ 0.48\right)
$$

The CFL (Courant–Friedrichs–Lewy) condition requires $\sigma < 0.5$ for the explicit scheme to remain stable. The clamp at $0.48$ provides a small safety margin.

### 9-Point Laplacian Stencil

Rather than the basic 5-point cross stencil, the simulation uses a **weighted 9-point stencil** for isotropic accuracy. The stencil weights are:

$$
\frac{1}{6}
\begin{bmatrix}
1 & 4 & 1 \\
4 & -20 & 4 \\
1 & 4 & 1
\end{bmatrix}
$$

This is computed by decomposing into cardinal and diagonal components:

$$
\nabla^2_4 = u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4\,u_{i,j}
$$

$$
\nabla^2_{\text{diag}} = u_{i-1,j-1} + u_{i+1,j-1} + u_{i-1,j+1} + u_{i+1,j+1} - 4\,u_{i,j}
$$

$$
\nabla^2 u \approx \frac{4\,\nabla^2_4 + \nabla^2_{\text{diag}}}{6}
$$

**Why 9-point?** The standard 5-point stencil has directional bias — waves travel faster along the grid axes than diagonals. The 9-point stencil dramatically reduces this anisotropy, producing rounder wavefronts.

### Dispersion Correction (Micro-Chop)

The difference between the diagonal and cardinal Laplacians captures high-frequency anisotropic content:

$$
\mu = \nabla^2_{\text{diag}} - \nabla^2_4
$$

This is blended into the acceleration by the dispersion parameter $\delta$:

$$
a = \sigma \cdot \left(\nabla^2 u + \delta \cdot \mu\right)
$$

At $\delta = 0$, waves are smooth. Higher values add high-frequency "chop" that mimics surface tension effects on small-scale ripples.

---

## Viscosity Model

Viscosity smooths the velocity field by blending each cell's velocity with the average of its 4-connected neighbours:

$$
\bar{v} = \frac{1}{4}\sum_{k \in \{L,R,U,D\}} \left(u_k^n - u_k^{n-1}\right)
$$

$$
\beta = \text{clamp}\!\left(\nu \cdot \Delta t \cdot 12,\; 0,\; 0.5\right)
$$

$$
v \leftarrow (1 - \beta)\,v + \beta\,\bar{v}
$$

Where $\nu$ is the viscosity parameter.

- The $\Delta t \cdot 12$ scaling makes the viscosity framerate-independent.
- Clamping $\beta$ to $0.5$ prevents over-smoothing that would cause instability.
- At $\nu = 0$, each cell is independent. At higher values, neighbouring cells synchronise, damping high-frequency content while preserving large-scale motion.

---

## Damping

Energy dissipation is modelled as exponential drag on the velocity:

$$
v \leftarrow v \cdot e^{-\gamma \cdot \Delta t}
$$

This is framerate-independent. At $\gamma = 0$, energy is conserved indefinitely. Higher values cause ripples to fade faster.

---

## Boundary Conditions

The edges use an **absorbing boundary** with configurable partial reflection. For each cell, the attenuation factor $\alpha_b$ is:

$$
d_{\text{edge}} = \min\!\left(x,\; W-1-x,\; y,\; H-1-y\right)
$$

$$
w = \max\!\left(1,\; b_w \cdot \min(W, H)\right)
$$

$$
\eta = \text{smoothstep}\!\left(0,\; 1,\; \frac{d_{\text{edge}}}{w}\right)
$$

$$
\alpha_b = \text{mix}\!\left(r,\; 1,\; \eta\right)
$$

Where:

| Symbol | Meaning |
|---|---|
| $d_{\text{edge}}$ | Distance from cell to nearest edge |
| $b_w$ | Boundary width fraction |
| $r$ | Reflection coefficient ($0$ = absorb, $1$ = mirror) |
| $\eta$ | Interior factor ($0$ at edge, $1$ deep inside) |

Both velocity and height are multiplied by $\alpha_b$:

$$
v \leftarrow v \cdot \alpha_b, \qquad u^{n+1} \leftarrow u^{n+1} \cdot \alpha_b
$$

The `smoothstep` ramp prevents hard discontinuities at the boundary edge.

---

## Disturbance Injection

### Gaussian Impulse Kernel

When the user touches the screen, disturbances are injected via the `applyDisturbances` compute kernel. Each disturbance $k$ at center $\mathbf{c}_k$ with radius $r_k$ and strength $s_k$ produces a **Gaussian radial falloff**:

$$
\boldsymbol{\delta} = \mathbf{uv} - \mathbf{c}_k
$$

$$
d^2 = \boldsymbol{\delta} \cdot \boldsymbol{\delta}
$$

$$
f(d) = \begin{cases}
\exp\!\left(\dfrac{-d^2}{r_k^2 \cdot 0.18}\right) & \text{if } d^2 < r_k^2 \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

$$
I = \sum_k s_k \cdot f_k(d)
$$

The $0.18$ factor in the denominator controls the Gaussian's tightness — it concentrates ~82% of the energy within the inner 42% of the radius, creating a sharp central peak with soft edges.

### Displacement–Velocity Split

The total impulse $I$ is split between **displacement** (position offset) and **velocity kick**:

$$
d_{\text{pos}} = 0.18 \cdot I \qquad (\text{displacement})
$$

$$
v_{\text{kick}} = 0.82 \cdot I \qquad (\text{velocity})
$$

These are injected into the Verlet pair:

$$
u^n \leftarrow u^n + d_{\text{pos}} + \tfrac{1}{2}\,v_{\text{kick}}
$$

$$
u^{n-1} \leftarrow u^{n-1} + d_{\text{pos}} - \tfrac{1}{2}\,v_{\text{kick}}
$$

**Why split?** In a Verlet scheme, velocity is implicit: $v = u^n - u^{n-1}$. To inject both position and velocity:

- Adding the same $d_{\text{pos}}$ to both textures shifts the surface without creating velocity.
- Adding $+\frac{1}{2}v_{\text{kick}}$ to current and $-\frac{1}{2}v_{\text{kick}}$ to previous creates a net velocity of $v_{\text{kick}}$:

$$
v_{\text{new}} = \left(u^n + d + \tfrac{1}{2}k\right) - \left(u^{n-1} + d - \tfrac{1}{2}k\right) = v_{\text{old}} + k
$$

The **82/18 split** biases toward velocity injection, which creates more natural-looking expanding ripples rather than a static bump.

---

## Rendering Pipeline

### Vertex Stage

A **full-screen triangle pair** (6 vertices, 2 triangles) covers the entire viewport:

| Vertex | Position (clip space) | UV |
|---|---|---|
| 0 | $(-1, -1)$ | $(0, 0)$ |
| 1 | $(+1, -1)$ | $(1, 0)$ |
| 2 | $(-1, +1)$ | $(0, 1)$ |
| 3 | $(+1, -1)$ | $(1, 0)$ |
| 4 | $(+1, +1)$ | $(1, 1)$ |
| 5 | $(-1, +1)$ | $(0, 1)$ |

### Fragment Stage — Water Shading

The fragment shader reads the wave height field and produces a stylised ocean surface.

#### 1. Height Sampling

The shader samples the wave height at 5 points (center + 4 cardinal neighbours) using nearest-neighbour filtering (preserving the pixel-art grid).

#### 2. Slope and Curvature

$$
s_x = u_R - u_L, \qquad s_y = u_U - u_D
$$

$$
|\mathbf{s}| = \sqrt{s_x^2 + s_y^2}
$$

$$
\kappa = |u_L + u_R + u_U + u_D - 4\,u_C|
$$

#### 3. Surface Normal

The height gradient is converted to a 3D surface normal for lighting:

$$
\hat{\mathbf{n}} = \text{normalize}\!\left(-2.2\,s_x,\; -2.2\,s_y,\; 1.0\right)
$$

The $2.2$ scaling exaggerates the normal deflection for more dramatic lighting.

#### 4. Blinn-Phong Lighting

A directional light and view-from-above camera:

$$
\hat{\mathbf{l}} = \text{normalize}(-0.28,\; 0.62,\; 0.74)
$$

$$
\hat{\mathbf{v}} = (0,\; 0,\; 1)
$$

$$
\hat{\mathbf{h}} = \text{normalize}(\hat{\mathbf{l}} + \hat{\mathbf{v}})
$$

**Diffuse**:

$$
I_d = \max(\hat{\mathbf{n}} \cdot \hat{\mathbf{l}},\; 0)
$$

**Fresnel** (Schlick-like, power 4):

$$
F = \left(1 - \text{saturate}(\hat{\mathbf{n}} \cdot \hat{\mathbf{v}})\right)^4
$$

**Specular** with adaptive gloss:

$$
g = \text{mix}\!\left(28,\; 88,\; 1 - \text{saturate}(2.6\,\kappa)\right)
$$

$$
I_s = \max(\hat{\mathbf{n}} \cdot \hat{\mathbf{h}},\; 0)^{\,g} \cdot (0.14 + 0.45\,F)
$$

Calm areas → tight highlights ($g = 88$). Turbulent areas → broad highlights ($g = 28$).

#### 5. Depth-Based Water Color

The wave height drives a blend between deep and shallow water tones. All three colors are **user-configurable** via `ColorPicker` controls in the UI:

$$
m = \text{saturate}(0.5 + 0.45\,u)
$$

| Color | Default RGB | Description |
|---|---|---|
| $C_{\text{deep}}$ | $(0.004,\; 0.028,\; 0.075)$ | Deep ocean navy |
| $C_{\text{shallow}}$ | $(0.008,\; 0.135,\; 0.22)$ | Ocean teal |
| $C_{\text{sky}}$ | $(0.36,\; 0.54,\; 0.72)$ | Desaturated sky blue |

$$
C_{\text{water}} = \text{mix}(C_{\text{deep}},\; C_{\text{shallow}},\; m)
$$

$$
C_{\text{reflect}} = \text{mix}\!\left((0.09, 0.16, 0.24),\; C_{\text{sky}},\; F\right)
$$

#### 6. Foam

White foam appears on wave crests and areas of high curvature:

$$
\text{foam} = \text{smoothstep}\!\left(0.22,\; 0.62,\; \kappa + 0.15\,|\mathbf{s}|\right)
$$

#### 7. Pixel Grid Accent

A subtle grid overlay emphasises the pixel-art aesthetic:

$$
\mathbf{cell} = \left|\text{fract}(\mathbf{uv} \cdot \text{texSize}) - 0.5\right|
$$

$$
p = \text{smoothstep}\!\left(0.45,\; 0.5,\; \max(\text{cell}_x,\; \text{cell}_y)\right)
$$

$$
C \leftarrow C \cdot \text{mix}(0.98,\; 1.02,\; p)
$$

This brightens the edges of each simulation cell by 2% and darkens the centers by 2%, creating a visible grid without harsh lines.

#### 8. Final Composition

$$
C = C_{\text{water}} \cdot (0.3 + 0.65\,I_d) + 0.42\,C_{\text{reflect}} + I_s + \text{foam} \cdot (0.2,\; 0.24,\; 0.25)
$$

### Activity-Based Transparency

The fragment shader outputs **transparent black** for calm surface areas. Wave activity is computed as:

$$
A = \text{saturate}\!\left(6\,|u| + 12\,|\mathbf{s}| + 8\,\kappa\right)
$$

If $A < 10^{-4}$, the fragment returns $(0, 0, 0, 0)$ immediately.

Otherwise, the alpha fades smoothly:

$$
\alpha = \text{smoothstep}(0,\; 0.15,\; A)
$$

$$
\text{output} = \left(\alpha \cdot C,\; \alpha\right) \qquad \text{(premultiplied)}
$$

This allows the SwiftUI view behind `WaveSimulationView` to show through — you can place any background (image, gradient, solid color) and ripples will composite over it.

### Premultiplied Alpha Blending

The render pipeline uses **premultiplied alpha** blending:

$$
C_{\text{src}} = C \cdot \alpha \qquad \text{(done in shader)}
$$

$$
C_{\text{final}} = C_{\text{src}} \cdot 1 + C_{\text{dst}} \cdot (1 - \alpha)
$$

```
sourceRGBBlendFactor      = .one
destinationRGBBlendFactor = .oneMinusSourceAlpha
```

This avoids dark fringing at ripple edges and is the standard compositing mode.

---

## Configurable Parameters

All parameters are exposed in the `WaveParameters` struct and controlled via the UI. Defaults are tuned for **realistic ocean physics**. Press **Defaults** in the control panel to restore them.

### Physics

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `waveSpeed` | $12.0$ | $6$–$32$ | Propagation speed. Higher = faster expanding ripples. Clamped by CFL. |
| `damping` | $0.35$ | $0.05$–$2.2$ | Energy loss rate ($\gamma$). Lower values let waves persist longer. |
| `viscosity` | $0.18$ | $0$–$1.2$ | Velocity smoothing ($\nu$). Higher = fewer high-frequency ripples. |
| `dispersion` | $0.12$ | $0$–$1$ | High-frequency chop ($\delta$). Blends anisotropic Laplacian difference. |
| `edgeReflection` | $0.42$ | $0.1$–$1$ | Boundary reflection coefficient ($r$). $0$ = absorb, $1$ = mirror. |
| `edgeWidth` | $0.065$ | $0.01$–$0.2$ | Absorption zone width ($b_w$), fraction of grid. |
| `brushRadius` | $0.035$ | $0.01$–$0.08$ | Touch radius in normalised UV space ($r_k$). |
| `impulse` | $0.55$ | $0.1$–$1.2$ | Touch strength multiplier ($s_k$). |
| `pixelSize` | $3.0$ | — | Screen pixels per simulation cell (controls grid resolution). |

### Colors

Three colors are configurable via `ColorPicker` controls:

| Color | Default RGB | Role |
|---|---|---|
| Deep | $(0.004,\; 0.028,\; 0.075)$ | Base color for wave troughs — deep ocean navy |
| Shallow | $(0.008,\; 0.135,\; 0.22)$ | Base color for wave crests — ocean teal |
| Sky | $(0.36,\; 0.54,\; 0.72)$ | Fresnel reflection tint — desaturated sky blue |

---

## File Reference

| File | Lines | Description |
|---|---|---|
| `WaveShaders.metal` | ~246 | GPU kernels and shaders — all simulation and rendering logic |
| `WaveRenderer.swift` | ~385 | Metal setup, frame loop, compute/render encoding, texture management |
| `WaveSimulationView.swift` | ~124 | SwiftUI↔UIKit bridge, MTKView configuration, touch handling |
| `ContentView.swift` | ~240 | SwiftUI host with control panel, sliders, color pickers |
| `PixelWaveApp.swift` | ~10 | App entry point |
