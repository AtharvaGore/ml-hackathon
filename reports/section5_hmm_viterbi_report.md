# PathSnap: Section 5 - HMM and Viterbi Route Reconstruction

## 5.1 Route Reconstruction Architecture

Map matching independently for each GPS observation (classifier-only) frequently assigns points to parallel streets or opposite-direction lanes due to GPS multipath error, resulting in physically disconnected trajectories. To enforce topological plausibility, we implemented a Hidden Markov Model (HMM) framework following the formulation of Newson & Krumm (2009) with dynamic network caching, adaptive temporal scaling, and classifier fusion.

```
GPS Observation z_t (lon, lat, t)
          │
          ▼
   Spatial KD-Tree Query (R = 50m)
          │
          ▼
   Candidate Arcs C_t = {c_{t,1}, ..., c_{t,K}}
          │
          ├──► Emission Modeling: ln P(z_t | c_{t,i})
          │    (Gaussian distance error + Classifier probability)
          │
          └──► Transition Modeling: ln P(c_{t,i} | c_{t-1,j})
               (Network shortest path d_net vs GPS Euclidean displacement d_gps)
                    │
                    ▼
          Viterbi Trellis Dynamic Programming (Log-Space)
                    │
                    ▼
          Globally Optimal Continuous Route Sequence s*
```

---

## 5.2 Compulsory Questions & Methodology

### 1. What are the hidden states in your implementation?
- The hidden states at each discrete observation step $t \in \{0, 1, \dots, T-1\}$ are the candidate directed road segments (arcs) $C_t = \{c_{t,1}, c_{t,2}, \dots, c_{t,K}\}$ within a spatial threshold $R = 50\text{ m}$ of observation $z_t$.
- Pruning the entire graph ($>100,000$ arcs) to local candidate states reduces the Viterbi trellis complexity from an intractable $O(T \cdot |E|^2)$ to $O(T \cdot K^2)$ where $K \le 10$, achieving an average decoding latency of **0.19 seconds per trip**.
- Each state records: `arc_id`, tail/head nodes $(u, v)$, arc length $L$, perpendicular distance $d$, and fractional projection $\tau \in [0, 1]$.

### 2. How are emission probabilities calculated?
The emission probability reflects the likelihood of observing GPS location $z_t$ assuming the vehicle is actually travelling along candidate segment $c_{t,i}$.
1. **Geometric Gaussian Error**:
   $$P_{\text{Gauss}}(z_t | c_{t,i}) = \frac{1}{\sqrt{2\pi}\sigma_z} \exp\left(-\frac{\text{dist}(z_t, c_{t,i})^2}{2\sigma_z^2}\right)$$
   In log-space:
   $$\ln P_{\text{Gauss}}(z_t | c_{t,i}) = -\frac{1}{2}\ln(2\pi\sigma_z^2) - \frac{\text{dist}(z_t, c_{t,i})^2}{2\sigma_z^2}$$
   where $\text{dist}(z_t, c_{t,i})$ is the exact orthogonal distance in projected metric space (meters), and $\sigma_z = 7.5\text{ m}$ corresponds to standard empirical urban GPS error.
2. **Classifier Score Fusion**:
   When integrated with an upstream candidate ranking classifier $p_{\text{model}}(c_{t,i}) \in [0, 1]$:
   $$\ln P(z_t | c_{t,i}) = (1 - \alpha)\ln P_{\text{Gauss}}(z_t | c_{t,i}) + \alpha \ln p_{\text{model}}(c_{t,i})$$

### 3. How are transition probabilities calculated?
The transition probability models the physical plausibility of moving from candidate $c_{t-1,j}$ to $c_{t,i}$ in the road network:
$$P(c_{t,i} | c_{t-1,j}) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{net}} - d_{\text{gps}}|}{\beta}\right)$$
In log-space:
$$\ln P(c_{t,i} | c_{t-1,j}) = -\ln\beta - \frac{|d_{\text{net}} - d_{\text{gps}}|}{\beta}$$
where:
- $d_{\text{gps}} = \|z_t - z_{t-1}\|_2$ is the direct Euclidean displacement between consecutive GPS readings.
- $d_{\text{net}}$ is the shortest path travel distance along the directed road network:
  - **Same arc** ($c_{t-1,j} == c_{t,i}$): $d_{\text{net}} = (\tau_{t,i} - \tau_{t-1,j}) \times L_{c}$.
  - **Directly connected arcs** ($v_{t-1} == u_t$): $d_{\text{net}} = (1 - \tau_{t-1,j}) L_{c_{t-1}} + \tau_{t,i} L_{c_t}$.
  - **Multi-hop transition**: $d_{\text{net}} = (1 - \tau_{t-1,j}) L_{c_{t-1}} + d_{\text{path}}(v_{t-1}, u_t) + \tau_{t,i} L_{c_t}$, where $d_{\text{path}}$ is obtained via bounded Dijkstra search ($O(\text{local subgraph})$).
- $\beta = 7.0\text{ m}$ is the exponential scale factor.

### 4. How are missing candidates or large GPS gaps handled?
- **Missing Candidates**: If no road arc lies within the standard $50\text{m}$ radius (e.g. severe GPS multipath or off-road maneuver), the decoder dynamically falls back to the nearest spatial arc rather than dropping the state.
- **Large GPS Gaps**: When timestamp difference $\Delta t = \text{time}_t - \text{time}_{t-1} > 2.0\text{ s}$ (e.g., tunnels or GPS dropouts), the effective scale parameter is adapted:
  $$\beta_{\text{eff}} = \beta \times \left(1 + \max\left(0, \frac{\Delta t - 2.0}{5.0}\right)\right)$$
  This accommodates the wider variance in possible network paths over extended time intervals without crashing the probability score.

### 5. What happens when consecutive GPS observations suggest disconnected road segments?
If two consecutive candidates are topologically disconnected in the directed road graph (no path exists within the plausible kinematic cutoff $v_{\max} \Delta t$), assigning $P = 0$ ($-\infty$ in log space) would break the entire Viterbi path. Instead:
- We assign a calibrated disconnection penalty $\ln P_{\text{trans}} = -35.0$.
- This strongly discourages disconnected hops while allowing the decoder to recover gracefully across digitizing errors or missing one-way overrides in the raw OSM map data.

### 6. Experimental Results & Performance Comparison

The decoder was evaluated across all **10 reserved Student-Testing trips** (19,161 total GPS observations).

| Trip ID | Observations | Classifier Top-1 | HMM Viterbi Top-1 | Accuracy Gain | Classifier Continuity | HMM Continuity | Continuity Gain | Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00000000** | 2,503 | 48.46% | **93.09%** | +44.63% | 90.53% | **99.16%** | +8.63% | 0.20s |
| **00000007** | 1,354 | 44.61% | **95.49%** | +50.89% | 92.54% | **100.00%** | +7.46% | 0.08s |
| **00000009** | 1,049 | 43.66% | **84.18%** | +40.51% | 64.98% | **93.80%** | +28.82% | 0.12s |
| **00000040** | 2,995 | 53.22% | **89.92%** | +36.69% | 89.45% | **99.27%** | +9.82% | 0.20s |
| **00000052** | 262 | 80.53% | **97.33%** | +16.79% | 78.16% | **96.55%** | +18.39% | 0.02s |
| **00000056** | 3,958 | 65.99% | **88.02%** | +22.03% | 75.99% | **96.13%** | +20.14% | 0.43s |
| **00000065** | 768 | 56.90% | **76.30%** | +19.40% | 79.66% | **97.91%** | +18.25% | 0.12s |
| **00000076** | 1,486 | 41.32% | **88.02%** | +46.70% | 72.53% | **96.09%** | +23.57% | 0.32s |
| **00000077** | 3,911 | 36.84% | **66.56%** | +29.71% | 85.75% | **99.05%** | +13.30% | 0.37s |
| **00000091** | 874 | 57.21% | **88.67%** | +31.46% | 75.03% | **96.91%** | +21.88% | 0.09s |
| **MEAN** | **1,916** | **52.88%** | **86.76%** | **+33.88%** | **80.46%** | **97.49%** | **+17.03%** | **0.19s** |

### Discussion of Improvements and Edge Cases
- **Why HMM drastically improved the route**:
  At intersections and parallel multi-lane service roads, GPS noise scatters raw observations onto neighboring links. A pointwise classifier picks these spurious links, causing ping-ponging and broken chains. The HMM enforces that transitions must adhere to shortest network paths matching physical speed, filtering out high-variance GPS spikes.
- **Edge cases where HMM did not completely eliminate errors (e.g. Trip 00000077)**:
  In Trip 77 (66.56% Top-1 accuracy), the trajectory traversed dense multi-level urban overpasses and complex service road roundabouts with missing candidate segments (candidate miss rate 5.18%). If the true arc is absent from the candidate pool during a complex junction, the Viterbi trellis is forced onto an adjacent parallel arterial to maintain topological continuity, accumulating lateral offset before re-synchronizing.
