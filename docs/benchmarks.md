# Literature Benchmark Validation (Phase 6)

This document records the formal benchmark validation of **PinchPy** against classic, peer-reviewed Pinch Analysis literature problems.

---

## 1. Summary of Benchmark Validations

| Benchmark Problem | Literature Source | Streams | $\Delta T_{min}$ (°C) | $Q_{H,min}$ (kW) | $Q_{C,min}$ (kW) | $T_{pinch,hot}$ (°C) | $T_{pinch,cold}$ (°C) | Heat Recovery (kW) | Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kemp (2007) Base** | Ian C. Kemp (2007) *Pinch Analysis & Process Integration*, Table 1.1 | 4 | 10.0 | **20.0** | **60.0** | 90.0 | 80.0 | 450.0 | Exact |
| **Kemp (2007) High $\Delta T$** | Ian C. Kemp (2007), Tables 2.2, 2.3 | 4 | 20.0 | **65.0** | **105.0** | 100.0 | 80.0 | 405.0 | Exact |
| **Towler & Sinnott / Linnhoff** | Towler & Sinnott, *Chemical Engineering Design*; Linnhoff et al. (1982) | 4 | 20.0 | **2,900.0** | **600.0** | 100.0 | 80.0 | 7,700.0 | Exact |
| **Robin Smith (2005)** | Robin Smith, *Chemical Process Design & Integration*, Table 16.1 | 2 | 10.0 | **3,000.0** | **1,000.0** | 50.0 | 40.0 | 11,000.0 | Exact |
| **4SP1 Interior Pinch** | Lee, Masso & Rudd (1970); Papoulias & Grossmann (1983) | 4 | 20.0 | **188.48** | **42.94** | 249.0 | 229.0 | 1,449.04 | Exact |
| **4SP1 Threshold Problem** | Lee, Masso & Rudd (1970); Papoulias & Grossmann (1983) | 4 | 10.0 | **145.54** | **0.00** | 70.0 | 60.0 | 1,491.98 | Exact |

---

## 2. Benchmark Case Specifications

### 2.1 Kemp (2007) 4-Stream Process
- **Hot Streams**:
  - `H1`: $T_s = 170\,^\circ\text{C}, T_t = 60\,^\circ\text{C}, CP = 3.0\,\text{kW/}^\circ\text{C}$ ($Q = 330\,\text{kW}$)
  - `H2`: $T_s = 150\,^\circ\text{C}, T_t = 30\,^\circ\text{C}, CP = 1.5\,\text{kW/}^\circ\text{C}$ ($Q = 180\,\text{kW}$)
- **Cold Streams**:
  - `C1`: $T_s = 20\,^\circ\text{C}, T_t = 135\,^\circ\text{C}, CP = 2.0\,\text{kW/}^\circ\text{C}$ ($Q = 230\,\text{kW}$)
  - `C2`: $T_s = 80\,^\circ\text{C}, T_t = 140\,^\circ\text{C}, CP = 4.0\,\text{kW/}^\circ\text{C}$ ($Q = 240\,\text{kW}$)

### 2.2 Towler & Sinnott / Linnhoff et al. (1982)
- **Hot Streams**:
  - `H1`: $T_s = 180\,^\circ\text{C}, T_t = 40\,^\circ\text{C}, CP = 40.0\,\text{kW/}^\circ\text{C}$ ($Q = 5,600\,\text{kW}$)
  - `H2`: $T_s = 150\,^\circ\text{C}, T_t = 60\,^\circ\text{C}, CP = 30.0\,\text{kW/}^\circ\text{C}$ ($Q = 2,700\,\text{kW}$)
- **Cold Streams**:
  - `C1`: $T_s = 30\,^\circ\text{C}, T_t = 180\,^\circ\text{C}, CP = 60.0\,\text{kW/}^\circ\text{C}$ ($Q = 9,000\,\text{kW}$)
  - `C2`: $T_s = 80\,^\circ\text{C}, T_t = 160\,^\circ\text{C}, CP = 20.0\,\text{kW/}^\circ\text{C}$ ($Q = 1,600\,\text{kW}$)

### 2.3 Robin Smith (2005) Table 16.1
- **Hot Stream**:
  - `H1`: $T_s = 160\,^\circ\text{C}, T_t = 40\,^\circ\text{C}, CP = 100.0\,\text{kW/}^\circ\text{C}$ ($Q = 12,000\,\text{kW} = 12\,\text{MW}$)
- **Cold Stream**:
  - `C1`: $T_s = 40\,^\circ\text{C}, T_t = 110\,^\circ\text{C}, CP = 200.0\,\text{kW/}^\circ\text{C}$ ($Q = 14,000\,\text{kW} = 14\,\text{MW}$)

### 2.4 Lee, Masso & Rudd (1970) / Papoulias & Grossmann (1983) 4SP1
- **Hot Streams**:
  - `H1`: $T_s = 160\,^\circ\text{C}, T_t = 93\,^\circ\text{C}, CP = 4.79\,\text{kW/}^\circ\text{C}$ ($Q = 320.93\,\text{kW}$)
  - `H2`: $T_s = 249\,^\circ\text{C}, T_t = 138\,^\circ\text{C}, CP = 10.55\,\text{kW/}^\circ\text{C}$ ($Q = 1,171.05\,\text{kW}$)
- **Cold Streams**:
  - `C1`: $T_s = 60\,^\circ\text{C}, T_t = 160\,^\circ\text{C}, CP = 7.62\,\text{kW/}^\circ\text{C}$ ($Q = 762.0\,\text{kW}$)
  - `C2`: $T_s = 116\,^\circ\text{C}, T_t = 260\,^\circ\text{C}, CP = 6.08\,\text{kW/}^\circ\text{C}$ ($Q = 875.52\,\text{kW}$)

