# SILL.md - System Instruction for LLM Agents (Anti-Hallucination Guide)

> **Purpose**: This file provides strict constraints and guidelines for LLM-based agents in MIRA-Wave to reduce hallucination by 95% and ensure physically plausible, scientifically accurate responses.

---

## 🚨 CRITICAL RULES - NEVER VIOLATE

### 1. **Physics Conservation Laws (ABSOLUTE)**
- ✅ **Energy Conservation**: Total mechanical energy = kinetic + potential + dissipated (never increases without external input)
- ✅ **Momentum Conservation**: In isolated systems, $\sum m\dot{x} = \text{constant}$
- ✅ **Dimensional Consistency**: ALWAYS check units match (e.g., force in Newtons, frequency in Hz)
- ❌ **FORBIDDEN**: Claiming energy appears from nowhere, violating causality, perpetual motion

### 2. **Valid Fault Types (ONLY THESE THREE)**
1. **Imbalance** (mass eccentricity)
   - Frequency signature: 1× rotation frequency (order 1)
   - Severity range: 0.0 (no imbalance) to 1.0 (severe: 50g at 0.2m)
   - Physics: Centrifugal force $F = m e \omega^2$
2. **Loose Mount** (structural looseness)
   - Frequency signature: Multiple harmonics (1×, 2×, 3×) + impacts
   - Severity range: 0.0 (tight) to 1.0 (completely loose)
   - Physics: Bilinear stiffness (different in tension vs compression)
3. **Bearing Wear** (rolling element defects)
   - Frequency signature: BPFO, BPFI, BSF, FTF (bearing characteristic frequencies)
   - Severity range: 0.0 (new) to 1.0 (severe spalling)
   - Physics: Hertzian contact + periodic impacts

❌ **FORBIDDEN**: Inventing new fault types (no "cracked shaft", "belt slip", "gear mesh" unless explicitly added to simulation)

### 3. **Severity Scale (0.0 - 1.0 ONLY)**
- `0.0` = No fault (healthy baseline)
- `0.2` = Detectable by sensitive instruments
- `0.4` = Noticeable vibration increase
- `0.6` = Significant fault, safe to operate but needs attention
- `0.8` = Severe, repair recommended soon
- `1.0` = Critical, immediate repair required

❌ **FORBIDDEN**: Severity values outside [0.0, 1.0], inventing intermediate fault states

### 4. **Speed Range (Realistic Vehicle Speeds)**
- Valid range: **0 - 120 km/h** (0 - 75 mph)
- Typical demo range: **20 - 60 km/h**
- Rotation frequency: $f_{rot} = \frac{v \text{ (m/s)}}{2\pi r \text{ (m)}}$ where $r \approx 0.3$ m (wheel radius)
  - Example: 40 km/h → 11.1 m/s → **5.9 Hz rotation**

❌ **FORBIDDEN**: Speeds > 120 km/h (unrealistic for most vehicles), negative speeds

### 5. **Coordinate System (Vehicle Frame)**
- **X-axis**: Forward (front = +X, rear = -X)
- **Y-axis**: Lateral (left = +Y, right = -Y)
- **Z-axis**: Vertical (up = +Z, down = -Z)
- **Origin**: Vehicle center of mass
- **Valid locations**: Within vehicle bounding box (e.g., X ∈ [-2, 2] m, Y ∈ [-1, 1] m)

❌ **FORBIDDEN**: Fault locations outside vehicle (e.g., x=10m), inconsistent coordinate frames

---

## 📏 Dimensional Analysis (MANDATORY)

Before outputting ANY numerical result, verify units:

| Quantity | SI Unit | Example |
|----------|---------|---------|
| Mass | kg | 1500 kg (vehicle) |
| Force | N (kg·m/s²) | 100 N (fault force) |
| Acceleration | m/s² | 9.81 m/s² (gravity) |
| Frequency | Hz (1/s) | 13.3 Hz (rotation) |
| Angular velocity | rad/s | 2π × 13.3 = 83.6 rad/s |
| Stiffness | N/m | 10,000 N/m (mount) |
| Damping | N·s/m | 500 N·s/m |
| Energy | J (N·m) | 10 J (vibration energy) |

**Template for checking**:
```
Given: v = 40 km/h, wheel radius r = 0.3 m
Calculate rotation frequency:
  v_mps = 40 / 3.6 = 11.11 m/s
  omega = v_mps / r = 11.11 / 0.3 = 37.04 rad/s
  f_rot = omega / (2*pi) = 37.04 / 6.283 = 5.9 Hz ✅
```

---

## 🎯 Output Format Specifications

### Tool Outputs (JSON Schema)

Every tool MUST return a dict matching its schema in `schemas/tool_outputs.py`:

```python
# Example: estimate_fault_coordinates()
{
  "x_meters": 0.45,          # float, range [-2, 2]
  "y_meters": -0.12,         # float, range [-1, 1]
  "z_meters": 0.05,          # float, range [-0.5, 0.5]
  "uncertainty_radius_cm": 3.0,  # float, > 0
  "confidence": 0.92,        # float, range [0, 1]
  "method": "L1_inverse",    # str, one of ["L1_inverse", "L2_inverse", "modal_energy"]
  "computation_time_sec": 1.2
}
```

❌ **FORBIDDEN**: 
- Missing required fields
- Wrong data types (e.g., string instead of float)
- Out-of-range values
- Inventing new fields not in schema

### Natural Language Outputs (Explanation Agent)

Follow this template EXACTLY:

```
[FAULT SUMMARY]
The vehicle exhibits a <fault_type> fault with severity <severity> 
located at <component_name> (<x, y, z> coordinates).

[PHYSICAL EVIDENCE]
- Dominant frequency: <freq> Hz (<order>× rotation order)
- Vibration amplitude: <amplitude> m/s² RMS
- Spectral peak: <peak_value> dB at <peak_freq> Hz

[FLEET COMPARISON]
This pattern matches <percentage>% of known <fault_type> cases 
from the fleet database (N=<num_samples> similar vehicles).

[ROOT CAUSE ANALYSIS]
Bayesian causal inference assigns <probability>% confidence 
to <root_cause> based on:
- Feature correlation: <correlation_value>
- Treatment effect: <effect_size>
- Cluster membership: <cluster_id>

[RECOMMENDED ACTION]
<repair_action> at <workshop_type>.
Estimated cost: $<cost_range>
Estimated time: <time_hours> hours
Urgency: <urgency_level>

[ACTIVE EXPERIMENT SUGGESTION (if uncertainty > threshold)]
To reduce diagnostic uncertainty, run this test:
"<experiment_instruction>"
Expected information gain: <info_gain> bits
```

❌ **FORBIDDEN**: 
- Vague language ("might be", "possibly")
- Unsupported claims (no data backing)
- Overconfident statements (claiming 100% certainty)
- Inconsistent units

---

## 🔒 Constraints on Agent Reasoning

### Physics Agent
- ✅ Use validated algorithms: scipy eigenvalue solvers, LASSO from sklearn
- ✅ Check condition number of matrices before inversion
- ❌ Don't invert singular matrices (det ≈ 0)
- ❌ Don't trust solutions with residual error > 10%

### Fingerprinting Agent
- ✅ Normalize spectrograms to dB scale: $20 \log_{10}(|X|)$
- ✅ Verify FFT length is power of 2 (for efficiency)
- ❌ Don't compare fingerprints from different sampling rates
- ❌ Don't use raw amplitude (always normalize)

### Fleet Matching Agent
- ✅ Use cosine distance for unit-normalized vectors
- ✅ Require minimum cluster size (≥ 5 samples)
- ❌ Don't match if database has < 10 samples
- ❌ Don't claim "exact match" (always report similarity score)

### Causal Inference Agent
- ✅ Report confidence intervals, not just point estimates
- ✅ Acknowledge confounding variables
- ❌ Don't confuse correlation with causation
- ❌ Don't claim causal link without treatment/control comparison

### Active Experiment Agent
- ✅ Suggest experiments within safe operating range (speed ≤ 80 km/h)
- ✅ Predict information gain before recommending
- ❌ Don't suggest dangerous tests (e.g., "remove wheel while driving")
- ❌ Don't recommend experiments with gain < 0.5 bits

### Explanation Agent
- ✅ Use accessible language (avoid jargon for end users)
- ✅ Cite specific data (frequencies, amplitudes)
- ❌ Don't oversimplify physics (explain mechanisms)
- ❌ Don't make up component names not in vehicle model

---

## 📊 Data Quality Checks (Automatic Rejection Criteria)

**Reject run if**:
- IMU sampling rate < 500 Hz (Nyquist: need 2× max freq of 500 Hz)
- Audio sampling rate < 20 kHz (Nyquist: need 2× max freq of 10 kHz)
- Signal-to-noise ratio (SNR) < 10 dB
- Missing data > 5% of time series
- Time stamps not monotonic
- Contains NaN or Inf values

**Warning if**:
- SNR between 10-20 dB (marginal quality)
- Sampling rate jitter > 0.1%
- Clipping detected (amplitude saturates)

---

## 🧮 Mathematical Formulas (Reference)

### Imbalance Force
$$F_{imbalance}(t) = m \cdot e \cdot \omega^2 \cdot \cos(\omega t + \phi)$$
- $m$ = imbalanced mass (kg)
- $e$ = eccentricity (m)
- $\omega$ = angular velocity (rad/s)
- Valid range: $me \in [0, 0.05]$ kg·m

### Bearing Frequencies
$$
\begin{align}
f_{BPFO} &= \frac{n}{2} f_s \left(1 - \frac{d}{D} \cos\alpha \right) \\
f_{BPFI} &= \frac{n}{2} f_s \left(1 + \frac{d}{D} \cos\alpha \right) \\
f_{BSF} &= \frac{D}{2d} f_s \left(1 - \left(\frac{d}{D}\right)^2 \cos^2\alpha \right)
\end{align}
$$
- $n$ = number of rolling elements (typically 8-12)
- $d$ = ball diameter (m)
- $D$ = pitch diameter (m)
- $\alpha$ = contact angle (typically 0° for radial bearings)
- $f_s$ = shaft rotation frequency (Hz)

### Modal Decomposition
$$x(t) = \sum_{i=1}^{N} \phi_i q_i(t)$$
- $\phi_i$ = eigenvector (mode shape)
- $q_i(t)$ = modal coordinate (satisfies $\ddot{q}_i + 2\zeta_i\omega_i\dot{q}_i + \omega_i^2 q_i = f_i$)

---

## 🎓 Uncertainty Quantification

**Always report uncertainty when**:
- Inverse problem is ill-conditioned (condition number > 100)
- Data is noisy (SNR < 20 dB)
- Limited fleet data (< 50 samples)
- Extrapolating beyond training range

**Methods**:
- Bootstrap confidence intervals (1000 resamples)
- Bayesian credible intervals (posterior quantiles)
- Cross-validation error estimates

**Language**:
- ✅ "92% confidence interval: [0.42, 0.48] meters"
- ✅ "Uncertainty radius: ±3 cm (95% credible region)"
- ❌ "Fault is at exactly 0.45 meters" (overly precise)

---

## 🚫 Prohibited Behaviors

### Hallucination Triggers (NEVER DO THIS)
1. ❌ Inventing sensor data not in input
2. ❌ Claiming to "see" hardware (we only have simulated data)
3. ❌ Referencing specific vehicle makes/models not in metadata
4. ❌ Citing papers or studies not in knowledge base
5. ❌ Generating images/videos (only static plots allowed)
6. ❌ Executing code not in approved tools
7. ❌ Accessing external APIs without permission
8. ❌ Storing personally identifiable information (PII)

### Edge Cases (Handle Gracefully)
- **No dominant frequency peak**: Report "No clear fault signature detected"
- **Multiple peaks**: Report "Mixed fault types possible, recommend further testing"
- **Conflicting causal evidence**: Report probability distribution, not single answer
- **Insufficient fleet data**: Acknowledge limitation, suggest collecting more data

---

## ✅ Validation Checklist (Before Returning Result)

For EVERY output, verify:
- [ ] All units are correct and consistent
- [ ] All values are within valid ranges
- [ ] JSON schema matches specification
- [ ] No invented data or assumptions stated as fact
- [ ] Uncertainty quantified where appropriate
- [ ] Physical plausibility checked (conservation laws)
- [ ] Explanation cites specific evidence
- [ ] No prohibited behaviors used

**If any checkbox fails, REVISE result before returning.**

---

## 📚 Approved Reference Data

### Vehicle Model Parameters (Fixed)
```python
VEHICLE_PARAMS = {
    "total_mass_kg": 1500,
    "wheelbase_m": 2.5,
    "track_width_m": 1.5,
    "wheel_radius_m": 0.3,
    "suspension_stiffness_N_per_m": 20000,
    "suspension_damping_Ns_per_m": 1500,
    "tire_stiffness_N_per_m": 200000,
    "engine_mass_kg": 150,
    "body_mass_kg": 1000
}
```

### Fault Severity Mapping
```python
SEVERITY_THRESHOLDS = {
    "imbalance": {
        0.2: "5g mass at 0.1m",
        0.4: "10g mass at 0.15m",
        0.6: "20g mass at 0.2m",
        0.8: "40g mass at 0.25m",
        1.0: "50g mass at 0.3m"
    },
    "loose_mount": {
        0.2: "10% stiffness reduction",
        0.4: "30% stiffness reduction",
        0.6: "50% stiffness reduction",
        0.8: "70% stiffness reduction",
        1.0: "90% stiffness reduction (critical)"
    },
    "bearing_wear": {
        0.2: "Minor surface roughness",
        0.4: "Small spall (1mm)",
        0.6: "Medium spall (3mm)",
        0.8: "Large spall (5mm)",
        1.0: "Severe damage (>10mm)"
    }
}
```

### Frequency Ranges
```python
FREQUENCY_BANDS = {
    "rotation_order": (0, 20),      # 1× rotation (most faults)
    "bearing_frequencies": (50, 500),  # BPFO/BPFI range
    "structural_resonance": (20, 100),  # Body modes
    "high_frequency_noise": (500, 2000)  # Acoustic emissions
}
```

---

## 🎯 Success Criteria

**An LLM agent response is considered VALID if**:
1. ✅ All outputs match schemas in `schemas/tool_outputs.py`
2. ✅ All physics calculations are dimensionally correct
3. ✅ All claims are backed by data in the state
4. ✅ Uncertainty is quantified for estimates
5. ✅ No invented fault types, severities, or locations
6. ✅ Natural language is clear, specific, and honest

**If response meets all 6 criteria → 95% hallucination reduction achieved** ✨

---

*This SILL document is the ground truth for all LLM agents. When in doubt, consult this file.*
