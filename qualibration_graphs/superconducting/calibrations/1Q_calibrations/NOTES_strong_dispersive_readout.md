# Readout Issues in the Strong Dispersive Regime (Alice cavity)

## The Problem

When doing **ge spectroscopy at Fock |n⟩** (node `28d_fNgN1_qubit_ge_spectroscopy`), the measured
baseline state population is ~0.5 instead of ~0, even though the Fock preparation is working
correctly (qubit is in |g⟩ after the sideband π-pulse).

**Symptom:** baseline ≈ 0.5, small contrast spectroscopy peak on top of it.

## Root Cause

The chain of effects is:

```
Alice photon (n=1)
  → qubit ge frequency shifts by χ_Alice (e.g. −282 MHz for alice in DR3-Run011)
  → qubit-readout resonator detuning Δ_qr changes by −282 MHz
  → dispersive shift χ_qr = g²_qr / Δ_qr changes significantly
  → readout resonator IQ trajectory shifts
  → |g, 1_Alice⟩ IQ point lands near the discrimination threshold
  → state reads ~0.5 instead of ~0
```

This is the **strong dispersive regime**: χ_Alice >> κ_Alice (282 MHz >> typical linewidth).
In this regime the readout is photon-number-resolved for Alice, even though the readout
resonator is nominally coupled to the qubit and not to Alice directly.

### Diagnostic that confirmed this

Adding a back-swap (`EF_x180`) after the Fock prep produced **no change** in the baseline
(0.51 → 0.49). Since EF_x180 converts |f⟩ → |e⟩ and leaves |g⟩ unchanged, the fact that
the baseline did not jump toward ~0.94 proved the qubit WAS in |g⟩ after the prep. The
issue is purely in the readout threshold, not in the Fock state preparation quality.

### Why node 28b (time Rabi) does not show this problem

In `28b_fNgN1_time_rabi`, the back-swap EF_x180 adds ~100 ns between sideband completion
and readout. During this window, a fraction of Alice photons decay. With the readout
happening with Alice partially in vacuum, the IQ point drifts back toward the
zero-photon blob and the threshold works correctly, giving a clean ~0.19 readout floor.
This is **not** a sign that the readout is immune to the Alice photon — it is an artifact
of the timing.

## What Other Labs Do

### 1. Map Alice out before readout (most common in sideband/bosonic qubit labs)
After the sequence, reverse the Fock prep to empty Alice before reading the qubit:
```
|g, 1_Alice⟩  --[f0g1 π]-->  |f, 0_Alice⟩  --[π_ef]-->  |e, 0_Alice⟩  → reads HIGH
|e, 1_Alice⟩  → needs extra handling (can use a selective π on ge@Fock-1 first)
```
This ensures the readout always happens with Alice in vacuum, keeping the threshold valid.
Works cleanly for ge saturation spectroscopy where the qubit ends up in |g⟩ or |e⟩.

### 2. Photon-number-resolved threshold calibration
Calibrate and store a separate IQ threshold for each relevant Alice photon number
(0, 1, 2, …). Apply the correct threshold depending on how many photons Alice is
expected to have at readout time. More infrastructure required.

### 3. Use raw IQ value instead of binary state discrimination
For calibrations that only need the **peak position** (e.g. chi calibration in 28d),
binary state discrimination is not required. Fit the raw I-quadrature vs frequency
directly. The shifted baseline does not affect the fit of the resonance frequency.

## Practical Note for Node 28d (chi calibration)

The chi calibration in `28d` only needs the peak position. The peak IS visible and
fit-able even with the 0.5 baseline — the node correctly extracts `chi_focka` from
the data. The shifted baseline is cosmetically bad but does not break the calibration.

For higher Fock levels (n ≥ 2), the shift grows by n × χ_Alice, making the baseline
progressively worse. At that point one of the solutions above becomes necessary.

## Parameters (DR3-Run011, q1-alice)
- `chi_focka (f0g1)` = −282.8 MHz
- Sideband RF frequency = 3302.35 MHz
- Readout threshold calibrated in vacuum (Alice in |0⟩)
