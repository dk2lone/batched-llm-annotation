# Null and power simulation for the frozen design

Created 2026-09-12T04:13:49.009729+00:00. Total runtime 1517.7 s.

Every scenario uses the exact frozen design: 2048 items, k = 16, 128 calls per assignment, 3 assignments, 16 length strata, and the real assignment generator.

No new experimental model labels exist. Step 0 ran on CPU only. No model was loaded, no GPU was used, and no joint, exact-context, solo, filler, fake-batch, or repeated-sampling output has been produced.

Every rejection rate below is a Monte Carlo estimate. The grid uses 300 trials per scenario, the threshold sweep uses 200, and each trial runs 999 randomization draws. A rate near 0.9 from 200 trials carries a standard error near 0.021.

## 1. Scenario grid

| marginal error rate | true excess covariance | reject at 0.05 | reject at Holm 0.0167 | reject and above floor | mean estimate | bias | interval covers truth |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0 | 0.047 | 0.007 | 0.000 | 0.00001 | +0.00001 | 0.957 |
| 0.05 | 0.002 | 0.993 | 0.993 | 0.470 | 0.00198 | -0.00002 | 0.693 |
| 0.05 | 0.004 | 1.000 | 1.000 | 0.997 | 0.00402 | +0.00002 | 0.520 |
| 0.05 | 0.0075 | 1.000 | 1.000 | 1.000 | 0.00752 | +0.00002 | 0.293 |
| 0.08 | 0.0 | 0.047 | 0.020 | 0.000 | 0.00000 | +0.00000 | 0.937 |
| 0.08 | 0.002 | 0.960 | 0.930 | 0.480 | 0.00200 | +0.00000 | 0.817 |
| 0.08 | 0.004 | 1.000 | 1.000 | 0.990 | 0.00388 | -0.00012 | 0.713 |
| 0.08 | 0.0075 | 1.000 | 1.000 | 1.000 | 0.00734 | -0.00016 | 0.447 |
| 0.113 | 0.0 | 0.077 | 0.037 | 0.000 | 0.00009 | +0.00009 | 0.920 |
| 0.113 | 0.002 | 0.847 | 0.750 | 0.463 | 0.00192 | -0.00008 | 0.907 |
| 0.113 | 0.004 | 1.000 | 1.000 | 0.973 | 0.00389 | -0.00011 | 0.833 |
| 0.113 | 0.0075 | 1.000 | 1.000 | 1.000 | 0.00744 | -0.00006 | 0.657 |
| 0.2 | 0.0 | 0.047 | 0.013 | 0.010 | -0.00001 | -0.00001 | 0.950 |
| 0.2 | 0.002 | 0.567 | 0.380 | 0.380 | 0.00200 | +0.00000 | 0.910 |
| 0.2 | 0.004 | 0.960 | 0.897 | 0.897 | 0.00388 | -0.00012 | 0.890 |
| 0.2 | 0.0075 | 1.000 | 1.000 | 1.000 | 0.00745 | -0.00005 | 0.820 |

## 2. Type-I error under a marginal accuracy shift

marginal shift under the true null: exact arm 3 points more accurate.

Rejection rate at alpha 0.05: 0.033. Mean estimate -0.00001.

A rate near 0.05 shows the centred statistic is not driven by the arms having different marginal error rates.

## 3. Monte Carlo draw count

| draws | Type-I at 0.05 | smallest reachable p | resolution adequate for Holm over 3 cells |
|---:|---:|---:|---|
| 499 | 0.060 | 0.002000 | True |
| 999 | 0.055 | 0.001000 | True |
| 1999 | 0.035 | 0.000500 | True |

Frozen production count: **2000**.

the count study reports Type-I rates at [499, 999, 1999] draws without using those rates to tune the production count. At 2000 draws the smallest reachable p is 0.000500, far below the three-cell Holm threshold of 0.0167.

## 4. Achieved power across the error-rate grid, and the solo-screen threshold

Criterion: detection power: the trial rejects at the three-cell Holm-adjusted alpha when the true excess covariance equals the 0.002 floor.

the joint claim gate also requires the point estimate to reach the floor. At a true effect exactly equal to the floor an unbiased estimator meets it about half the time, so this rate is near 0.5 by construction and is not a power failure.

| marginal error rate | p(1-p) | implied rho at the floor | floor reachable | detection power at Holm 0.0167 | Monte Carlo SE | joint claim gate |
|---:|---:|---:|---|---:|---:|---:|
| 0.001 | 0.00100 | - | False | 0.000 | 0.000 | 0.000 |
| 0.002 | 0.00200 | - | False | 0.000 | 0.000 | 0.000 |
| 0.0025 | 0.00249 | 0.8020 | True | 0.745 | 0.031 | 0.425 |
| 0.005 | 0.00498 | 0.4020 | True | 0.985 | 0.009 | 0.425 |
| 0.01 | 0.00990 | 0.2020 | True | 1.000 | 0.000 | 0.420 |
| 0.015 | 0.01477 | 0.1354 | True | 1.000 | 0.000 | 0.465 |
| 0.02 | 0.01960 | 0.1020 | True | 1.000 | 0.000 | 0.420 |
| 0.03 | 0.02910 | 0.0687 | True | 1.000 | 0.000 | 0.480 |
| 0.04 | 0.03840 | 0.0521 | True | 1.000 | 0.000 | 0.475 |
| 0.05 | 0.04750 | 0.0421 | True | 0.995 | 0.005 | 0.475 |
| 0.06 | 0.05640 | 0.0355 | True | 0.990 | 0.007 | 0.505 |
| 0.08 | 0.07360 | 0.0272 | True | 0.925 | 0.019 | 0.455 |
| 0.1 | 0.09000 | 0.0222 | True | 0.810 | 0.028 | 0.510 |
| 0.113 | 0.10023 | 0.0200 | True | 0.720 | 0.032 | 0.470 |
| 0.15 | 0.12750 | 0.0157 | True | 0.580 | 0.035 | 0.525 |
| 0.2 | 0.16000 | 0.0125 | True | 0.310 | 0.033 | 0.305 |
| 0.3 | 0.21000 | 0.0095 | True | 0.200 | 0.028 | 0.200 |

Frozen threshold: marginal error rate **0.005**, which is **11** solo errors out of 2048 items.

Highest grid error rate that still reaches the target: **0.08**. Grid error rates above that band which miss the target: [0.1, 0.113, 0.15, 0.2, 0.3].

at a fixed absolute floor the implied within-call correlation falls as p(1-p) grows, so detection power falls at high marginal error rates. Any rate above the band is reported as a disclosed power limit of the fixed design, not as a screen failure.

this smallest qualifying prespecified grid value becomes solo_screen.criteria.error_information_threshold. A cell below it is labelled information-poor and stays in the published panel.

## 5. The 0.002 covariance floor

Detectable at the frozen power target somewhere on the grid: **True**. Decision: kept at 0.002.

Detection power at the floor, by marginal error rate:

| marginal error rate | detection power |
|---:|---:|
| 0.05 | 0.993 |
| 0.08 | 0.930 |
| 0.113 | 0.750 |
| 0.2 | 0.380 |

The floor is never lowered and the effect is never redefined. Where p(1-p) is below 0.002 no within-call correlation can reach the floor, so such a cell is information-poor by arithmetic. Where power at the floor is below the target, the design is disclosed as underpowered for the smallest meaningful effect at that marginal error rate.

## 6. Named diagnostics

- `typeI_rate_at_delta_zero`
- `typeI_rate_with_marginal_shift`
- `power_at_floor_0_002`
- `power_at_pilot_like_0_0075`
- `information_threshold_min_error_rate`
- `monte_carlo_resolution`
- `estimator_bias`
- `effect_interval_coverage`

## 7. Tuning boundary

the scenario grid is chosen from the disclosed discovery inputs only. No pilot per-assignment value, no pilot interval bound, and no pilot verdict is used to set a threshold.
