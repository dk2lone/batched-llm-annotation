# X1R4 A8 technical budget release

Budget verdict: **PASS**. This CPU-only forecast inspected timing and completion records, not model predictions or scientific results, and launched no GPU.

The sixteen-cell base forecast is 27.302 A100-hours. After the 1.6x guard, 2.882 prior hours, and the 10-hour reserve, the projected total is 56.565 of 80 A100-hours. Guarded headroom is 23.435 hours.

Released scope: the first 1,000 scheduled A8 attempts for each of the four models; no remaining attempt may start until that model's technical-only measured reforecast passes. Paid execution still requires Daniel's separate authorization.

The forecast uses the completed Qwen/MultiNLI B3 meter as its absolute anchor and sealed synthetic runtime measurements only as relative model-task weights. Production timing after 1,000 attempts replaces this forecast before any further release.
