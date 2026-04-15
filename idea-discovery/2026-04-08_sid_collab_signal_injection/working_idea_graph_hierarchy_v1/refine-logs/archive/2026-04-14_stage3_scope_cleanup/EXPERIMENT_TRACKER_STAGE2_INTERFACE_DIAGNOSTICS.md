# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R301 | M1 | prefix stability / SID rearrangement audit | `current v2` vs `R202a` / `R202b-r075` / `R205` | Industrial | changed `l1/l2/full`, prefix overlap | MUST | COMPLETED | `R202a` shows near-full SID remapping but still keeps `41%` l1-pair and `61%` l2-pair retention |
| R302 | M2 | code polysemy / semantic consistency | `current v2` vs `R202a` (optionally strongest original) | Industrial | token semantic spread, prefix-conditioned drift | MUST | COMPLETED | `current v2` and `R202a` are nearly identical on token semantic spread / prefix drift |
| R303 | M3 | downstream learnability attribution | `current v2_on_p05 SFT` vs `R208` | Industrial test | SID change vs improved/worsened | MUST | COMPLETED | improved examples are harder and benefit from stronger l2 rewrite; worsened examples show lower l1 routing stability |
| R304 | M4 | SID learnability probe | `current v2` vs `R202a` | Industrial | level-wise predictability | NICE | COMPLETED | `R202a` helps `a`-level predictability slightly but hurts `b|a` and `c|ab`, especially on hard examples |
