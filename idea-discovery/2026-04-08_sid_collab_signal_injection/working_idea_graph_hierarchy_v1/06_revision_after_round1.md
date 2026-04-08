# Revision After Round 1

## Main revision

The main idea is upgraded from:

- `level-wise graph feature fusion`

to:

- `level-wise graph-regularized hierarchical quantization`

## Updated method core

At SID level `l`, the model no longer only consumes a graph-derived representation.  
It also learns a graph mixture over the multiplex graph bank and uses that mixture to regularize code learning at that level.

That makes the graph contribution structural rather than cosmetic.

## Mid-scale decision

This round does not freeze a single universal mid-scale operator yet.  
Instead it treats Level 2 graph construction as a small, deliberate design choice with 2-3 concrete options:

- community graph
- band-pass spectral residual
- diffusion residual

The final paper version should keep one in the main method and move the others to appendix or pilot analysis.
