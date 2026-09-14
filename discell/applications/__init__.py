"""Doc-11: applications of the clean intrinsic state z, one module each.

Every application carries its own pre-registered validation (doc 11) and
its own CLI; nothing here is imported by the model or training code.
Shared data dependencies (nuclear DAPI, nuclear-only counts, the pinned
dissociated reference) live in ``shared`` and are built once, reused by
all. Sequencing (cost-ordered): a4_cycle -> a1_states -> a3_trajectories
-> a6_qc -> a5_transfer -> a2_clones.
"""
