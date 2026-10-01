Here's a reading order in five blocks. Earlier reading settles the story, and later reading is then judged against it. The flag counts show where most of the cleanup is.

### Block A: the story (settle this first)
| # | Section | Flags | What to decide |
|---|---|---|---|
| **R1** | Abstract | 1 | Is this the paper's claim, in this order? Everything else follows from it. |
| **R2** | Introduction: opening and **Contributions** | 7 | Which contributions stay (a flag proposes folding (iii) into (ii)). These set the red thread. |
| **R3** | Conclusion, including **Limitations** | 4 | Does it close the same story? Limitations is where the honest negatives live: the per-cell channel, the planted gap, the baselines at their defaults. |

### Block B: the evidence (it decides what the method section must carry)
| # | Section | What to decide |
|---|---|---|
| **R4** | §3.6 Comparison with Other Models, fig:tradeoff and fig:separation | The strongest result. Is the framing right, and fair to resolVI, MintFlow and Cellina? |
| **R5** | §3.5 The Leakage Sweep and **Breakdown points** | The paper's central rule. The κ\* table updates tonight. |
| **R6** | §3.3 What the Intrinsic State Carries | Niche residual, cycle, subtypes. |
| **R7** | §3.4 What the Response Carries | Largely flagged TRIM, and it repeats the response appendix. Decide how much stays in the main text. |
| **R8** | §3.2 Model Quality, and §3.1 Recovery on Simulated Sections | Short. Check the tone. |
| **R9** | §3.0 Setup (sections, labels, configuration, baselines) | Candidates to condense into one paragraph and move the rest out. |

### Block C: the method (33 flags; trim to what Block B needs)
| # | Section | What to decide |
|---|---|---|
| **R10** | §2.2 Generative Model and its five "Why …" paragraphs | Keep the model, move most of the justifications to the rationale appendix. |
| **R11** | §2.4 Objective and **Scaling** | The ω/intrinsic-path claim, which ω = 0 tests tomorrow, and the α_w disclosure. |
| **R12** | §2.5 Conditional Invariance: penalty, escalation, probe | Flagged: move the closed-form penalty to the appendix. |
| **R13** | §2.8 What Is Identified, and the Leakage Sweep | Flagged: move the gauge proposition to the appendix. Definition 1 stays. |
| **R14** | §2.3 Inference Networks, §2.1 Setting | Largely fine; small trims. |
| **R15** | §2.6 Batching, §2.7 Selection | Keep the held-out-neighbour disclosure and move the cost arithmetic out. |

### Block D: the appendix the main text leans on
| # | Section | What to decide |
|---|---|---|
| **R16** | Synthetic Recovery, including the Planted Per-Cell Response | The rule-based negative result. Check that it reads as scoped, not as a failure. |
| **R17** | Planted Worlds (the gap result changed today and is being rewritten) | Read it after the rewrite lands. |
| **R18** | Readouts, and The Response in Detail | Definitions a reviewer checks. Merge candidates with §3.4. |
| **R19** | Sensitivity, The Leakage Sweep in Full, Counterfactual with Cellina | Mostly tables; skim. |

### Block E: the rest (skim; flag-driven)
| # | Section | Note |
|---|---|---|
| **R20** | Design Rationale (9 flags) | Obsolete development history lives here. Cut candidates. |
| **R21** | Derivations | Keep; small fixes are done. |
| **R22** | Implementation, and Experimental Details (sections, lineage labels, baselines, licences, costs) | Reviewer reference material; keep and condense. |

**How I'd suggest we work:**
1. **You read in order and decide per flag.** Mark each one in the `.tex`: change `\flag{MOVE}{…}` to `\flag{OK}{…}`, `\flag{NO}{…}` or `\flag{EDIT}{your note}`. That keeps every decision visible in the PDF. Short comments in chat work too.
2. **After each block, we act on it.** The writer rewrites the prose, and I move blocks, check every number against its source and rebuild. Do block A before any trimming: if the story shifts, the right cuts in B–D shift with it.
3. **Block B before block C.** The method should explain only what the results use, so trim the method once you know which results stay in the main text.
4. **Leave block D until after tomorrow's results** (ω = 0, the κ\* transport claims, the planted gap rewrite), since they change D's content.

Shall I put the R-numbers into the PDF as small markers at each section start, in the same style as the flags, so you can follow the order while reading? I'd do it once the planted-worlds rewrite finishes, so the two don't collide in the same file.