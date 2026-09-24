export const meta = {
  name: 'manuscript-critical-review',
  description: 'Reviewer-style critical pass over the built DISCELL AISTATS manuscript: 7 lenses, adversarial verify, completeness critic, written checklist',
  phases: [
    { title: 'Review', detail: '7 independent reviewer lenses over the built manuscript', model: 'opus' },
    { title: 'Verify', detail: 'skeptic per finding; fix-check on majors', model: 'sonnet' },
    { title: 'Completeness', detail: 'critic asks what was missed; new findings verified', model: 'opus' },
    { title: 'Write', detail: 'dedup, prioritise, write review_<date>.md', model: 'opus' },
  ],
}

const DATE = (args && args.date) || 'undated'
const OUT = `submission_paper/aistats/review_${DATE}.md`
const ROOT = '/home/rmolen/github/DisCell'
const PDFTXT = '/tmp/claude-1000/-home-rmolen-github-DisCell/feef6a80-b1ae-42cb-8556-3b40ad13bdc6/scratchpad/pdf'

const CONTEXT = `
You are part of a critical, reviewer-style audit of a research manuscript. Repo root: ${ROOT}.

MANUSCRIPT (AISTATS 2026 submission "DISCELL"; one paragraph per source line, so file:line is meaningful):
Built files, the ONLY ones in scope:
  submission_paper/aistats/sections/abstract.tex, introduction.tex, method.tex, conclusion.tex, checklist.tex
  submission_paper/aistats/appendix/derivations.tex, rationale.tex, implementation.tex
  (main.tex, macros.tex define structure and macros.)
OUT OF SCOPE: sections/experiments.tex and the appendices data/calibration/synthetic/validation/additional_results (commented out of the build; evaluation not written yet). Do not report on them and do not ask for experiments/results to be added; you MAY say a claim needs evidence later.

AUTHOR CONVENTIONS (a proposed change that violates these is wrong):
  - The article never references code (no file/function/flag/variable/run names).
  - The article currently carries NO empirical numbers (model-defining constants such as widths, grid, tau, prune radius are allowed; measured results are not).
  - Markers: \\needsource (unsourced claim), \\todo{}, \\pending{} (awaits a rerun). Page overflow is tolerated until trimming, but main-text length is a real concern (8 pages).
  - Readers: ML reviewers AND spatial-transcriptomics readers.

ALREADY KNOWN — read these first and DO NOT re-report their items; reference them by ID instead (you may disagree with one, say so):
  - submission_paper/aistats/citation_audit.md  (citation audit, items A1–A11, B, C1–C20, D)
  - docs/paperlog.md  (manuscript edits already made today)
  - docs/todo.md section "7. Paper" (suggestion to move the "Two hops, one pass" paragraph) and item 8.11 (invariance escalation framing: 330/330 runs use the adversary; per-section escalation claim untrue)
  - submission_paper/aistats/revision_2026-09-17.md and questions_and_issues.md (older open items)

FACT-CHECK SOURCES:
  - Code: discell/model/{networks,elbo,train,prepare,metrics,sweep}.py, discell/preprocess/{geometry,xenium}.py, discell/data/loader.py
  - Pinned run configs: data/datasets/*/runs/best*/config.json
  - Cited papers: PDFs in submission_paper/articles/; extracted text in ${PDFTXT}/{DisCOVER,resolVI,SIMVI,mintflow,Celcome}.raw.txt

FACTS ESTABLISHED EARLIER TODAY (usable; re-verify cheaply if you lean on one for a MAJOR finding):
  - Every run on disk uses invariance=adversary (alpha_a=0.3, 6 head steps per model step, head lr 0.002); no reported fit uses the closed-form penalty.
  - Pinned: omega=1, alpha_w=0.1, d_z=20, d_w=6, kappa operating point 0.1, nmi_guard 0.9, second_kl=True, gat_sources=type_only, 500 epochs / patience 40 for references. alpha_z is ~1/mean-count (ovarian 0.0035 vs mean count ~262; lung 0.002 vs ~441; ovary FF 0.00035 vs ~1663).
  - Prior scale sigma_w fixed at 1 (elbo.py ~107). Decoder softmax is over all G panel genes (networks.py log_rho); control probes are dropped at load.
  - The model graph prunes edges at 40 um (model/prepare.py:36); the bundle graph (Delaunay, Voronoi faces clipped to 30 um discs) has no edge with centroid distance >= 60 um. tau = 20 um.
  - Agreement NMI: k-means with k=K on mu_z, <=30k-cell subsample, sklearn default ARITHMETIC normalisation. Checkpoint accepted iff held-out recon improves AND NMI >= 0.9 x running max.
  - Mean counts per cell: GSE core 140, ovarian 262, lung 441, ovary FF 1663.

HARD RULE: you are READ-ONLY. Do not create, modify or delete any file.
`

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'file:line (relative to submission_paper/aistats/), or a section name for structural issues' },
          quote: { type: 'string', description: 'short verbatim quote of the offending text, or "" for structural issues' },
          category: { type: 'string', description: 'e.g. math, code-mismatch, soundness, overclaim, positioning, clarity, structure, length, consistency, notation, domain' },
          severity: { type: 'string', enum: ['major', 'minor', 'nit'] },
          issue: { type: 'string', description: 'what is wrong, stated precisely' },
          why: { type: 'string', description: 'why a reviewer would care / consequence if left' },
          proposed_change: { type: 'string', description: 'concrete fix: replacement wording, an equation correction, or a structural action. Must respect the author conventions.' },
          evidence: { type: 'string', description: 'support, each piece tagged [TEXT] manuscript file:line, [CODE] path:line, [PDF] paper+section, [DATA], or [MEM] (own knowledge, unverified)' },
          relates_to: { type: 'string', description: 'existing IDs this touches (e.g. A2, C10, todo 8.11, paperlog) or ""' },
        },
        required: ['location', 'quote', 'category', 'severity', 'issue', 'why', 'proposed_change', 'evidence', 'relates_to'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['confirmed', 'partly', 'refuted'] },
    reason: { type: 'string', description: 'what you checked and what you found, with file:line / PDF section' },
    corrected_change: { type: 'string', description: 'if partly: the corrected issue/fix; else ""' },
  },
  required: ['verdict', 'reason', 'corrected_change'],
}

const LENSES = [
  { key: 'math', title: 'Mathematical correctness and notation',
    brief: `Check every equation, derivation and formal argument: the bound in the objective section and app:bound (is the KL placement right, is the one-sample estimate claim right), the intrinsic path and the (1+omega) argument, the renormalisation equation for isolated cells, the Gaussian conditional MI and its properties (scale invariance, dropping a simplex column), the adversary objective and its sign conventions, the probe definition and the claim MSE equals the Gaussian cross-entropy up to constants, the gauge/identifiability arguments (rotation, softmax constant, per-type translation — are they actually correct as stated?), the feedback-amplification heuristic, the counterfactual/abduction statement. Also notation: symbols defined before use, one symbol per concept, dimensions consistent (e.g. d_v = K-1+12), macros used consistently.` },
  { key: 'code', title: 'Claims versus the implementation',
    brief: `Every factual statement about what the model or pipeline does must match the code and pinned configs. Go through tab:learned, tab:architecture, tab:deviations, tab:deviations2 row by row and every procedural sentence in method.tex (widths, heads, activation, image embedding width, probe details, tiles, held-out fraction, evaluation cadence, stopping rule, EMA/shrinkage/support floor/jitter, probability floor, isolated-cell handling, what the encoders receive, what is under stop-gradient, how w_j of neighbours is drawn, adversary inputs and targets, image-niche construction). Report every mismatch with [CODE] path:line evidence, and every description that is true of a default but not of the pinned runs.` },
  { key: 'soundness', title: 'Methodological soundness (as a critical AISTATS reviewer)',
    brief: `Read as a hostile but fair ML reviewer. Where does the paper claim more than it demonstrates or argues? Is the kappa-sweep-as-sensitivity-analysis framing sound (what does "stable across the grid" license)? Is the invariance target well-posed when t itself is derived from the contaminated counts? Is the probe protocol a valid test (what can it miss)? Is the stopping rule / seed-selection procedure open to selection bias ("one member selected")? Is the objective a coherent estimation target? Are the identifiability statements correct and sufficient? Is the z/w allocation argument convincing? What are the strongest objections a reviewer would raise to the core premise, and does the text pre-empt them? What is the actual novelty relative to SIMVI, DisCoVR, resolVI, NCEM? Propose text changes that fix or honestly concede each point.` },
  { key: 'positioning', title: 'Literature and positioning',
    brief: `Building on (not repeating) citation_audit.md: are the novelty claims fair? Is related work balanced and complete for an AISTATS audience (disentangled VAEs, invariant representation learning, spatial omics models, contamination correction, sensitivity analysis / partial identification)? Are any characterisations of other methods unfair or wrong (you can read the PDFs' extracted text)? Are there claims presented as new that are standard? Where should a sentence explicitly differentiate DISCELL from a named method? Give concrete replacement wording.` },
  { key: 'clarity', title: 'Clarity, structure and length',
    brief: `AISTATS main text is 8 pages. Identify redundancy (method text vs tab:deviations, repeated arguments, the same point made in intro and method and appendix), paragraphs that belong in an appendix, missing signposting, terms used before definition, overly long sentences, places where a figure/diagram of the model would do more than prose, the empty conclusion, the abstract (does it state the problem, method, and what is delivered?), the contributions list (are they claims the paper supports?). Prioritise by how much each change improves a first read. Give concrete moves and rewordings.` },
  { key: 'consistency', title: 'Internal consistency',
    brief: `Cross-check across all built files: constants (tau, 30 um clip, 40 um prune, kappa grid, d_z, d_w, widths, 0.9 guard, EMA 0.05), terminology (leak fraction, foreign influx, clean composition, response, context, niche, agreement), claims in abstract/introduction/contributions versus what the body actually does, cross-references (do \\cref targets exist and point to the section that actually contains the claim — check labels in the .tex files), \\todo/\\pending/\\needsource inventory (which block submission), and the checklist answers (is each [Yes] actually true of the current text? e.g. complexity analysis, complete proofs while a heuristic is labelled as such, asset citations).` },
  { key: 'domain', title: 'Domain realism (spatial transcriptomics / Xenium)',
    brief: `Read as a spatial-transcriptomics methods reviewer. Are the measurement assumptions realistic and stated honestly: section-wide kappa, leaked rate scaling with the receiving cell, Poisson closure, no ambient term, Voronoi-face contact proxy, the image model on 4 Xenium morphology channels with an ego-masked disc, labels derived from contaminated counts, isolated cells, cell-cycle claims, subclonal structure claims, segmentation (nuclear expansion)? What would a biologist object to? Where is biological language imprecise or overclaimed? Propose concrete wording or caveats.` },
]

function reviewPrompt(l) {
  return `${CONTEXT}
YOUR LENS: ${l.title}.
${l.brief}

Read every in-scope file in full before reporting. Report every genuine issue under your lens (no cap), ordered by importance. For each: exact location, short verbatim quote, what is wrong, why it matters, a CONCRETE proposed change (replacement text where possible), and evidence with provenance tags. Severity: major = a reviewer would count it against acceptance or it is factually wrong; minor = should be fixed before submission; nit = polish. Do not pad: a precise short list beats a long vague one. Do not re-report items already in citation_audit.md, paperlog.md or todo 8.11 / section 7 — reference them.`
}

function verifyPrompt(f, lensTitle) {
  return `${CONTEXT}
You are a skeptical second reviewer. Another reviewer (lens: ${lensTitle}) reported the finding below. Try hard to REFUTE it: open the manuscript at the location, read the surrounding paragraph, and check the cited code/PDF/data evidence yourself. Refute if the issue is not real, is already handled elsewhere in the built text, duplicates an item in citation_audit.md / paperlog.md / todo 8.11, or rests on a misreading. Say 'partly' if the issue is real but overstated, mis-located, or the proposed change is wrong — then give the corrected version. Say 'confirmed' only if you checked it and it holds. Default to 'refuted' if you cannot find support.

FINDING:
${JSON.stringify(f, null, 2)}`
}

function fixCheckPrompt(f) {
  return `${CONTEXT}
A MAJOR finding below has been raised about the manuscript. Your job is narrower: judge the PROPOSED CHANGE. Is it correct (would it make the text true)? Is it sufficient? Does it introduce a new error, violate the author conventions (no code references, no empirical numbers), or conflict with other parts of the built text? Read the relevant files. Verdict 'confirmed' = the change is right as proposed; 'partly' = the issue stands but the change needs correcting (give the corrected change); 'refuted' = the issue itself does not hold.

FINDING:
${JSON.stringify(f, null, 2)}`
}

const stats = {}

async function verifyOne(f, lens) {
  const phaseName = 'Verify'
  const checks = [() => agent(verifyPrompt(f, lens.title), { label: `refute:${lens.key}:${f.location}`, phase: phaseName, schema: VERDICT, model: 'sonnet' })]
  if (f.severity === 'major') {
    checks.push(() => agent(fixCheckPrompt(f), { label: `fixcheck:${lens.key}:${f.location}`, phase: phaseName, schema: VERDICT, model: 'sonnet' }))
  }
  const vs = (await parallel(checks)).filter(Boolean)
  const refute = vs[0]
  if (!refute) return { ...f, lens: lens.key, verdict: 'unverified', verify_notes: 'verifier failed', corrected_change: '' }
  if (refute.verdict === 'refuted') return null
  const fix = vs[1]
  let verdict = refute.verdict
  let corrected = refute.corrected_change || ''
  let notes = `refute-check: ${refute.verdict} — ${refute.reason}`
  if (fix) {
    notes += ` | fix-check: ${fix.verdict} — ${fix.reason}`
    if (fix.verdict === 'refuted') verdict = 'contested'
    else if (fix.verdict === 'partly') { verdict = verdict === 'confirmed' ? 'partly' : verdict; corrected = [corrected, fix.corrected_change].filter(Boolean).join(' || ') }
  }
  return { ...f, lens: lens.key, verdict, verify_notes: notes, corrected_change: corrected }
}

phase('Review')
log(`Reviewing the built manuscript through ${LENSES.length} lenses; output will be ${OUT}`)

const perLens = await pipeline(
  LENSES,
  l => agent(reviewPrompt(l), { label: `review:${l.key}`, phase: 'Review', schema: FINDINGS, model: 'opus', effort: 'high' }),
  async (res, l) => {
    const fs = (res && res.findings) || []
    stats[l.key] = { raised: fs.length }
    log(`${l.key}: ${fs.length} findings raised, verifying`)
    const out = (await parallel(fs.map(f => () => verifyOne(f, l)))).filter(Boolean)
    stats[l.key].kept = out.length
    stats[l.key].refuted = fs.length - out.length
    log(`${l.key}: ${out.length} kept, ${fs.length - out.length} refuted and dropped`)
    return out
  },
)

let all = perLens.filter(Boolean).flat()

phase('Completeness')
const CRITIC_LENS = { key: 'critic', title: 'Completeness critic' }
for (let round = 1; round <= 2; round++) {
  const condensed = all.map((f, i) => `${i + 1}. [${f.severity}] ${f.location} — ${f.issue.slice(0, 160)}`).join('\n')
  const crit = await agent(`${CONTEXT}
You are the completeness critic for a reviewer-style audit. Below is the condensed list of verified findings so far (round ${round}). Read the in-scope manuscript files in full and ask: what did the reviewers MISS? Consider paragraphs nobody commented on, equations nobody checked, tables nobody cross-checked, the abstract, the checklist, the appendices, and cross-cutting issues (the paper's overall argument, whether the contributions are supported, what the single most damaging reviewer objection is). Report ONLY genuinely new issues not already covered below or in citation_audit.md / paperlog.md / todo 8.11. If nothing substantive is missing, return an empty list.

FINDINGS SO FAR:
${condensed}`, { label: `critic:round${round}`, phase: 'Completeness', schema: FINDINGS, model: 'opus', effort: 'high' })
  const fresh = (crit && crit.findings) || []
  log(`critic round ${round}: ${fresh.length} new findings raised`)
  if (!fresh.length) break
  const kept = (await parallel(fresh.map(f => () => verifyOne(f, CRITIC_LENS)))).filter(Boolean)
  stats[`critic${round}`] = { raised: fresh.length, kept: kept.length, refuted: fresh.length - kept.length }
  log(`critic round ${round}: ${kept.length} kept, ${fresh.length - kept.length} refuted`)
  all = all.concat(kept)
  if (!kept.filter(f => f.severity !== 'nit').length) break
}

phase('Write')
const summaryStats = Object.entries(stats).map(([k, v]) => `${k}: raised ${v.raised}, kept ${v.kept}, refuted ${v.refuted}`).join('; ')
log(`Writing ${all.length} verified findings (${summaryStats})`)

const written = await agent(`${CONTEXT}
EXCEPTION TO THE READ-ONLY RULE: you must create exactly ONE file, ${ROOT}/${OUT}, and modify nothing else.

You are writing the final reviewer-style change list for the author. Input: ${all.length} findings that survived adversarial verification (JSON below). Each carries a verdict: confirmed, partly (use corrected_change where given — it supersedes proposed_change), contested (the issue was confirmed but a second checker disputed the fix — say so), or unverified.

Do this:
1. Deduplicate: merge findings that describe the same problem from different lenses (keep the best evidence and the most precise fix; note which lenses raised it).
2. Where a finding touches an existing item (citation_audit.md A/B/C/D IDs, todo 8.11, todo section 7 suggestion, paperlog entries), reference that ID instead of repeating it; if a finding DISAGREES with an existing item, keep it and say so explicitly.
3. Spot-check anything that looks doubtful by opening the file at the cited line; drop what does not hold and list what you dropped at the end.
4. Write the file in the same style as submission_paper/aistats/citation_audit.md (read it first): a short header (scope, date ${DATE}, how to use, provenance tags), then
   - "Reviewer summary": the 5–10 concerns a real AISTATS reviewer would lead with, each one or two sentences, linking to item IDs.
   - Section "R. Major" then "S. Minor" then "T. Nits", items numbered R1.., S1.., T1.., each as a markdown checkbox with: location (file:line), a short quote, **Issue**, **Change** (concrete replacement text or action), **Why**, **Evidence** (with provenance tags), **Status** (verdict; lenses that raised it), **Relates to**.
   - Within each severity, group by manuscript section in reading order (abstract, intro, 2.1 ... 2.8, conclusion, checklist, appendices).
   - A closing "Dropped during write-up" list and a one-line tally: ${summaryStats}.
5. Respect the author conventions in every proposed change (no code references in article text, no empirical numbers in article text — numbers may appear in Evidence/Why, which is for the author only).

Return a short plain-text summary: path written, counts per severity, and the top 5 items.

FINDINGS JSON:
${JSON.stringify(all)}`, { label: 'write-review', phase: 'Write', model: 'opus', effort: 'high' })

return { output: OUT, stats, n_findings: all.length, writer_summary: written }
