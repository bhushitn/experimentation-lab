# Architecture Decision Record

Status: accepted at project start. Each decision states the choice, the evidence, and the rejected alternative. Revisit only if a later phase surfaces contradicting evidence.

## ADR-1: Pathology list and priority

**Decision.** Six pathologies, built in this order:

1. Peeking / sequential testing
2. Multiple comparisons
3. Contamination / non-compliance (ITT vs per-protocol)
4. Network interference (SUTVA violations, cluster vs user randomization)
5. Heterogeneous treatment effects and subgroup fishing
6. Novelty / primacy effects (time-varying treatment effects)

**Evidence.** Order is by (a) frequency in real experimentation practice and interview questions, and (b) how clean a headline number each produces for the README results table. Peeking and multiple comparisons yield the sharpest quantified claims (measured false positive rate vs nominal alpha across N simulated runs) and share simulation machinery (many replications of a clean experiment), so they come first and de-risk the engine. Contamination and interference need the population and graph layers respectively, so they follow. HTE and novelty reuse everything built before them.

**Rejected alternative.** A longer list including Simpson's paradox, survivorship bias in metric definitions, and sample ratio mismatch. Cut for scope: six pathologies each verified against ground truth beats nine where three are shallow. Sample ratio mismatch is the most tempting cut candidate to restore later; it is cheap once assignment exists. Noted as possible future work, not committed.

## ADR-2: Statistical methods and citations

**Decision.** The toolbox in `src/lab/stats/`:

| Method | Validation reference | Citation |
|---|---|---|
| Welch two-sample t-test | `scipy.stats.ttest_ind(equal_var=False)` | scipy docs (validation target, not origin) |
| Chi-square / two-proportion z-test | `scipy.stats.chi2_contingency`, `statsmodels.stats.proportion` | scipy/statsmodels docs |
| Mann-Whitney U | `scipy.stats.mannwhitneyu` | scipy docs |
| CUPED | own implementation; variance reduction measured on simulated data | Deng, Xu, Kohavi, Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data", WSDM '13, pp. 123-132. DOI: [10.1145/2433396.2433413](https://dl.acm.org/doi/10.1145/2433396.2433413) |
| Benjamini-Hochberg FDR | `statsmodels.stats.multitest.multipletests(method="fdr_bh")` | Benjamini, Hochberg, "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing", JRSS-B 57(1), 1995, pp. 289-300. DOI: [10.1111/j.2517-6161.1995.tb02031.x](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x) |
| Group-sequential boundary via alpha spending (O'Brien-Fleming-type spending function) | published O'Brien-Fleming boundary tables and R `ldbounds` reference values, hard-coded in tests | Lan, DeMets, "Discrete sequential boundaries for clinical trials", Biometrika 70(3), 1983, pp. 659-663. [Publisher page](https://academic.oup.com/biomet/article/70/3/659/247777) |
| Bootstrap percentile CI | `scipy.stats.bootstrap` | scipy docs |
| Power / MDE calculators | `statsmodels.stats.power` | statsmodels docs |

All three named-origin citations (CUPED, Benjamini-Hochberg, Lan-DeMets) verified against publisher pages on 2026-07-11.

**Why alpha spending for the sequential method.** Lan-DeMets spending does not require the number of looks to be fixed in advance, which matches the pathology being simulated: an experimenter who peeks daily for an unknown number of days. O'Brien-Fleming-type spending is the industry default because it conserves alpha early, when estimates are noisy.

**Rejected alternative.** Always-valid inference / mSPRT (the Optimizely-style approach). More fashionable, but there is no widely available reference implementation to validate against, and this repo's rule is that every statistical claim is checked against a reference or ground truth. Alpha spending validates against published boundary tables. mSPRT noted in the explainer as further reading, not implemented.

## ADR-3: Layer 2 is a single agent

**Decision.** One agent: a single Anthropic API call sequence with structured output validated by Pydantic schemas, plus deterministic tool calls into Layer 1 for any number that appears in the memo.

**Evidence.** The task is one-pass: read a structured experiment design, match it against a fixed checklist of validity threats, cite the Layer 1 simulation that quantifies each threat, emit a memo. There is no parallelism to exploit, no role separation that survives scrutiny, and no state to coordinate. Every quantitative claim in the memo comes from executing Layer 1 code, never from LLM arithmetic; the agent's job is classification and prose, which one model call does.

**Rejected alternative.** A multi-agent graph (planner, reviewer, critic). That is resume-driven design here: it would add latency, cost, and failure modes to a task with no decomposition benefit, purely to look sophisticated. The prior two portfolio repos already demonstrate multi-agent orchestration; this layer's job is to show the pattern transfers, not to be the star.

## ADR-4: Site stack

**Decision.** Plain HTML/CSS/JS with D3 for visuals, scroll-driven steps via the native `IntersectionObserver` API, no build step. Published with GitHub Pages "deploy from a branch" from `/docs` on `main`, with a `.nojekyll` file so assets are served as-is.

**Evidence from the MLU-Explain study.** MLU-Explain ([source repo](https://github.com/aws-samples/aws-mlu-explain)) is Svelte + D3, one directory per article, each with its own npm build producing static assets. The parts of its design language this project borrows are pedagogical, not tooling: one concept per page, generous whitespace, the visual is the argument, narrative advances on scroll. Svelte earns its keep there through component reuse across 16+ articles by multiple authors. At 6-8 pages by one author, a build pipeline buys nothing and costs: CI must build the site, and the chain from committed simulation code to on-page figure gets one compilation step harder to audit. GitHub's own docs recommend branch deployment when files are already browser-ready ([publishing source docs](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)).

**Rejected alternatives.** Svelte (build step unjustified at this scale, see above). A scrollytelling library such as Scrollama: it wraps `IntersectionObserver`, which covers the needed behavior in a few dozen lines; the dependency is adopted only if resize/mobile edge cases prove painful in the first explainer, and that trigger is recorded here so the decision is auditable.

## ADR-5: Conventions

- Docstrings: NumPy style, because the library's audience reads scipy/statsmodels docs and the toolbox mirrors their conventions.
- Python packaging: `pyproject.toml` with hatchling; ruff for lint, mypy for types, pytest for tests.
- License: MIT proposed (permissive, maximizes reuse of an educational artifact; the repo owner attaches the final license).
- No emojis and no filler vocabulary (leverage, seamless, delve, and similar), anywhere. Enforced in the self-review phase with a wordlist grep.

## ADR-6: One running fictional product for all narrative framing

**Decision.** Every explainer and example memo is set in the same clearly
fictional product, "a short-video app", never named after a real company, with
a synthetic-data disclosure on every page. Simulation parameters are chosen to
match each vignette (e.g. a 10% signup rate for the conversion scenarios, a
zero-inflated lognormal for creator payouts).

**Evidence.** A coherent world makes six explainers read as one course instead
of six disconnected stat lectures, and concrete stakes ("ship the feed layout?")
are what the target interviews test. Honest synthetic framing is defensible
under adversarial probing; a case study dressed up as real company experience
is disqualifying the moment it is questioned.

**Rejected alternative.** Naming a real product (or implying the author ran
these experiments at an employer). Also rejected: no narrative at all, which
reads as a textbook and loses the recommendation-memo register the repo exists
to demonstrate.
