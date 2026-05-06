# RULE 81: Scientific Research Protocol - Full Detail

## Core Principle
Every architectural or feature decision must be grounded in science.
We do not reinvent the wheel. We read what others have done, we understand why,
then we innovate beyond it.

## Trigger
BEFORE any of these:
- Designing a new algorithm or model architecture
- Choosing between technical approaches
- Starting a new project (section 0-10%)
- Any claim that "our approach is novel"
- Hitting a technical blocker that theory might resolve

## Search Sources (in priority order)
| Source | URL | Best for |
|--------|-----|----------|
| arXiv | arxiv.org | Preprints - fastest, bleeding edge |
| Semantic Scholar | semanticscholar.org | Citations, related work graph |
| Papers With Code | paperswithcode.com | Implementation benchmarks |
| Google Scholar | scholar.google.com | Broad coverage |
| ACM Digital Library | dl.acm.org | CS systems, algorithms |
| IEEE Xplore | ieeexplore.ieee.org | Engineering, signal processing |

## Search Protocol

### Step 1: Terminology scan (15 min)
Search 3-5 keyword variants of the problem.
Example for NeuroDose: "nootropic tracking", "cognitive enhancement monitoring", "pharmacokinetic modeling mobile"
Goal: understand the exact vocabulary the field uses.

### Step 2: Landmark papers (30 min)
Find the 2-3 most cited papers on the topic (seminal work).
These explain the foundations. Read abstract + conclusion + figures.
Note: year, authors, citation count, core contribution.

### Step 3: Recent papers (30 min)
Filter: 2022-2026. Find 3-5 papers showing the current frontier.
Look for: state-of-the-art benchmarks, open problems, failure modes.

### Step 4: Synthesis (15 min)
Write a research summary in kuro-rules/RESEARCH_MEMORY/<project>-<topic>-<date>.md

## Research Summary Format
```markdown
# Research: [Topic] - [Project] - [Date]

## Why this research
[What decision or feature prompted this]

## Landmark papers
1. [Title] ([Year]) - [Authors] - [Core contribution in 1 sentence]
2. ...

## Recent frontier (2022-2026)
1. [Title] ([Year]) - [Key finding relevant to us]
2. ...

## Open problems in the field
- [Problem 1]
- [Problem 2]

## What we learned
[3-5 actionable insights for our project]

## How we apply it
[Concrete changes to our approach based on this research]

## Our potential innovation angle
[What gap exists that our project could fill]
```

## When to Write a Paper
IF any of the following are true:
- We have a novel approach not found in literature
- We have results that beat published benchmarks
- We solved an open problem identified in the research
THEN: write a short paper draft in kuro-rules/RESEARCH_MEMORY/<project>-paper-draft.md
Format: abstract + problem + method + preliminary results + future work

## Domain-Specific Search Strategies

### For AI/Neural Networks (Neural, NeuralDbg, Aquarium)
Keywords: neural network debugging, gradient flow visualization, NAS survey, hyperparameter optimization
Key venues: NeurIPS, ICML, ICLR, CVPR

### For Quantitative Finance (OpenQuant, Console)
Keywords: transformer financial forecasting, regime detection HMM, market microstructure
Key venues: Journal of Financial Economics, Quantitative Finance, arXiv q-fin

### For Bioinformatics/Nootropics (NeuroDose)
Keywords: pharmacokinetic modeling, cognitive biomarkers, nootropic efficacy
Key venues: Nature Neuroscience, Journal of Psychopharmacology, PubMed

### For Blockchain/Crypto (Helium)
Keywords: consensus mechanisms survey, DeFi protocol design, cryptographic primitives
Key venues: IEEE S&P, Financial Cryptography, ACM CCS

### For Algorithm Visualization (Dissect, Bloom DB)
Keywords: probabilistic data structures survey, Bloom filter variants, agent orchestration
Key venues: VLDB, SIGMOD, arXiv cs.DS

### For Physics/Math (AEther, Logical Calculus)
Keywords: Minkowski spacetime visualization, non-Hermitian quantum systems, infinitesimal calculus foundations
Key venues: Physical Review, arXiv math-ph, Journal of Mathematical Physics

## Enforcement
IF feature implementation starts without research summary:
  ACTION: STOP - run Step 1-4 first (90 min total)
  ACTION: Save summary to RESEARCH_MEMORY before first commit
  DO NOT: Treat intuition as equivalent to scientific grounding
  DO NOT: Skip "old" papers - foundational work from 1990s-2010s often contains the insight

IF research reveals our approach is already solved:
  ACTION: Do not rebuild. Use or adapt existing work.
  ACTION: Cite the paper in code comments: # Based on [Author et al., Year]
  EXCEPTION: Rebuild only if existing solutions have license conflicts or performance gaps we can measure.