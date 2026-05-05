#import "template.typ": *

#show: project.with(
  title: "Breaking the Tabular Ceiling: Heterogeneous Graph-Aware Multimodal Fusion for M&A Synergy Prediction",
  author: "Hard Joshi", // 
  student_id: "2512658", // 
  degree: "Data Science and Artificial Intelligence", // 
  supervisor: "Arish Siddiqui", // 
  date: datetime(year: 2026, month: 05, day: 08), // 
  
  abstract: [
   #set par(justify: true, leading: 0.65em)
    
   Mergers and acquisitions destroy shareholder value in 70–90% of cases, yet existing predictive models consistently fail to identify value-destroying deals before announcement. This study argues that the root cause is architectural rather than computational: every prior generation of quantitative M&A model — from logistic regression to transformer-based NLP — treats each firm as an isolated data point, blind to the supply-chain topology, competitive network structure, and section-level semantic signals that collectively determine whether a proposed combination will create or destroy value. This dissertation addresses that gap by constructing, evaluating, and interpreting a heterogeneous multimodal architecture that encodes all three information dimensions simultaneously, and by formally testing whether each dimension carries predictive signal that is irreducible from the others.
   
   A tri-modal late-fusion model (HeteroGraphSAGE) is built and evaluated on a universe of completed US domestic M&A transactions announced between 2000 and 2023, comprising approximately 3,000 deals with full multimodal coverage, sourced from LSEG Refinitiv, SEC EDGAR, and Bloomberg SPLC. Three feature blocks are fused via late concatenation: 56 financial ratio features (Block A), section-conditioned FinBERT embeddings of Management Discussion & Analysis and Risk Factor disclosures extracted separately from pre-announcement 10-K filings (Block B), and two-hop heterogeneous GraphSAGE embeddings derived from firm-level supply-chain networks (Block C). A dual-evaluation framework is applied: a classifier pipeline that predicts binary Cumulative Abnormal Return (CAR) direction and a regression pipeline that predicts CAR magnitude. Both pipelines are evaluated under five-fold stratified cross-validation with strict temporal splits and an 11-day event-window embargo to prevent forward-looking leakage. Three formal hypotheses are tested — H1 (Topological Alpha), H2 (Semantic Divergence), and H3 (Topological Arbitrage) — using paired t-tests, OLS regression, and Levene's variance test respectively, with Bonferroni correction applied across all three tests.
   
   All three hypotheses are supported. The full multimodal model achieves AUC-ROC = 0.5655, a statistically meaningful +0.0247 lift over the financial-only baseline (0.5408), confirming that supply-chain topology encodes directional signal that tabular models cannot recover (H1). OLS estimation on a semantic-divergence subsample of 1,140 deals recovers opposite-signed coefficients for MD&A similarity ($beta = +0.0044$) and Risk Factor similarity ($beta = -0.0080$), confirming that section-level semantic divergence is economically directional when the two sections are modelled separately rather than pooled (H2). Levene's test across betweenness-centrality quantile groups yields $F = 7.07$ ($p = 0.0079$), confirming that structurally central acquirers experience statistically compressed announcement-return variance (H3). The study additionally establishes that adding undifferentiated FinBERT text to the financial baseline reduces AUC by $-0.012$, demonstrating that naive NLP actively destroys predictive value when filing-section semantics are conflated — a methodological finding with direct implications for subsequent M&A NLP research. Continuous CAR magnitude remains intractable to regression across all configurations ($R^2 < 0$), precisely locating the boundary between tractable and structurally difficult M&A prediction sub-problems. All empirical results are rendered dynamically through the Deal Intelligence Terminal, an interactive research artefact ensuring full reproducibility.
  ],
  
  acknowledgments: [
   #set par(justify: true, leading: 0.65em)

    This project would not have been possible without the support of several people who were generous with their time, guidance, and patience at various stages of a process that was significantly more ambitious than it looked on paper.
    
    My deepest thanks go to my supervisor, Arish Siddiqui, whose guidance was direct, honest, and exactly what this project needed. He never let me settle for the convenient framing of a result or the easy conclusion, and the dissertation is considerably stronger for it. Having a supervisor who takes the work seriously is not something to take for granted.
    
    I am grateful to the University of East London for providing access to institutional data infrastructure, Bloomberg Terminal access, and the computational environment that made the empirical programme feasible. Some of the datasets used in this project — supply-chain topology, institutional equity returns, and SEC filing archives — each individually exceed what I could have accessed independently. The fact that this research was conducted at Bachelors level is, in large part, a function of that access.
    
    This project was built in the company of people who kept things in perspective when the results were stubborn and the deadline was not. Thank you to my friends and family — particularly those who sat through more than one explanation of why a negative $R^2$ is not a failure — for their interest, their humour, and their patience.
  ]
)

// --- MAIN BODY ---

#include "chapters/01-intro.typ"

#include "chapters/02-lit-review.typ"

#include "chapters/03-methodology.typ"

#include "chapters/04-findings-and-outcomes.typ"

#include "chapters/05-evaluation.typ"

#include "chapters/06-conclusion.typ"


// --- REFERENCES ---
#bibliography("works.bib", style: "harvard-cite-them-right")

// --- APPENDICES ---
#show: appendix

= Initial Project Proposal // Content for Appendix A 
[The initial project proposal is attached as a separate document.]

= Final Project Proposal // Content for Appendix B 
[The final project proposal, detailing the scope and methodology of the dissertation, is attached as a separate document.]

= Application for Approval of Research Activities // Content for Appendix C 
[The ethical approval form and application for research activities have been submitted to the University of East London.]

= Client Consent Form // Content for Appendix D
[Not applicable for this project as it relies entirely on publicly available financial datasets (SEC EDGAR, LSEG Refinitiv, Bloomberg SPLC) and involves no human participants.]
