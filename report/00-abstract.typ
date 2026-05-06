// ============================================================
//  00-abstract.typ  (v2 - Simplified)
//  Abstract
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================
#set par(justify: true, leading: 0.65em)

= Abstract <ch-abstract>

A large body of research finds that a majority of mergers and acquisitions fail to create value for shareholders, yet the tools used to predict these outcomes have barely changed in decades. Most models still treat companies as isolated islands of financial data, ignoring the complex networks of suppliers and customers that actually drive their success. This dissertation builds a new kind of "graph-aware" model that sees the economy as a web of relationships. By combining traditional financial ratios with supply-chain maps and a section-by-section analysis of corporate filings, I show that we can identify value-creating deals more accurately than before. 

Using a dataset of 2,864 US acquisitions from 2000 to 2023, I built a multimodal system called HeteroGraphSAGE. The results prove that knowing a company's position in the supply chain provides a "topological alpha"—a predictive edge that financial numbers alone can't capture. I also found that natural language processing (NLP) is a double-edged sword: if you just "dump" all the text from a company report into a model, accuracy actually drops. However, when you specifically compare strategic plans (MD&A) against risk disclosures, the model begins to understand the true trade-offs of a deal. While predicting the exact dollar value of a merger remains incredibly difficult, this study demonstrates that we can reliably predict the *direction* of the outcome, providing a clearer way to triage deals in a noisy market.
