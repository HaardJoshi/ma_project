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
    
   A large body of research finds that a majority of mergers and acquisitions fail to create value for shareholders, yet the tools used to predict these outcomes have barely changed in decades. Most models still treat companies as isolated islands of financial data, ignoring the complex networks of suppliers and customers that actually drive their success. This dissertation builds a new kind of "graph-aware" model that sees the economy as a web of relationships. By combining traditional financial ratios with supply-chain maps and a section-by-section analysis of corporate filings, I show that we can identify value-creating deals more accurately than before. 

   Using a dataset of 2,864 US acquisitions from 2000 to 2023, I built a multimodal system called HeteroGraphSAGE. The results prove that knowing a company's position in the supply chain provides a "topological alpha"—a predictive edge that financial numbers alone can't capture. I also found that natural language processing (NLP) is a double-edged sword: if you just "dump" all the text from a company report into a model, accuracy actually drops. However, when you specifically compare strategic plans (MD&A) against risk disclosures, the model begins to understand the true trade-offs of a deal. While predicting the exact dollar value of a merger remains incredibly difficult, this study demonstrates that we can reliably predict the *direction* of the outcome, providing a clearer way to triage deals in a noisy market.
  ],
  
  acknowledgments: [
   #set par(justify: true, leading: 0.65em)

    First and foremost, I want to thank my supervisor, Arish Siddiqui. There were several points where I was ready to settle for a "good enough" result or a simpler model, but Mr. Arish pushed me to dig deeper into the graph theory and the section-specific NLP that eventually became the core of this project. His honest, no-nonsense feedback kept me on track when the data felt overwhelming.
    
    I am also grateful to the UEL technical team and the library staff for helping me navigate the institutional data access. A special mention must go to the quiet corners of the library where I spent weeks trying to debug the SPLC node-mapping logic and the bizarre Python 3.14 environment freezes that nearly derailed the final evaluation. 
    
    Finally, thank you to my family and friends. To those who listened to my frustrated rants about "negative R-squared" values and the "M2 reversal" over coffee, and to those who simply reminded me to take a walk and step away from the terminal—thank you. Your support made the long nights of data engineering a lot more manageable.
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
#set heading(numbering: none)
#bibliography("works.bib", style: "harvard-cite-them-right")

// --- APPENDICES ---
#show: appendix

#include "chapters/appendix-a.typ"

#include "chapters/appendix-b.typ"

#include "chapters/appendix-c.typ"

#include "chapters/appendix-d.typ"

#include "chapters/appendix-e.typ"
