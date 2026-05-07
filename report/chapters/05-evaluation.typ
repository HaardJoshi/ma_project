// ============================================================
//  05-evaluation.typ  (v1 - Full Academic Rigour)
// Chapter 5: Evaluation
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================


= Evaluation <ch-evaluation>

== Introduction

This chapter provides a structured, objective evaluation of both the research product and the research process. Product evaluation assesses what was built - the multimodal M&A synergy prediction pipeline and its operational proof-of-concept, the Deal Intelligence Terminal - against its stated objectives, performance benchmarks, and the academic literature it set out to surpass. Process evaluation reflects critically on the methodological decisions made throughout the project: what worked as intended, what produced unexpected results, and what a future researcher should do differently.

The distinction between product and process is not cosmetic. A product can succeed against benchmarks while the process that generated it contains correctable inefficiencies; conversely, a rigorous process can still produce a product whose limitations are significant. Both dimensions are evaluated here with equal honesty, because intellectual credibility depends on the willingness to apply the same critical standards to one's own work that were applied to the prior literature in @ch-litreview.

== Product Evaluation <sec-product-evaluation>

=== Objectives and Achievement

The study set out three primary objectives, each of which maps directly to a testable hypothesis and an architectural component:

+ *Objective 1:* Demonstrate that graph topology adds statistically significant directional signal to M&A synergy prediction beyond financial fundamentals. This corresponds to H1 and the Block C (HeteroGraphSAGE #cite(<hamilton2017>, form: "prose")) component.

+ *Objective 2:* Show that the semantic direction of textual similarity depends on document section, with MD&A and Risk Factor similarity carrying opposite predictive signs. This corresponds to H2 and the section-aware FinBERT pipeline (#cite(<araci2019>, form: "prose")) (Block B).

+ *Objective 3:* Establish that network centrality compresses the variance of announcement returns, providing a structural signal about market predictability @larcker2013. This corresponds to H3 and the betweenness centrality analysis.

All three objectives were met. H1 was supported with an AUC gain of +0.0247 (M3 vs. M1); H2 was supported with $beta_("MDA") = +0.0044$ and $beta_("RF") = -0.0080$ in the correct directions; and H3 was supported with Levene's $F = 7.0745$ ($p = 0.0079$) confirming variance compression across centrality quantile groups. The fourth implicit objective - the construction of an interactive research platform - was achieved in the Deal Intelligence Terminal, which renders all empirical results dynamically and serves as an operational proof of concept for real-time multimodal deal analysis.

=== Performance Against the Literature

The primary AUC benchmark in this domain is the financial-only logistic regression and gradient-boosting ceiling documented across M&A studies. #cite(<palepu1986>, form: "prose") and #cite(<barnes1990>, form: "prose") consistently reported pseudo-$R^2$ below 0.10 on ratio-based models. #cite(<zhang2024>, form: "prose") achieved modest accuracy improvements over logistic baselines using random forests but remained architecturally bounded by the independence assumption. No prior study directed a multimodal fusion model incorporating heterogeneous graph topology at binary CAR direction classification.

#figure(
  table(
    columns: (1.2fr, 1fr, 1.2fr, 0.8fr),
    align: (left, center, left, center),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(240) },
    table.header(
      [*Study*], [*Method*], [*Target / Metric*], [*AUC / Acc.*\*],
    ),
    [@palepu1986; @barnes1990], [Logit / MDA], [Acquisition likelihood], [N/A (Acc.~58%)],
    [@zhang2024], [XGBoost on ratios], [Deal success proxies], [~0.56-0.58\*],
    [@elhoseny2022], [Deep MLP (AWOA-DL)], [Financial distress], [0.958\*],
    [@hajek2024], [FinBERT sentiment], [Acquisition occurrence], [N/A (F1 ~0.71\*)],
    [*This study - M1*], [*XGBoost (Fin.)*], [*Binary CAR direction*], [*0.5408*],
    [*This study - M3*], [*Multimodal Fusion*], [*Binary CAR direction*], [*0.5655*],
  ),
  caption: [Contextualised product performance. Asterisked figures derive from different prediction targets and are not directly comparable; they are included to contextualise the difficulty of the prediction problem addressed here.],
) <tbl-lit-comparison>

The comparison in @tbl-lit-comparison requires careful interpretation. The AUC range reported by #cite(<zhang2024>, form: "prose") (~0.56-0.58) applies to proxies for long-term deal success, not short-term announcement CAR direction, making direct comparison difficult despite the overlapping numerical values. Similarly, high accuracy figures from #cite(<elhoseny2022>, form: "prose") and high F1 figures from #cite(<hajek2024>, form: "prose") address different, structurally easier prediction targets - financial distress detection and deal occurrence classification - rather than the genuinely harder problem of binary CAR direction. The ceiling this study pushes against is the approximately 0.55 AUC limit on ratio-based M&A models, not the performance of architectures optimised for different financial tasks. Against that correct benchmark, the M3 gain of +0.0247 is meaningful and directionally consistent with the theoretical prediction.

=== Product Limitations

The product should be evaluated against the standards of a deployable decision tool as well as an academic contribution. Against that second standard, three limitations are significant.

*AUC and deployment readiness.* An AUC of 0.5655, while better than the documented tabular ceiling for this exact prediction task, does not reach the performance threshold that an investment bank or advisory firm would require before incorporating a model into live deal screening. Commercial M&A analytics tools typically target AUC above 0.70 for directional predictions with real capital at stake. The product as built is a proof of concept establishing that multimodal signal exists and is structurally recoverable - it is not a production system. A dissertation-scale study produces a proof of concept establishing that multimodal signal exists; it is not designed to meet the deployment thresholds of commercial analytics platforms.

*Text architecture incompleteness.* The section-aware FinBERT pipeline demonstrates that section semantics matter and that undifferentiated text is harmful, but the actual implementation still compresses each section into PCA-reduced vectors before fusion. A richer implementation would apply cross-section attention, learning the interaction between MD&A and Risk Factor signals within a unified text encoder rather than treating them as independent scalar contributions. The current architecture makes the theoretical argument but does not fully execute the representational sophistication it implies.

*Graph coverage sparsity.* Bloomberg SPLC and related data sources provide reliable supply-chain relationships for large, publicly traded firms with extensive analyst coverage. Mid-cap and smaller acquirers frequently have partial or absent graph coverage. The product's performance on deals involving structurally well-documented acquirers may not generalise to the long tail of smaller transactions where graph data are sparse, creating a selection bias toward deals where the product works best. Future iterations must integrate alternative graph construction methods-such as extracting supplier-customer linkages from automated parsing of earnings call transcripts or proprietary Private Equity datasets-to resolve this sparsity.

=== The Deal Intelligence Terminal as Product

The Deal Intelligence Terminal deserves evaluation as a distinct product component. Its role is not merely to visualise results - it constitutes the operational proof that the multimodal architecture can be presented interactively to non-technical stakeholders without sacrificing rigour. The terminal's five phases (Data Profile, Model Evidence, SHAP Attribution, Hypothesis Lab, and Scenario Analysis) map directly onto the five empirical claims of the dissertation, ensuring that every assertion made in the text has a dynamic, interrogable visual counterpart.

The product succeeds here. Every hypothesis result, SHAP decomposition, and ablation comparison is rendered in real time from the underlying results JSON files, meaning that any examiner or future researcher who modifies the data pipeline will see updated results immediately without re-editing the text. This reproducibility property exceeds what is typically achieved in static academic reports and constitutes a secondary product contribution beyond the model itself.

== Process Evaluation <sec-process-evaluation>

=== Research Design Decisions That Worked

Several design decisions proved more valuable in practice than their theoretical justification alone suggested.

*Binary classification over continuous regression.* The decision to classify CAR direction rather than predict its magnitude was theoretically motivated in @ch-litreview but repeatedly validated in practice. Every time a regression model was tested on continuous CAR values, negative $R^2$ values confirmed that magnitude is dominated by noise. Had the study committed to regression as the primary objective, the project would have generated results too weak to build a coherent argument around. The binary formulation created a tractable problem precisely because it is harder to drown a directional signal in noise than a magnitude signal.

*Late fusion architecture.* The decision to encode each modality independently and fuse downstream rather than training end-to-end proved correct for the dataset available. With 2,864 complete multimodal observations, end-to-end training would almost certainly have collapsed into memorisation of training noise rather than genuine representation learning. Late fusion imposed a useful discipline: each modality had to be meaningful on its own before being combined, and the ablation ladder directly tests that property.

*Treating the M2 reversal as a finding, not a failure.* In a less carefully designed study, the M2 AUC drop would have been hidden, smoothed, or blamed on a FinBERT implementation bug. Elevating it to a first-class theoretical finding - that undifferentiated text actively destroys predictive value - transforms what could have been an embarrassment into one of the study's clearest and most defensible contributions. This required the intellectual courage to report a negative result prominently rather than burying it in an appendix.

*Leakage control discipline.* Fitting all preprocessing within the cross-validation loop rather than globally was operationally expensive but scientifically essential. Early experiments revealed that globally-fitted scalers inflated validation AUC non-trivially. Enforcing per-fold fitting reduced reported performance but produced estimates that are, to the extent possible given the data, realisable in deployment. A study that inflates performance through silent leakage produces findings that are formally unpublishable and practically misleading.

=== Research Design Decisions That Underperformed

An honest process evaluation also requires identifying where the process fell short of what was possible.

*Hyperparameter search was counterproductive.* The effort invested in Bayesian search via Optuna (#cite(<akiba2019>, form: "prose")) yielded a critical domain-specific finding: surrogate-based optimizers systematically over-regularize in high-noise financial regimes. While it reduced AUC across every configuration, this provided a valuable methodological lesson: in noisy financial signal environments, architectural diversity (what features you include) is more valuable than optimiser sophistication (how finely you tune the same feature set). Future work should invest optimiser effort only after establishing that the feature space contains enough signal to reward fine-grained search.

*Graph coverage could have been deeper.* The HeteroGraphSAGE component achieves the headline AUC gain, but the graph neighbourhood depth was limited to two hops due to computational constraints. Economic theory suggests that second and third-order supplier dependencies - the suppliers of suppliers - carry meaningful risk propagation signals that two-hop neighbourhoods capture only partially. The gain demonstrated is therefore likely a lower bound on what a fully realised graph component could achieve with deeper neighbourhood sampling or a richer edge-weighting scheme.

*PCA on FinBERT embeddings.* This was evaluated as a defensive design choice in @ch-methodology and that assessment stands. But the process evaluation adds a practical observation: the PCA compression was decided at the architecture design stage primarily on theoretical grounds, and its empirical effect was never rigorously ablated independently of the other text components. A cleaner experimental design would have tested compressed versus uncompressed embeddings in isolation at a stage where their independent effect could be measured, rather than evaluating them only within the full text block. This gap in the ablation ladder means it is not possible to quantify how much predictive value the PCA step sacrificed. However, this represents a natural limitation of a full-scope study at dissertation scale, and constitutes a clear methodological agenda item for a journal extension.

=== Scope and Time Management

The project scope was ambitious relative to the time and data constraints of a BSc dissertation. Integrating three data environments (Yahoo Finance, Bloomberg SPLC, SEC EDGAR), building a full feature engineering pipeline across three modalities, training and evaluating multiple model families under proper cross-validation, implementing three formal hypothesis tests, building a production-quality interactive dashboard, and writing an academic dissertation simultaneously represents a substantial engineering and research undertaking.

The consequence of this scope is that depth in certain areas was necessarily sacrificed for breadth. The graph component could have been more deeply developed; the text pipeline could have included a richer dual-stream encoder; the regression analysis could have been more systematically reported. Each of these represents a genuine product limitation, and each originates in a process decision to maintain the full three-modality scope rather than narrowing to one or two modalities and pursuing them more thoroughly.

Whether this was the correct trade-off is debatable. The broadest intellectual contribution - that multimodal fusion combining financial, textual, and topological signals improves M&A prediction - required all three modalities to be present in the experiment. Removing any one of them would have prevented the ablation ladder from demonstrating the specific contribution of each stream. The scope was therefore not incidental; it was structurally necessary for the central argument. The cost was depth of execution in each individual stream.

=== Reflections on the Research Question

Looking back across the full project, the research question - whether heterogeneous graph topology adds predictive signal to M&A outcome classification beyond financial and textual baselines - was well-posed and proved to be answerable at dissertation scale. The answer is yes, and the evidence is sufficiently clear that it survives the most obvious methodological objections.

What was perhaps underestimated at the outset was how much of the project's intellectual value would come from negative results: the M2 reversal showing that naive NLP hurts, the negative $R^2$ values confirming that continuous CAR regression is intractable, and the hyperparameter search confirming that tuning without architectural improvement produces nothing. These negative findings are collectively as valuable as the positive AUC gain, because they precisely specify the boundary conditions under which multimodal fusion succeeds and fails. A project that produced only positive results without these boundary tests would have made a narrower and less credible contribution.

The study therefore succeeds not merely because AUC improved, but because it provides a structured empirical account of what types of information help, what types hurt, and why the architectural choices made in @ch-methodology were necessary rather than optional. Reporting the parts that didn't work, like the regression results and the M2 reversal, made the final AUC gain feel much more like a real finding rather than just a fluke.

== Personal Reflection and Mental Wealth <sec-personal-reflection>

Building a multimodal model for M&A wasn't just a technical challenge; it was an exercise in managing the frustration that comes with high-stakes data engineering. Over the course of the project, I had to completely change how I approached technical failure and personal productivity to meet the Mental Wealth requirements of the course.

The most difficult period was midway through the implementation, when I was struggling to merge the Bloomberg SPLC data with the SEC filing archives. I hit a series of "silent" bugs where deals were being dropped because of minor ticker mismatches, and for a while, it felt like the more data I added, the worse the model performed. The "Python 3.14 freeze" I encountered during the final training run was a breaking point—I spent twelve hours trying to fix a compilation error that turned out to be a known environment bug. Initially, I took these roadblocks personally, feeling that my inability to "just fix it" meant I wasn't up to the task. 

However, the real breakthrough in my personal development came when I stopped trying to brute-force every problem. I had to unlearn the "hacker" habit of working until 3:00 AM on four cups of coffee. I started setting hard cut-off times for coding and forced myself to take breaks even when a bug wasn't fixed. This sounds simple, but for a project this size, it was the only way to stay sharp enough to spot the "M2 reversal" pattern in the results. I realized that the negative results—like the failure of the regression models—weren't personal failures; they were actually some of the most interesting findings in the whole study. Learning to step away from the terminal and look at the project with a rested mind was probably the most valuable skill I gained, and it’s something I’ll take with me into my professional career.
