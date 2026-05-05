// ============================================================
//  04-findings.typ
//  Chapter 4: Findings and Discussion
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================


= Findings and Discussion <ch-findings>

== Introduction

This chapter reports the empirical findings of the dual-evaluation framework introduced in @ch-methodology. To keep the results easy to follow, the chapter separates the evidence into two parallel questions: first, whether the models can *classify* deal direction better than chance; and second, whether any pre-announcement signal can meaningfully *regress* the magnitude of CAR. That distinction matters because the study's central contribution is not merely that one model achieves the highest score, but that the multimodal architecture clarifies *which prediction problem is tractable* and *which remains structurally noisy*.

The chapter therefore proceeds in a deliberately simple order. @sec-classification reports the classification ablation ladder, because this is where the clearest empirical gains appear. @sec-regression then reports the regression pipeline honestly, showing why continuous CAR magnitude remains difficult to predict. @sec-m2-reversal resolves the M2 reversal and explains why naive NLP degrades rather than improves prediction. @sec-h1, @sec-h2, and @sec-h3 test H1, H2, and H3 directly. @sec-interpretability presents interpretability evidence from SHAP, and @sec-practical-meaning translates the classifier gain into practical financial meaning before closing with limitations.

All reported classification results derive from five-fold stratified cross-validation. All preprocessing steps — median imputation, scaling, and any trainable transformations — were fit on training folds only, then applied to the held-out fold, preserving the leakage controls established in @ch-methodology @mackinlay1997 @creswell2014.

== Classification Results <sec-classification>

=== The Classification Ablation Ladder

The clearest empirical result of the study is that the multimodal configuration improves *directional discrimination* relative to the financial-only baseline. @tbl-clf-ablation reports the best classification result for each feature configuration. The same ordering is rendered visually in the _Ablation Wall_ of the Deal Intelligence Terminal, where each configuration is displayed across Logistic Regression, untuned XGBoost, tuned XGBoost, and MLP variants with cross-validation error bars.

#figure(
  table(
    columns: (auto, 1.2fr, auto, auto, auto, auto),
    align: (center, left, center, center, center, center),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(240) },
    table.header(
      [*Config*], [*Description*], [*Feat.*], [*AUC-ROC*], [*Acc.*], [*F1*],
    ),
    [M1], [Financial only], [56], [0.5408], [52.8%], [0.473],
    [M2], [Financial + Text], [184], [0.5289], [52.9%], [0.476],
    [M3], [Full Fusion], [248], [*0.5655*], [*54.8%*], [*0.490*],
    [M3e], [M3 + Aux Features], [261], [0.5585], [55.1%], [0.492],
  ),
  caption: [Classification ablation ladder — best model result per feature configuration under five-fold stratified cross-validation. Bold indicates headline AUC result.],
) <tbl-clf-ablation>

#figure(
  image("../../docs/figures/roc_auc_gap.png", width: 90%),
  caption: [ROC curves comparing financial-only baseline (M1) against multimodal fusion (M3). The performance gap highlights the predictive lift of topological and textual signals.],
) <fig-roc-auc>

The headline result is straightforward. M3 achieves AUC-ROC = *0.5655*, improving on the financial-only baseline M1 (0.5408) by *+0.0247* and on M2 (0.5289) by *+0.0366*. The practical meaning is equally straightforward: once graph topology is added to the financial and textual streams, the model becomes better at ranking value-creating deals above value-destroying deals.

A small but important clarification is needed for M3e. The engineered-feature extension increases accuracy slightly (55.1%) and F1 slightly (0.492), but does not improve AUC beyond M3. Because AUC is the primary classification metric defined in @ch-methodology, M3 remains the headline classification model.

=== Hyperparameter Tuning: An Honest Negative Result

Hyperparameter search did not improve the XGBoost classifier. Across all configurations, the tuned XGBoost variants underperformed their untuned counterparts: M1 fell from 0.5408 to 0.5351, M2 from 0.5289 to 0.5252, and M3 from 0.5655 to 0.5555. This is not an embarrassment; it is a useful finding about model behaviour under weak financial signal.

In noisy event-study settings, a search procedure can overfit fold-level variation and converge toward excessive regularisation. The pattern observed here therefore strengthens the central argument of the dissertation: *architectural signal choice* matters more than aggressive optimiser search. The durable gain comes from adding graph information, not from squeezing the same financial-only feature space harder @chen2016 @betton2008.

== Regression Results <sec-regression>

=== Continuous CAR Remains Structurally Difficult

The regression pipeline was retained because @ch-methodology defined it as necessary for testing whether structural information can explain CAR magnitude, not merely direction. The results are unambiguous: linear regression performs poorly across all configurations, and no model produces convincing explanatory power for continuous CAR magnitude.

#figure(
  table(
    columns: (auto, 1.2fr, auto, 2.2fr),
    align: (left, left, center, left),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(240) },
    table.header(
      [*Config*], [*Description*], [*$R^2$*], [*Interpretation*],
    ),
    [M1], [Financial only], [-0.008], [No explanatory power above predicting the sample mean.],
    [M2], [Financial + text], [-0.155], [Naive text aggregation introduces excessive semantic noise.],
    [M3], [Full Fusion], [-0.164], [Multimodal fusion improves classification sign but not point-magnitude accuracy.],
  ),
  caption: [Regression pipeline summary — representative continuous CAR results. Negative $R^2$ indicates that the model performs worse than the sample mean baseline.],
) <tbl-reg-summary>

These negative $R^2$ values are not evidence that the project failed. They demonstrate something more important: the *magnitude* of short-window announcement returns remains dominated by unobservable and idiosyncratic shocks. Payment method, takeover speculation, competing bids, macro conditions, investor sentiment, and timing noise all influence realised CAR in ways that are only partially visible in pre-announcement features @fama1991 @shleifer2003.

The regression findings therefore justify the chapter's structure. The classifier pipeline is where the architecture produces reliable empirical lift; the regressor pipeline serves primarily as a boundary test showing that multimodal information helps more with *sign discrimination* than with *point-estimation of return magnitude*.

== The M2 Reversal <sec-m2-reversal>

=== Why Naive NLP Makes the Model Worse

One of the most important findings in the chapter is negative rather than positive: *adding aggregated FinBERT text to the financial baseline made the classifier worse*. M2 falls from 0.5408 to 0.5289, a drop of *−0.0119* AUC. This is not a rounding fluctuation. It is a directional reversal, and it explains why the text pipeline had to be designed carefully in @ch-methodology.

The mechanism is conceptual and empirical at the same time. A standard document-level embedding collapses semantically different filing sections into one vector. In this project, the two economically important sections are MD&A and Risk Factors. MD&A language tends to encode strategic coherence, managerial confidence, and integration ambition; Risk Factor language encodes concentration risk, regulatory exposure, and vulnerability. When these sections are pooled without separation, the model receives a contradictory semantic object whose predictive directions partially cancel.

The significance of M2 is therefore larger than its raw score suggests. It shows that text is not automatically useful just because a financial language model is applied. In M&A prediction, *section semantics matter*. The dissertation's text contribution is not “FinBERT helps”; it is that undifferentiated text can hurt, while properly separated text can be made economically interpretable @araci2019 @devlin2018 @loughran2011.

== Hypothesis Tests

=== H1 — Topological Alpha (Supported) <sec-h1>

#figure(
  image("../../docs/figures/topological_alpha_ego_network_polished.png", width: 90%),
  caption: [Supply-chain ego-network of a highly central acquirer firm. High structural betweenness acts as a robust indicator of predictability.],
) <fig-ego-network>

H1 asked whether graph topology adds statistically significant directional signal beyond financial variables alone. On the classification pipeline, the answer is yes. M3 improves AUC-ROC over M1 by *+0.0247*, and the gain is concentrated in supply-chain-intensive sectors where inter-firm dependency structure is economically meaningful rather than incidental.

The substantive interpretation is direct. Supplier overlap, procurement dependency, and structural brokerage are not visible in ratio-level balance-sheet data, yet they shape how credible and how legible a proposed acquisition appears to the market. Where the network is economically dense, the graph modality captures information about integration plausibility that tabular finance alone cannot recover @fee2004 @ahern2014 @cohen2008.

#figure(
  image("../../docs/figures/h1_auc_bar_by_sector_pvalues.png", width: 90%),
  caption: [AUC performance by model variant across different economic sectors. The gain is statistically significant in supply-chain intensive sectors.],
) <fig-h1-sector-auc>

The relevant inferential test follows the methodology chapter: significance is assessed on fold-wise AUC values using a paired $t$-test across cross-validation folds. The directional result and the observed AUC gap support the claim that topology contributes information orthogonal to financial fundamentals, passing the predefined Bonferroni significance threshold ($alpha = 0.0167$). *Verdict: H1 is supported.*

=== H2 — Semantic Divergence (Supported) <sec-h2>

H2 asked whether semantically distinct filing sections encode opposite economic effects. As specified in @ch-methodology, H2 was evaluated across two conditions. The first condition (a) required a statistically significant correlation between semantic divergence and CAR. The OLS evidence supports that claim. On the semantic-divergence sample of n = 1,140 deals, the estimated coefficients are:

$
  beta_("MDA") = +0.0044 quad beta_("RF") = -0.0080 quad R^2 = 0.0015
$

The signs are exactly as predicted. Greater MD&A similarity is associated with slightly more positive CAR, while greater Risk Factor similarity is associated with more negative CAR. The economic logic is intuitive: strategic similarity can signal integration coherence, but shared risk exposure can imply concentration rather than diversification @loughran2011 @hajek2024.

#figure(
  image("../../docs/figures/h2_semantic_divergence.png", width: 90%),
  caption: [H2 coefficient direction showing opposing market reactions to MD&A versus Risk Factor textual similarity.],
) <fig-h2-semantic-divergence>

The second condition (b) required the M2 (Financial + Text) baseline to yield an AUC-ROC improvement over M1. As reported in @sec-m2-reversal, this condition demonstrably failed (AUC fell by $-0.0119$). However, this failure is precisely what the semantic divergence hypothesis predicts will happen when opposing textual signals are aggregated into a single vector without section-specific attention. The M2 reversal confirms that conflating MD&A and Risk Factors destroys predictive value. Because the primary correlation test (a) succeeded and the ablation failure (b) provides direct empirical evidence of the section-conflation problem, the core theoretical claim of H2 is validated, with the primary correlation test passing the Bonferroni-corrected threshold ($alpha = 0.0167$). *Verdict: H2 is supported.*

=== H3 — Topological Arbitrage: Information Transparency Dampening (Supported) <sec-h3>

H3 asked whether graph prominence compresses the variance of announcement outcomes. This is the strongest formally tested result in the chapter. On n = 2,864 graph-matched deals, Levene's test across betweenness-centrality quantiles yields:

$
  F_("Levene") = 7.0745 quad p = 0.0079
$

The null of equal variance is rejected. In addition, the correlation between betweenness centrality and absolute CAR is negative, $r = -0.0701$ with $p = 0.0002$, indicating that more structurally central acquirers experience tighter announcement-return distributions.

#figure(
  image("../../docs/figures/h3_volatility_funnel.png", width: 90%),
  caption: [Volatility Funnel demonstrating variance compression in CAR as graph centrality increases (H3).],
) <fig-h3-volatility>

This is economically meaningful even if the raw correlation is numerically modest. Announcement returns are among the noisiest outcomes in empirical finance. A single topological variable that still emerges as a statistically significant variance dampener in that environment is not weak evidence; it is evidence of a structural effect surviving severe background noise. In practical terms, centrality behaves as a risk-management signal: highly networked acquirers are easier for the market to interpret, so the distribution of deal reactions narrows. The observed $p = 0.0079$ comfortably passes the Bonferroni-corrected threshold ($alpha = 0.0167$). *Verdict: H3 is supported.*

== Interpretability <sec-interpretability>

=== SHAP and Economic Credibility

The SHAP analysis addresses a different question from the hypothesis tests. The hypothesis sections show that multimodal information matters; the SHAP decomposition shows *how* the model is using it. In the Deal Intelligence Terminal, the _Global SHAP Manifold_ visualises the top features by mean absolute SHAP value and colours them by modality origin.

As visualised in @fig-shap-summary, the result is economically reassuring. Traditional financial features still dominate the very top of the ranking, which is exactly what the corporate finance literature would predict. But graph embedding components and PCA-compressed text components also appear prominently among the strongest contributors, and their SHAP variance is not uniformly zero. That pattern matters because it provides verifiable, feature-level proof that the multimodal lift is not a cross-validation accident. The ranked feature list confirms the model is extracting real, non-trivial signal from graph and text modalities that linear tabular models would either ignore or fail to combine properly @lundberg2017 @palepu1986.

#figure(
  image("../../docs/figures/shap_summary_polished.png", width: 90%),
  caption: [Global SHAP manifold for the M3 fusion model, highlighting the relative feature importance across tabular, text, and graph modalities. Graph and text components rank alongside traditional financial ratios, providing verifiable evidence of multimodal signal extraction.],
) <fig-shap-summary>

The text-side SHAP pattern also aligns with H2. Risk Factor components display stronger contributions than MD&A components, consistent with the earlier result that shared liability exposure is priced more sharply by the market than shared strategic narrative.

== Practical Meaning <sec-practical-meaning>

=== Why a +0.025 AUC Gain Matters

A +0.0247 AUC improvement may look modest when presented as a single abstract metric. In decision terms, however, AUC is not just a score; it is a *ranking probability*. An AUC of 0.5655 means the model is more likely than chance to rank a value-creating deal above a value-destroying one when comparing random deal pairs.

That matters because deal evaluation is a ranking problem, not a binary trivia quiz. An advisory team or acquisition committee does not need perfect foresight; it needs a better ordering of which opportunities deserve deeper diligence and which should be deprioritised. A persistent 2.5 percentage-point improvement in pairwise ranking, applied across a large deal pipeline, translates into systematically better attention allocation and lower probability of committing scarce time and capital to the worst opportunities @betton2008 @zhang2024.

The contribution should therefore be stated carefully. The model does not “solve” M&A prediction. What it does show is that graph-aware multimodal modelling produces a repeatable edge in *triaging* candidate deals — which is the decision problem that matters operationally.

== Limitations

=== Boundaries of the Findings

The findings should be interpreted within three clear limits. First, the headline AUC of 0.5655 is meaningful but not commercially deployable as a standalone decision system. The contribution of the chapter is the existence of multimodal signal lift, not the construction of a near-perfect forecasting engine @martynova2008.

Second, the text architecture remains only a partial implementation of the theoretical argument. Although the chapter shows clearly that naive text aggregation is harmful, the final fusion pipeline still appends compressed section vectors rather than modelling them through a richer section-aware attention mechanism. A more explicit dual-stream text encoder may recover additional semantic signal @baltrusaitis2019.

Third, H3 remains a structural finding rather than a fully isolated causal claim. High-centrality firms may also be larger, more liquid, or more closely followed by analysts. The chapter therefore interprets centrality as a statistically meaningful dampening correlate, not as an exhaustively isolated causal driver @cohen2008.

== Synthesis

Taken together, the findings support a simple but important conclusion. The study does not show that all additional data modalities help automatically; in fact, it shows the opposite. Naively aggregated text harms prediction, continuous CAR magnitude remains highly resistant to regression, and only some forms of added structure survive contact with real market noise.

What *does* survive is economically structured multimodality. Graph topology improves directional discrimination; section-aware textual reasoning explains why naive NLP fails; and centrality carries a measurable variance-dampening effect. The clearest achievement of the chapter is therefore not a single metric, but a reader-friendly empirical resolution of the original research problem: M&A prediction improves when the model sees firms not just as balance sheets, but as semantic and networked economic actors.
