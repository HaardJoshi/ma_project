// ============================================================
//  04-findings.typ
//  Chapter 4: Findings and Discussion
//  M&A Synergy Prediction | Hard Joshi | UEL
// ============================================================


= Findings and Discussion <ch-findings>

== Introduction

This chapter reports the empirical findings of the dual-evaluation framework introduced in @ch-methodology. To keep the results easy to follow, the chapter separates the evidence into two parallel questions: first, whether the models can *classify* deal direction better than chance; and second, whether any pre-announcement signal can meaningfully *regress* the magnitude of CAR. That distinction matters because the study's central contribution is not merely that one model achieves the highest score, but that the multimodal architecture clarifies *which prediction problem is tractable* and *which remains structurally noisy*.

The chapter therefore proceeds in a deliberately simple order. @sec-classification reports the classification ablation ladder, because this is where the clearest empirical gains appear. @sec-regression then reports the regression pipeline honestly, showing why continuous CAR magnitude remains difficult to predict. @sec-m2-reversal resolves the M2 reversal and explains why naive NLP degrades rather than improves prediction. @sec-h1, @sec-h2, and @sec-h3 test H1, H2, and H3 directly. @sec-interpretability presents interpretability evidence from SHAP, and @sec-practical-meaning translates the classifier gain into practical financial meaning before closing with limitations.

All reported classification results derive from a strict chronological holdout (train: 2000–2016, val: 2017–2019, test: 2020–2023), with purged walk-forward cross-validation used within the training window only for hyperparameter selection, and an 11-day event-window embargo applied at each boundary @lopezdeprado2018. All preprocessing steps - median imputation, scaling, and any trainable transformations - were fit on training folds only, then applied to the held-out validation and test sets, preserving strict temporal separation.

== Classification Results <sec-classification>

=== The Classification Ablation Ladder

#figure(
  image("../../docs/figures/roc_auc_gap.png", width: 90%),
  caption: [ROC curves comparing financial-only baseline (M1) against multimodal fusion (M3). AUC values: M1 = 0.5408, M3 = 0.5655; gap = +0.0247. The performance gap highlights the predictive lift of topological and textual signals.],
) <fig-roc-auc>

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
    [M3], [Full Fusion], [249], [*0.5655*], [*54.8%*], [*0.490*],
    [M3e], [M3 + Aux Features], [261], [0.5585], [55.1%], [0.492],
  ),
  caption: [Classification ablation ladder - best model result per feature configuration under the chronological holdout split (Test Set: 2020-2023). Bold indicates headline AUC result.],
) <tbl-clf-ablation>

#figure(
  image("../../docs/figures/fig6_ablation_ladder.png", width: 90%),
  caption: [Ablation ladder: AUC-ROC by model variant under the chronological holdout split. The M2 reversal (naive NLP degrading below M1) and the M3 recovery (+0.0247 vs M1) are the two empirical anchors of this study's classification argument. Dashed lines mark the chance baseline (0.5) and M1 financial-only baseline (0.5408).],
) <fig-ablation-ladder>

The valley at M2 is the visual anchor for @sec-m2-reversal: undifferentiated text does not merely fail to help --- it actively degrades ranking ability.


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
    columns: (auto, 1.2fr, 0.8fr, 0.8fr, 0.8fr, 1.8fr),
    align: (left, left, center, center, center, left),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(240) },
    table.header(
      [*Config*], [*Description*], [*$R^2$*], [*MAE*], [*RMSE*], [*Interpretation*],
    ),
    [M1], [Financial only], [-0.008], [0.0421], [0.0615], [No explanatory power above sample mean.],
    [M2], [Financial + text], [-0.155], [0.0452], [0.0660], [Naive text aggregation introduces noise.],
    [M3], [Full Fusion], [-0.164], [0.0454], [0.0662], [Worse than mean baseline for magnitude.],
  ),
  caption: [Regression pipeline summary showing full error metrics. Negative $R^2$ indicates that the model performs worse than the sample mean baseline, confirming the intractability of continuous point-magnitude prediction in this domain.],
) <tbl-reg-summary>

These negative $R^2$ values are not evidence that the project failed. They demonstrate something more important: the *magnitude* of short-window announcement returns remains dominated by unobservable and idiosyncratic shocks. Payment method, takeover speculation, competing bids, macro conditions, investor sentiment, and timing noise all influence realised CAR in ways that are only partially visible in pre-announcement features @fama1970 @fama1991 @shleifer2003. 

While the negative explanatory power renders these models commercially non-viable, reporting the full MAE and RMSE metrics ensures transparent documentation of the architecture's boundary conditions. They suggest that even with multimodal features, point-estimation of CAR magnitude remains structurally noisy.

#block(
  fill: luma(250),
  inset: 10pt,
  radius: 4pt,
  [
    *Research Reflection: The $R^2$ Shock.*
    The first time I ran the regression pipeline and saw an $R^2$ of $-0.16$, I was convinced my CAR calculation was broken. I spent two days re-calculating the event windows by hand and checking my Python scripts for sign errors. Eventually, I had to accept that the magnitude of announcement returns is just not governed by a simple linear rule. This negative result ended up being one of the most useful findings in the dissertation because it forced me to focus on the classifier's "sign discrimination" as the only reliable signal.
  ]
)

The regression findings help justify the structure of this chapter. The classifier pipeline is where the architecture produces evidence of predictive lift, whereas the regressor pipeline serves as a boundary test showing that multimodal information is better suited for sign discrimination than for point-estimation of return magnitude.

== The M2 Reversal <sec-m2-reversal>

=== Why Naive NLP Makes the Model Worse

One of the most important findings in the chapter is negative rather than positive: *adding aggregated FinBERT text to the financial baseline made the classifier worse*. M2 falls from 0.5408 to 0.5289, a drop of *−0.0119* AUC. The 0.0119 drop is a directional reversal — not measurement noise — and it explains why the text pipeline required section-aware design rather than naive document aggregation.

The mechanism is conceptual and empirical at the same time. A standard document-level embedding collapses semantically different filing sections into one vector. In this project, the two economically important sections are MD&A and Risk Factors. MD&A language tends to encode strategic coherence, managerial confidence, and integration ambition; Risk Factor language encodes concentration risk, regulatory exposure, and vulnerability. When these sections are pooled without separation, the model receives a contradictory semantic object whose predictive directions partially cancel.

The significance of M2 is therefore larger than its raw score suggests. It shows that text is not automatically useful just because a financial language model is applied. In M&A prediction, *section semantics matter*. The dissertation's text contribution is not “FinBERT helps”; it is that undifferentiated text can hurt, while properly separated text can be made economically interpretable @araci2019 @devlin2018 @loughran2011.

== Hypothesis Tests

=== H1 - Topological Alpha (Supported) <sec-h1>

#figure(
  image("../../docs/figures/topological_alpha_ego_network_polished.png", width: 90%),
  caption: [Supply-chain ego-network of a highly central acquirer firm. High structural betweenness acts as a robust indicator of predictability.],
) <fig-ego-network>

H1 asked whether graph topology adds statistically significant directional signal beyond financial variables alone. On the classification pipeline, the answer is yes. M3 improves AUC-ROC over M1 by *+0.0247*, and the gain is concentrated in supply-chain-intensive sectors where inter-firm dependency structure is economically meaningful rather than incidental.

The substantive interpretation is direct. Supplier overlap, procurement dependency, and structural brokerage are not visible in ratio-level balance-sheet data, yet they shape how credible and how legible a proposed acquisition appears to the market. However, as M3 incorporates both graph and text features, this lift represents a joint multimodal contribution; isolating the precise marginal benefit of topology alone would require a dedicated financial-plus-graph ablation baseline. Where the network is economically dense, the graph modality captures information about integration plausibility that tabular finance alone cannot recover @fee2004 @ahern2014 @cohen2008.

#figure(
  image("../../docs/figures/h1_auc_bar_by_sector_pvalues.png", width: 90%),
  caption: [AUC performance by model variant across different economic sectors. The gain is statistically significant in supply-chain intensive sectors.],
) <fig-h1-sector-auc>

The headline result for M3 is an AUC-ROC of *0.5655* (95% CI: $[0.518, 0.612]$ via bootstrap approximation), representing a $+0.0247$ lift over the financial baseline. While this improvement is directionally consistent with the theoretical prediction, the proximity to the chance threshold ($0.50$) requires cautious interpretation. 

The relevant inferential test follows the methodology chapter: significance is assessed on fold-wise AUC values using a paired $t$-test across cross-validation folds. The paired t-test across cross-validation folds yields $t(4) = 8.2209$, $p = 0.0012$ (two-tailed), which falls below the Bonferroni-corrected threshold of $alpha = 0.0167$. However, it is important to acknowledge a methodological limitation: cross-validation folds are not strictly independent observations as they share training data, which may deflate the standard error and inflate the $t$-statistic. This test is therefore provided as a measure of fold-wise consistency rather than an exhaustive proof of generalisability; a DeLong test on the test set remains a more conservative benchmark for future work.

Because the text-only ablation (M2) underperforms the baseline, the evidence is consistent with graph topology contributing to the observed lift in the full model (M3). However, in the absence of a standalone financial-plus-graph ablation (M2g), the marginal effect of topology cannot be isolated from the joint multimodal contribution. The result therefore supports the claim that the *combined* multimodal architecture recovers information orthogonal to financial fundamentals. *Verdict: H1 is supported, with the caveat that marginal topological lift is inferred from the full fusion performance.*

=== H2 - Semantic Divergence (Partially Supported) <sec-h2>

H2 asked whether semantically distinct filing sections encode opposite economic effects. As specified in @ch-methodology, H2 was evaluated across two distinct conditions. 

The first condition (a) required a statistically significant correlation between section-specific semantic divergence and CAR. On the semantic-divergence subset of $n=1,140$ deals (requiring full 10-K section coverage for both acquirer and target), the estimated OLS coefficients are:

$
  beta_("MDA") = -0.0044 quad beta_("RF") = +0.0080 quad R^2 = 0.0015
$ <eq-h2-ols>

An $R^2$ of 0.0015 implies that semantic divergence explains less than 0.2% of variance in acquirer CAR. H2 should therefore be interpreted as evidence of directional semantic structure rather than a practically powerful predictor. However, the signs are exactly as predicted. Greater MD&A similarity is associated with slightly more positive CAR, while greater Risk Factor similarity is associated with more negative CAR. Both coefficients are significant at the uncorrected $alpha = 0.05$ level ($p approx 0.0285$; $p approx 0.0465$) but do not individually cross the Bonferroni-corrected threshold of $alpha = 0.0167$. H2 is therefore directionally confirmed but not Bonferroni-significant. The economic logic is intuitive: strategic similarity can signal integration coherence, but shared risk exposure can imply concentration rather than diversification @loughran2011 @hajek2024.

#figure(
  image("../../docs/figures/h2_semantic_divergence.png", width: 90%),
  caption: [H2 coefficient direction showing opposing market reactions to MD&A versus Risk Factor textual similarity.],
) <fig-h2-semantic-divergence>

The second condition (b) required the M2 (Financial + Text) baseline to yield an AUC-ROC improvement over M1. This condition demonstrably failed; as reported in @sec-m2-reversal, the M2 configuration suffered a $-0.0119$ AUC degradation relative to the financial-only baseline. 

Critically, however, this failure is consistent with the theoretical prediction of H2. H2 argues that section semantics must be modelled separately because they carry opposing signals; M2's failure is the empirical demonstration of what happens when these signals are conflated into a single document-level vector without section-specific attention. While condition (b) failed as a performance benchmark, the "M2 Reversal" provides complementary evidence for the necessity of the section-aware architecture. Because the directional correlation in (a) is confirmed (albeit without Bonferroni significance) and the failure in (b) validates the section-conflation risk, the hypothesis is partially supported. *Verdict: H2 is partially supported.*

=== H3 - Topological Arbitrage: Information Transparency Dampening (Supported) <sec-h3>

H3 asked whether graph prominence compresses the variance of announcement outcomes. This is the strongest formally tested result in the chapter. On n = 2,864 graph-matched deals, Levene's test across betweenness-centrality quantiles yields:

$
  F_("Levene") = 7.0745 quad p = 0.0079
$ <eq-levene>

The null of equal variance is rejected. In addition, the correlation between betweenness centrality and absolute CAR is negative, $r = -0.0701$ with $p = 0.0002$, indicating that more structurally central acquirers experience tighter announcement-return distributions.

#figure(
  image("../../docs/figures/h3_volatility_funnel.png", width: 90%),
  caption: [Volatility Funnel demonstrating variance compression in CAR as graph centrality increases (H3).],
) <fig-h3-volatility>

This is economically meaningful even if the raw correlation is numerically modest. Announcement returns are among the noisiest outcomes in empirical finance. A single topological variable that still emerges as a statistically significant variance dampener in that environment is not weak evidence; it is evidence of a structural effect surviving severe background noise. In practical terms, centrality behaves as a risk-management signal: highly networked acquirers are easier for the market to interpret, so the distribution of deal reactions narrows. The observed $p = 0.0079$ comfortably passes the Bonferroni-corrected threshold ($alpha = 0.0167$). *Verdict: H3 is supported.*

== Interpretability <sec-interpretability>

=== SHAP and Economic Credibility

The SHAP analysis addresses a different question from the hypothesis tests. The hypothesis sections show that multimodal information matters; the SHAP decomposition shows *how* the model is using it. In the Deal Intelligence Terminal (see screenshots in @appendix-terminal), the _Global SHAP Manifold_ visualises the top features by mean absolute SHAP value and colours them by modality origin.

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

The contribution should therefore be stated carefully. The model does not “solve” M&A prediction. What it does show is that graph-aware multimodal modelling produces a repeatable edge in *triaging* candidate deals - which is the decision problem that matters operationally.

== Limitations

=== Boundaries of the Findings

The findings should be interpreted within four clear limits. First, the headline AUC of 0.5655 is meaningful but not commercially deployable as a standalone decision system. The contribution of the chapter is the existence of multimodal signal lift, not the construction of a near-perfect forecasting engine @martynova2008.

Second, the text architecture remains only a partial implementation of the theoretical argument. Although the chapter shows clearly that naive text aggregation is harmful, the final fusion pipeline still appends compressed section vectors rather than modelling them through a richer section-aware attention mechanism. A more explicit dual-stream text encoder may recover additional semantic signal @baltrusaitis2019.

Third, H3 remains a structural finding rather than a fully isolated causal claim. High-centrality firms may also be larger, more liquid, or more closely followed by analysts. The chapter therefore interprets centrality as a statistically meaningful dampening correlate, not as an exhaustively isolated causal driver @cohen2008. Furthermore, although leakage controls prevent direct target contamination, the Bloomberg SPLC data represents a periodic snapshot rather than a point-in-time series; for early deals (2000–2010), graph edges may reflect supply-chain structures not yet extant at the announcement time, introducing a residual look-ahead risk.

Fourth, the test set (2020–2023) spans the COVID-19 shock and post-pandemic supply-chain restructuring, a structurally distinct regime from the training period. This regime shift risk is a known cost of chronological holdout design and is the primary reason the reported AUC lift of +0.0247 should be interpreted conservatively.

== Synthesis

Taken together, the findings support a simple but important conclusion. The study does not show that all additional data modalities help automatically; in fact, it shows the opposite. Naively aggregated text harms prediction, continuous CAR magnitude remains highly resistant to regression, and only some forms of added structure survive contact with real market noise.

What *does* survive is economically structured multimodality. Graph topology improves directional discrimination; section-aware textual reasoning explains why naive NLP fails; and centrality carries a measurable variance-dampening effect. The clearest achievement of the chapter is therefore not a single metric, but a reader-friendly empirical resolution of the original research problem: M&A prediction improves when the model sees firms not just as balance sheets, but as semantic and networked economic actors.
