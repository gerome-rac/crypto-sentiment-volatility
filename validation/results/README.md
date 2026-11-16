# Hand validation: result tables

A stratified random sample of 300 posts (20 for each combination of platform and coin) was labeled by hand for emotion
(the seven EmTract classes) and tone (positive, neutral, negative). The post texts are not part of this repository.

- `hand_labels.csv`: one row per post. `post_code` is the code of the post in the sample (not a platform id);
  `coder_A_*` and `coder_B_*` are the labels of the two coders, `reference_*` the reference label fixed by the author.
  An empty cell means that there is no label.
- `agreement_of_the_two_coders.csv`: share of equal labels and Cohen's kappa on the posts that both coders labeled, with
  the reading used in the thesis (kappa of 0.61 or more: reliable; 0.41 to 0.60: tentative; 0.40 or less: failed).
- `models_against_reference_labels.csv`: EmTract's top class and probabilities and FinBERT's top class and polarity score
  against the reference labels. `weights`: `cell_equal` gives every combination of platform and coin the same weight
  (the figures of the thesis), `post` weights by posts, `unweighted` describes the sample only. Intervals: bootstrap
  within the sampling strata, 2,000 draws, 95%.

Thesis: Section 4.4 and Appendix B.
