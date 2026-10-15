# Method

Equations of what the code computes. Symbols: $x$ is a $512 \times 512$ RGB frame with values in $[0,1]$, $y \in \{0,1,2,3,4\}$ the DR grade (ICDR scale), $K = 5$ grades and $K-1 = 4$ thresholds, $\sigma$ the logistic function.

## 1. Pipeline

```
raw image ─ Stage 1 quality (EfficientNet-B0) ───────────────────────────────┐
   │                                                                         │
   └─ Stage 2 gamma + CLAHE ─ fundus box (crop arm) ─ resize 512 ─ grader frame
            │                                              │
            ├─ Stage 3 vessels (LWNet)                      └─ grader (3 seeds) ─ grade, uncertainty, explanations
            └─ Stage 4 lesions (U-Net) ──────── supporting views, lesion overlap, written summary ◄┘
```

The grade comes from the RGB grader alone. The vessel and lesion maps are supporting views.

## 2. Preprocessing and frames

Stage 2 applies a gamma curve and then contrast-limited adaptive histogram equalisation (CLAHE) to the lightness channel only:

$$x' = 255\left(\frac{x}{255}\right)^{1/\gamma}, \quad \gamma = 1.2; \qquad \text{CLAHE: clip limit } 2.0,\ 8 \times 8 \text{ tiles, on } L \text{ of LAB.}$$

The fundus box is found on the raw image $g$ (grey values): a pixel is lit when $g > 12$; a row (column) belongs to the fundus when more than 2 % of its pixels are lit; the box spans the first to last such row and column, and the whole image is kept when a side is shorter than 10 % of the image side. The crop arm cuts the Stage-2 image to this box; both arms then resize to $512 \times 512$ (bilinear, anti-aliased, aspect ratio not kept).

## 3. Grader and CORN

The grader is ConvNeXt-Tiny (ImageNet initialisation) on $(x - \mu)/s$ with the ImageNet mean and standard deviation, followed by global average pooling, LayerNorm and a linear layer to four logits $z_0, \dots, z_3$.

**CORN** (conditional ordinal regression for neural networks) models $\sigma(z_k) = P(y > k \mid y \ge k)$. The cumulative probabilities are products,

$$P(y > k) = \prod_{i=0}^{k} \sigma(z_i), \qquad \text{non-increasing in } k,$$

the decoded grade counts those above one half,

$$\hat y = \sum_{k=0}^{3} \mathbb{1}\left[P(y>k) > 0.5\right],$$

and the class probabilities are differences: $P(y=0) = 1 - P(y>0)$, $P(y=k) = P(y>k-1) - P(y>k)$ for $k = 1,2,3$, $P(y=4) = P(y>3)$. The expected grade is $\mathbb{E}[y] = \sum_k P(y>k)$.

**Weighted CORN loss.** Image $n$ with grade $y_n$ contributes the binary cross-entropy of every threshold $k \le \min(y_n, 3)$ against $\mathbb{1}[y_n > k]$ (thresholds above its grade are not defined for it), scaled by the weight of its grade:

$$\mathcal{L} = \frac{\sum_n w_{y_n} \sum_{k=0}^{\min(y_n,3)} \mathrm{BCE}\!\left(z_{nk},\ \mathbb{1}[y_n>k]\right)}{\sum_n \left(\min(y_n,3)+1\right)}.$$

The denominator counts pairs without weights. With $p_c$ the grade distribution of an epoch, $w_c \propto p_c^{-1/2}$ normalised so that $\sum_c p_c w_c = 1$, i.e. $w_c = p_c^{-1/2} / \sum_j p_j^{1/2}$; for the joint APTOS + EyePACS epoch these are $0.6456, 1.8421, 1.1954, 2.8293, 2.5958$.

**Optimisation.** AdamW with learning rate $10^{-4}$ and decoupled weight decay $0.05$ (not applied to one-dimensional parameters), batch size 16. The monitored value after every epoch is $\tfrac12(\kappa_{\text{APTOS val}} + \kappa_{\text{EyePACS val}})$ with $\kappa$ the quadratic weighted kappa below.

## 4. Quadratic weighted kappa

With $O_{ij}$ the confusion counts (row: reference, column: prediction), $E_{ij} = (\sum_j O_{ij})(\sum_i O_{ij})/N$ and $w_{ij} = (i-j)^2/(K-1)^2$,

$$\kappa = 1 - \frac{\sum_{ij} w_{ij} O_{ij}}{\sum_{ij} w_{ij} E_{ij}}.$$

AUROC at cut $c \in \{1,2,3,4\}$ scores "grade $\ge c$" by $P(y > c-1)$; "mean cut AUROC" averages the four.

## 5. Stage 4: lesion segmentation

A U-Net with an SE-ResNet-101 encoder gives four probability maps $s_k = \sigma(z_k)$ (MA, HE, EX, SE). Inference runs on the whole Stage-2 image resized to $1536 \times 1536$; each map is reduced to $512 \times 512$ by the mean and by the maximum over non-overlapping $3 \times 3$ blocks, stored as `uint8` $= \mathrm{round}(255 s)$ and interleaved per class (`MA:mean, MA:max, HE:mean, ...`).

Training loss, per class $k$ over a batch $B$ of $512 \times 512$ patches, with field-of-view mask $f$ and label $y$:

$$\ell_k = \frac{1}{|B|}\sum_{i \in B} \frac{\sum_p f\,\big[-w^{+}_k\, y \log s_k - (1-y)\log(1-s_k)\big]}{\sum_p f} \;+\; \left[1 - \frac{2\sum f\, s_k\, y + 1}{\sum f\,(s_k + y) + 1}\right],$$

$w^+_k = \mathrm{clip}\big(\sqrt{N^-_k / N^+_k},\ 1,\ 20\big)$ from the training pixel counts, and $\mathcal{L} = \tfrac1K \sum_k \ell_k$ (the Dice term is pooled over the batch). The exported weights are an exponential moving average, $\bar\theta \leftarrow 0.999\,\bar\theta + 0.001\,\theta$.

## 6. Uncertainty

For the three seeds $s = 1,2,3$ the ensemble averages the cumulative probabilities, $\bar P(y>k) = \tfrac13 \sum_s P_s(y>k)$, decodes with the rule above, and derives three scores per image (higher is less certain):

$$H = -\sum_{c} \bar p_c \log \bar p_c \ \ (\text{nats, at most } \ln 5 \approx 1.61), \qquad
D = \mathrm{SD}_s\big(\mathbb{E}_s[y]\big), \qquad
M = -\min_k \left|\bar P(y>k) - \tfrac12\right|.$$

**Selective prediction.** Images are ranked by a score and the least certain are referred; at coverage $c$ the $\mathrm{round}(cN)$ most certain images are kept and $\kappa$ is measured on them. Random referral (200 random orders, seed 20261012) is the reference. **Error detection** is the AUROC of the score for "the ensemble grade is wrong". The model has no dropout layer, so this is deep-ensemble uncertainty, not Monte Carlo dropout.

**Test-time augmentation.** The 8 flips / quarter turns $T_t$ of the dihedral group are applied to the frame; the spread is $\mathrm{SD}_t(\mathbb{E}[y \mid T_t x])$ and the grade is flagged when the decoded grades differ between views.

**Reliability wording of the demo** (a wording aid, not a validated method). "Review recommended" if any of: Stage 1 rates the image Reject; the three seeds decode to different grades; a test-time-augmentation view changes the decoded grade of a seed; $H$ exceeds the 80th percentile of $H$ on the pooled APTOS and EyePACS validation sets. The statement quotes the exact and within-one-grade rates of the validation images on the same side of that threshold. The validation sets also selected the checkpoints, so those rates are optimistic, and images from other cameras or clinics can be less reliable.

## 7. Explanations

All methods explain the expected grade $s(x) = \mathbb{E}[y]$.

**Grad-CAM.** With $A^k$ the $16 \times 16$ feature maps before pooling,
$\alpha_k = \frac{1}{Z}\sum_{ij} \partial s / \partial A^k_{ij}$ and $L = \mathrm{ReLU}\big(\sum_k \alpha_k A^k\big)$, divided by its maximum and resized to the frame.

**Integrated Gradients** from the black image $x'$ with $m$ midpoint steps:

$$\mathrm{IG}_i(x) = (x_i - x'_i)\,\frac1m \sum_{j=1}^{m} \frac{\partial s}{\partial x_i}\!\left(x' + \tfrac{j - 1/2}{m}(x - x')\right), \qquad \sum_i \mathrm{IG}_i \approx s(x) - s(x').$$

The reported gap $|\sum_i \mathrm{IG}_i - (s(x) - s(x'))|$ is the completeness check; the map is $\sum_c |\mathrm{IG}|$ divided by its maximum. 32 steps in the localisation study, 16 in the interactive demo.

**Occlusion.** The drop $s(x) - s(x \text{ with a } 64 \times 64 \text{ patch set to black})$ at stride 32 (64 in the demo), averaged over the patches that cover a pixel; the positive part is shown.

**Kernel SHAP** over $M = 50$ SLIC superpixels: Shapley values $\phi$ of the coalition value $v(S) = s(x_S)$ (excluded superpixels black) by weighted least squares with the Shapley kernel $\pi(z) = \dfrac{M-1}{\binom{M}{|z|}\,|z|\,(M-|z|)}$ over 512 coalitions (paired with complements), the empty and full coalitions enforced so that $\phi_0 + \sum_j \phi_j = s(x)$. Figures only.

**Localisation against expert masks.** For a map $h$ and the union $\Lambda$ of an image's lesion masks: pointing game $\mathbb{1}[\arg\max h \in \Lambda]$; precision of the top 10 % of pixels, $|\mathrm{top}_{10\%}(h) \cap \Lambda| / |\mathrm{top}_{10\%}(h)|$; AUROC of $h$ as a lesion score; the chance level of the first two is the lesion share of the frame $|\Lambda|/|\text{frame}|$ and of the AUROC 0.5. Overlap describes where a map is high; it is not evidence that the model detects lesions.

## 8. Statistics

**Contrasts.** For models $a$ and $b$ scored on the same images and seed-labelled pairs $s \in \{42, 123, 2026\}$: $\Delta = \tfrac13\sum_s \big[\kappa(a_s) - \kappa(b_s)\big]$. **Bootstrap:** 2,000 resamples; within each grade, images are drawn with replacement (so every resample keeps the grade counts), the same resample is applied to every model, and the 2.5th and 97.5th percentiles of $\Delta$ give the 95 % interval (seed 20260927). A contrast is read as supported when the interval lies wholly above zero, reversed when wholly below, and otherwise as no difference shown; no equivalence or non-inferiority statement is made, and intervals are not adjusted for multiplicity.
