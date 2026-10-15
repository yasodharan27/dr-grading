# References

Citations only (no PDFs). DOIs and arXiv identifiers were checked against Crossref and the arXiv API.

## Models and methods

- **CORN.** Shi, X., Cao, W., Raschka, S. *Deep neural networks for rank-consistent ordinal regression based on
  conditional probabilities.* Pattern Analysis and Applications 26, 2023. doi:10.1007/s10044-023-01181-9
- **ConvNeXt.** Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S. *A ConvNet for the 2020s.* CVPR 2022.
  doi:10.1109/CVPR52688.2022.01167
- **EfficientNet.** Tan, M., Le, Q. V. *EfficientNet: Rethinking model scaling for convolutional neural networks.* ICML
  2019. arXiv:1905.11946
- **U-Net.** Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional networks for biomedical image segmentation.*
  MICCAI 2015. doi:10.1007/978-3-319-24574-4_28
- **Squeeze-and-Excitation (SE-ResNet).** Hu, J., Shen, L., Sun, G. *Squeeze-and-excitation networks.* CVPR 2018.
  doi:10.1109/CVPR.2018.00745
- **LWNet (Stage 3).** Galdran, A., Anjos, A., Dolz, J., Chakor, H., Lombaert, H., Ben Ayed, I. *State-of-the-art retinal
  vessel segmentation with minimalistic models.* Scientific Reports 12, 6174, 2022. doi:10.1038/s41598-022-09675-y
  (preprint: *The Little W-Net That Could: State-of-the-art retinal vessel segmentation with minimalistic models*,
  arXiv:2009.01907). Code and checkpoint: https://github.com/agaldran/lwnet, MIT licence.
- **segmentation_models_pytorch.** Iakubovskii, P. *Segmentation Models PyTorch.* https://github.com/qubvel-org/segmentation_models.pytorch
- **ImageNet.** Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L. *ImageNet: A large-scale hierarchical
  image database.* CVPR 2009. doi:10.1109/CVPR.2009.5206848
- **AdamW.** Loshchilov, I., Hutter, F. *Decoupled weight decay regularization.* ICLR 2019. arXiv:1711.05101
- **Mixed-precision training.** Micikevicius, P., et al. *Mixed precision training.* ICLR 2018. arXiv:1710.03740
- **CLAHE.** Zuiderveld, K. *Contrast limited adaptive histogram equalization.* In: Graphics Gems IV, Academic Press,
  1994, pp. 474-485. ISBN 0-12-336155-9.

## Uncertainty and statistics

- **Deep ensembles.** Lakshminarayanan, B., Pritzel, A., Blundell, C. *Simple and scalable predictive uncertainty
  estimation using deep ensembles.* NeurIPS 2017. arXiv:1612.01474
- **Selective classification.** Geifman, Y., El-Yaniv, R. *Selective classification for deep neural networks.* NeurIPS
  2017. arXiv:1705.08500
- **Weighted kappa.** Cohen, J. *Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial
  credit.* Psychological Bulletin 70, 1968. doi:10.1037/h0026256
- **Bootstrap.** Efron, B. *Bootstrap methods: Another look at the jackknife.* The Annals of Statistics 7, 1979.
  doi:10.1214/aos/1176344552

## Explanations

- **Grad-CAM.** Selvaraju, R. R., et al. *Grad-CAM: Visual explanations from deep networks via gradient-based
  localization.* ICCV 2017. doi:10.1109/ICCV.2017.74
- **Integrated Gradients.** Sundararajan, M., Taly, A., Yan, Q. *Axiomatic attribution for deep networks.* ICML 2017.
  arXiv:1703.01365
- **Occlusion.** Zeiler, M. D., Fergus, R. *Visualizing and understanding convolutional networks.* ECCV 2014.
  doi:10.1007/978-3-319-10590-1_53
- **SHAP / Kernel SHAP.** Lundberg, S. M., Lee, S.-I. *A unified approach to interpreting model predictions.* NeurIPS
  2017. arXiv:1705.07874
- **SLIC superpixels.** Achanta, R., et al. *SLIC superpixels compared to state-of-the-art superpixel methods.* IEEE
  TPAMI 34, 2012. doi:10.1109/TPAMI.2012.120

## Grading scale

- **ICDR scale.** Wilkinson, C. P., et al. *Proposed international clinical diabetic retinopathy and diabetic macular
  edema disease severity scales.* Ophthalmology 110, 2003. doi:10.1016/S0161-6420(03)00475-5

## Datasets

Dataset terms apply to anything trained on them; see `data/README.md`.

- **APTOS 2019 Blindness Detection.** Asia Pacific Tele-Ophthalmology Society, Kaggle competition, 2019.
  https://www.kaggle.com/c/aptos2019-blindness-detection
- **EyePACS (Kaggle Diabetic Retinopathy Detection).** EyePACS, Kaggle competition, 2015.
  https://www.kaggle.com/c/diabetic-retinopathy-detection. Cuadros, J., Bresnick, G. *EyePACS: An adaptable
  telemedicine system for diabetic retinopathy screening.* Journal of Diabetes Science and Technology 3, 2009.
  doi:10.1177/193229680900300315
- **EyeQ.** Fu, H., Wang, B., Shen, J., Cui, S., Xu, Y., Liu, J., Shao, L. *Evaluation of retinal image quality assessment
  networks in different color-spaces.* MICCAI 2019. doi:10.1007/978-3-030-32239-7_6
- **IDRiD.** Porwal, P., et al. *IDRiD: Diabetic retinopathy - segmentation and grading challenge.* Medical Image
  Analysis 59, 2020. doi:10.1016/j.media.2019.101561 (data: CC BY 4.0)
- **DDR.** Li, T., Gao, Y., Wang, K., Guo, S., Liu, H., Kang, H. *Diagnostic assessment of deep learning algorithms for
  diabetic retinopathy screening.* Information Sciences 501, 2019. doi:10.1016/j.ins.2019.06.011
- **TJDR.** Mao, J., Ma, X., Bi, Y., Zhang, R. *TJDR: A high-quality diabetic retinopathy pixel-level annotation
  dataset.* arXiv:2312.15389, 2023. https://github.com/NekoPii/TJDR
