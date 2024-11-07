# An AI Approach for Predicting the Active Surface of Deep Drawing Tools in Try-Out

> **Abstract:**
> *The tool try-out process of deep drawing tools is often tedious, iterative, and manual, leading to suboptimal results and prolonged ramp-up phases. Toolmakers first capture spotting patterns of the tool surfaces and then manually remove material based on these patterns. A key challenge is the complex interaction between the tools, the sheet metal, and the press, making it hard to predict issues that may propagate to later steps in the tool try-out process. To address this, a data-driven AI approach is proposed. Using an encoder-decoder model, it predicts the tool active surface in contact from the pressure distribution of deep drawing tools. It is trained on simulated pressure distributions, which serve as a quantitative representation of the spotting patterns. The approach is benchmarked against image-to-image translation methods such as U-Net and Pix2Pix.*
# Content
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)

# Installation

```
conda create -n active-surface-prediction
conda activate active-surface-prediction
conda env update -n active-surface-prediction --file environment.yml
```

# Training

```
python train.py
```
To recalculate the summary statistics for the training data after changing the initial seed, please run: `python calc_stats.py`.

# Evaluation

```
python eval.py
```