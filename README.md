# ADNI Research Project

## Overview
This project focuses on **early diagnosis of Alzheimer's Disease (AD)** using **brain MRI images** and **machine learning techniques**. It is based on the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** dataset and explores **deep learning models** to classify subjects into three categories: **Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN)**.

The study employs **Residual Networks (ResNet)** for feature extraction and a **three-layer neural network** for classification. Our approach improves diagnostic accuracy and provides insights into early disease detection.

## Data Source
- **Dataset**: [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/)
- **MRI Type**: Axial T2*-weighted MRI scans
- **Labels**: AD, MCI, CN
- **Data Size**: 1289 subjects from ADNI3 phase

## Objectives
- Develop a **deep learning-based classification model** for early-stage Alzheimer's diagnosis.
- Implement **ResNet50** for MRI feature extraction and apply **neural networks with Softmax** for classification.
- Conduct **cross-validation** to evaluate model robustness.
- Provide interpretable insights into **MRI-based AD diagnosis**.

## Methodology
1. **Data Preprocessing**
   - Extract **Axial T2* MRI images** from ADNI3.
   - Normalize and resize images to **256×256 pixels**.
   - Apply **Gaussian-weighted feature selection** for dimensionality reduction.

2. **Feature Extraction**
   - Use **ResNet50** as a **feature extractor**, outputting a **2048-dimensional feature vector**.
   - Implement an **encoder to reduce feature dimensionality**.

3. **Classification Model**
   - Use a **three-layer neural network**:
     - **Two hidden layers** (1024 and 512 neurons).
     - **Softmax output layer** for classification.
   - Train using **categorical cross-entropy loss** and **Adam optimizer**.

4. **Evaluation Metrics**
   - **Accuracy**: 83.24% in cross-validation.
   - **ROC Curve & AUC**: Assess classification confidence.
   - **Confusion Matrix**: Analyze misclassifications.

## Implementation
- **Programming Language**: Python
- **Libraries & Tools**:
  - `TensorFlow`, `PyTorch`
  - `scikit-learn`
  - `OpenCV`
  - `NumPy`, `Pandas`
  - `Matplotlib`, `Seaborn`

## Results
- **83.24% accuracy** in cross-validation, outperforming baseline models.
- **Enhanced differentiation between AD, MCI, and CN**, offering **better clinical interpretability**.
- **ResNet-based feature extraction** proved effective in handling high-dimensional MRI data.

## Repository Structure
```
├── data/                 # MRI dataset and preprocessing scripts
├── models/               # ResNet feature extractor & classifier implementation
├── notebooks/            # Jupyter Notebooks for experiments
├── results/              # Evaluation metrics, visualizations
├── README.md             # Project documentation
└── requirements.txt      # Dependencies list
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/felixazzzz/ADNI-research.git
   cd ADNI-research
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the classification model:
   ```bash
   python src/train_model.py
   ```

## Future Work
- Improve **interpretability** by analyzing **MRI feature importance**.
- Expand dataset to include **other MRI modalities**.
- Optimize **classification performance on MCI cases** to enhance early diagnosis.

## References
- **ADNI Dataset**: [ADNI Official Website](https://adni.loni.usc.edu/)
- **Related Work**: 
  - Janghel & Rathore: 99.95% accuracy on fMRI dataset.
  - Sarraf & Tofighi: 98.84% accuracy on binary MRI classification.
  - Our study: **83.24% accuracy on multi-class MRI classification**, with **higher clinical relevance**.

## Contributors
- **Jingze Zhang** - Researcher
- **Yanming Shen** - Co-researcher
- **Peng Sun** - Research Mentor
- **Ming-chun Huang** - Research Mentor
  
## License
This project is for educational and research purposes only.
