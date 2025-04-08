# Application of eXplainable AI (XAI) Methods

**Yawen Xiao**  
Email: [grace.xiao@mail.utoronto.ca](mailto:grace.xiao@mail.utoronto.ca)

## Dataset

The dataset and additional documentation are available at:  
🔗 [CIC EV Charger Attack Dataset 2024 (CICEVSE2024)](https://www.unb.ca/cic/datasets/evse-dataset-2024.html)

### Overview

This dataset supports cybersecurity research in electric vehicle charging systems by enabling **binary classification** of **benign vs. attack** behavior. It captures measurable changes in power consumption under both normal and compromised conditions, allowing machine learning models to detect patterns indicative of malicious activity.

### Background

With increasing connectivity in electric vehicle (EV) charging infrastructure, there is a rising risk of cyberattacks that can disrupt services or compromise user data. Changes in power consumption from internal components of the charging station can serve as indicators of malicious activity. Monitoring this data offers a **non-intrusive and efficient method** for anomaly detection and real-time threat response.

### Dataset Description – Power Consumption of EVSE-B

Data is collected from a Raspberry Pi embedded in an EV charging station (EVSE-B). The Raspberry Pi logs power consumption data under both **normal operation** and during various **simulated attack scenarios**.

### Features

- `Time` : Timestamp of the reading  
- `Shunt_voltage (mV)` : Voltage drop across a shunt resistor (I2C wattmeter)  
- `Bus_voltage` : Supplied DC voltage  
- `Current_mA` : Electrical current drawn by EVSE-B  
- `Power_mw` : Total power consumption of EVSE-B  

### Target

- `0` = Benign  
- `1` = Attack  

### Use Case

This dataset is designed for training **binary classification models** to detect cyberattacks based on power usage patterns. Such models can improve the **security and reliability** of EV charging infrastructure by enabling early detection of abnormal behavior.

## Handling the Time Column in Modeling

The `time` column in the dataset provides the timestamp of each power consumption reading. While it clearly defines the temporal order of data collection, it is unclear whether time itself is a strong sequential feature — that is, whether behavioral changes in the EVSE are inherently tied to the progression of time. Due to this uncertainty, I decided to explore both modeling strategies: (1) treating time as an index to apply time-series modeling, and (2) using time as a regular feature within the feature set for classification tasks.

### Comparison in Feature Selection

Both approaches were evaluated and compared in the **Feature Selection** phase under two configurations: `XAI-IoT` and `XAI-IoT - time series`. This dual-path exploration helps determine the best modeling goal — whether for **real-world deployment or real-time detection**, which would favor time-series modeling, or for **feature importance analysis and behavior classification**, where using time as a feature may be more effective. These experiments help clarify whether temporal evolution offers predictive power or simply coincides with certain attack intervals.

### Challenges with Time-Series Approach

Time-series modeling requires strict preservation of temporal order during data splitting. However, this introduces a critical issue: in the current dataset, the **last time period (used by test set)** contains **only attack samples (label = 1)**. Consequently, the test set lacks class diversity, and the confusion matrix confirms that all ground truth labels in this partition are 1. This situation not only biases evaluation metrics but also undermines the ability of the model to generalize to benign behavior in future unseen data. Such skewed class distribution, if unaddressed, may lead to misleading model performance, overfitting to attack patterns, and poor interpretability of true model behavior.

### Final Decision and Adjustment

Given the imbalance and risks introduced by the time-series split, I decided to stop time-based indexing and instead proceed by **using time as a feature**. To support future analysis, I transformed the original timestamp into a **binary categorical feature**: `daytime = 1` and `nighttime = 0`, based on hour-of-day. This transformation simplifies the temporal information while retaining potential behavioral shifts linked to daily operational cycles of the EVSE.

## Handling Unbalanced Data

The labels in the dataset are highly imbalanced.
- **Label 0 (benign)**: 14,363 samples  
- **Label 1 (attack)**: 100,935 samples  

To address this, I randomly reduced the samples with label 1 to match the count of label 0, resulting in a balanced dataset:  
- **Label 0 (benign)**: 14,363 samples  
- **Label 1 (attack)**: 14,363 samples  

This **undersampling** technique helps deal with imbalanced data by manually reducing the data samples from the overrepresented class. While undersampling leads to a loss of information due to the discarded majority-class samples, I still retained sufficient data for training after balancing.

### Strengths:
1. **Balanced Dataset**: Helps prevent model bias toward the majority class, which is crucial for highly imbalanced datasets like this one.
2. **Faster Training**: Smaller datasets reduce computational requirements and speed up training.
3. **Simplicity**: Easy to implement without requiring changes to the model or specialized techniques.

### Weaknesses:
1. **Loss of Information**: Discarding majority-class samples may reduce the diversity and richness of training data.
2. **Risk of Overfitting**: With fewer samples, the model might overfit to the reduced dataset, particularly on the majority class.
3. **Sample Representativeness**: Random sampling might not fully preserve the characteristics of the original data distribution.

## Feature Selection

A Random Forest model is used to rank the importance of features. The results, including feature rankings, confusion matrix, and ROC curve on the test set, are visualized in the file **'Feature Engineering.ipynb'**.

Top N features are selected for the next training phase. To identify the best N, I save subsets of features with different importance thresholds:

### Case 1: 'time' as a feature:
- All features (5 features)
- Importance > 0.1 (4 features)
- Importance > 0.2 (3 features)

### Case 2: 'time' as an index (time-series):
- All features (4 features)
- Importance > 0.2 (3 features)

## Model Training - Multilayer Perceptron (MLP)

### Hyperparameter Tuning

- **Learning Rates**: 0.001, 0.01
- **Dropout Rates**: 0.2, 0.4
- **Hidden Units (Layer 1)**: 32, 64, 128
- **Hidden Units (Layer 2)**: 16, 32
- **Batch Sizes**: 16, 32, 64

### Best Model (All Features)

- **Learning Rate**: 0.001
- **Dropout Rate**: 0.2
- **Hidden Units (Layer 1)**: 128
- **Hidden Units (Layer 2)**: 32
- **Batch Size**: 64
- **Highest Validation Accuracy**: 0.9061

Additional accuracy and loss data can be found in different `MLP.ipynb` files. Since the model trained on all features achieved the highest validation accuracy and true positive rate, it is used for the subsequent XAI interpretation.

## Explainable Artificial Intelligence (XAI)

XAI methods, including **SHAP** and **LIME**, are applied to the best model trained with all features for a comprehensive analysis. To experiment with different models, change the code as follows: 
```python
feature_set = "all_features"
