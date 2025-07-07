# Inter-Flow Latency-Induced Service Degradation Detection

This is the supporting page for the "*On the Feasibility of Inter-Flow Service Degradation Detection*" research project.  
The study builds on the findings of [ServDeg-Dataset](https://doi.org/10.1109/ACCESS.2024.3456588) and [Intra-Flow Service Degradation](https://doi.org/10.23919/CNSM62983.2024.10814309) and extends the analysis from intra-flow to inter-flow relationships in service degradation (SD). This repository includes all necessary code, models, and metrics used throughout the study.

## Repository Structure

The repository is organized to support reproducibility and clarity across preprocessing, feature engineering, model training, and evaluation.

### Notebooks

1. **[Inter_Flow_Prediction_Models.ipynb](Inter_Flow_Prediction_Models.ipynb)**  
   *Title: Prediction Model Results and Interpretations for Inter-Flow SD Behavior*  
   This notebook presents classification and regression results for SD prediction based on inter-flow features, compares multiple ML models (Logistic Regression, XGBoost, MLP), and interprets their predictive behavior.

2. **[Inter-Flow EDA (m=10).ipynb](Inter-Flow%20EDA%20(m=10).ipynb)**  
   *Title: Exploratory Analysis of Inter-Flow Relationships at m=10*  
   Visualizations and descriptive statistics for overlapping flows, SD coverage, time shifts, and correlations by application and location.

3. **[Inter-Flow EDA (m in {5,10}).ipynb](Inter-Flow%20EDA%20(m%20in%20%7B5%2C%2010%7D).ipynb)**  
   *Title: Cross-m Analysis of Inter-Flow Characteristics*  
   Similar to the above, but compares behaviors for multiple horizontal split points (`m = 5, 10`).

---

### Python Scripts

1. **[inter_flow_statistics.py](inter_flow_statistics.py)**  
   Computes detailed statistics about overlaps between observable (O) parts of different flows and their relation to SDs in the non-observable (NO) parts.  
   - Calculates SD coverage, center distances, app-category overlaps, and relative timing features.
   - Produces compressed `.parquet` results for both full dataset and filtered flows used in training.

2. **[create_inter_flow_sets.py](create_inter_flow_sets.py)**  
   Prepares inter-flow features by identifying overlapping O parts from preceding flows.
   - Extracts overlap-based features such as relative timing and application distribution.
   - Outputs merged input (`X`) and target (`y`) files for multiple prediction tasks (classification + 4 regression targets).
   - Scales features and filters out non-informative columns.

3. **[inter_flow_models.py](inter_flow_models.py)**  
   Trains and evaluates ML models for inter-flow SD prediction.
   - Classification: Logistic Regression, XGBoost, MLP.
   - Regression: Ridge Regression, XGBoost, MLP.
   - Uses `GridSearchCV` for parameter tuning.
   - Saves predictions, probabilities, performance metrics, and best parameters per split value `M`.

4. **[create_with_SD_REG_metrics.py](create_with_SD_REG_metrics.py)**  
   Re-evaluates regression performance on a filtered subset of the test set that contains only NO parts with SDs, avoiding class imbalance bias.

---

### Configuration

- **[setup.json](setup.json)**  
  Contains file paths
---

### Data

These directories are created upon running the scripts / notebooks with the following contents:

- **[inter_results/M<M>](inter_results/)**:  
  Contains performance results and best model settings for each threshold `M`.  
  - `CLASS_metrics.csv`, `CLASS_best_params.csv`, `CLASS_predictions.parquet`, `CLASS_probs.parquet`  
  - `REG_{task}_metrics.csv`, `REG_{task}_best_params.csv`, `REG_{task}_predictions.parquet`  
  - `{task}` ∈ {`count`, `max_len`, `max_start`, `max_end`}

- **[train_test/inter/M<M>](train_test/inter/M*/)**:  
  Prepared datasets for training and testing.
  - `X_train.parquet`, `X_test.parquet` (original features)
  - `X_train_scaled.parquet`, `X_test_scaled.parquet` (normalized features)
  - `y_{stage}_{type}.parquet` (targets for classification and regression tasks)

- **[inter_stats_used/](inter_stats_used/)**:  
  Overlap statistics used for constructing features and analysis. Organized by day and location.

---

## Prediction Tasks

The targets predicted for each flow's NO part are:
- **Classification**: Whether the NO part contains any SDs.
- **Regression**:
  - Count of SD events.
  - Length of the longest SD event.
  - Start and end indices of the longest SD (relative to O part start).

---

## Documentation

The theoretical background, modeling methodology, and interpretation of results are provided in the accompanying research documentation. For extended methodology on observable vs non-observable segmentation, see [ServDeg-Dataset](https://github.com/FlowFrontiers/ServDeg-Dataset).

---
