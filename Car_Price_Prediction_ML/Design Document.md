# ML System Design: Used Car Price Prediction

## 1. Project Overview & Problem Definition

- **Background:** The Indian used car market is experiencing significant growth, driven by factors like better value proposition compared to new cars (which depreciate quickly) and the emergence of more organized, reliable retailers enhancing **product quality** assurance. This expanding market involves a high volume of transactions between individual buyers and sellers, as well as dealerships.

### Problem Statement

- **Core Problem:** Prospective buyers and sellers of used cars in India face significant difficulty in accurately determining a fair market price. The core pain point stems from market complexity and information asymmetry. This is due to:
  - A vast variety of car brands, models, and variants.
  - Frequent model updates and facelifts (often yearly or bi-annually) make direct comparisons difficult.
  - Vehicle condition, mileage, age, location, and specific features heavily influence price, but assessing their combined impact is challenging for non-experts.
- **Impact of problem:** This lack of transparent and reliable pricing information leads to:
  - **Inefficiency:** Buyers and sellers spend considerable time manually researching prices across various platforms.
  - **Inaccuracy:** Manual estimations are often imprecise, leading to potential financial loss (sellers underselling, buyers overpaying).
  - **Friction:** Price negotiation can be contentious and prolonged due to the lack of a trusted benchmark.
  - Solving this problem by providing an accessible, data-driven price prediction tool can create transparency, efficiency, and trust in used car transactions.
- **Existing Solution and Limitations:** The current common approach is a manual process where users visit multiple used car websites (e.g., online classifieds, dealership sites) and search for comparable vehicles. This is time-consuming and inefficient.

### Goals & Scope

- **Goals:**
  - **(Offline Model Performance)** Achieve a Mean Absolute Percentage Error (MAPE) below 15% (or a similarly defined regression metric like R² > 0.85 - _metric needs finalization_) on cross-validation datasets for price prediction.
  - **(Online API Performance)** Serve 95% of prediction requests via the FastAPI endpoint in under 500ms.
  - **(Online System Reliability)** Maintain high availability (e.g., 99.9% uptime) for the prediction API and frontend application.
  - **(Functional)** Deliver a simple React web application allowing users to input car details (e.g., make, model, year, mileage, location) and receive a predicted price.
  - **(Functional)** Deploy the prediction model via a FastAPI endpoint hosted on AWS.
- **Antigoals:**
  - V1 will **not** predict future price trends or depreciation rates.
  - V1 will **not** provide explanations or confidence intervals for the price prediction.
  - V1 will **not** initially support niche vehicle types (e.g., classic cars, commercial vehicles, imported CBU units).
  - V1 will **not** support price prediction for regions outside major Indian metropolitan areas.
- **Scope:**
  - **Geographical:** Limited to major Indian metropolitan areas (e.g., Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, Kolkata - _exact list TBD_).
  - **Temporal Relevance:** The model will be retrained, and data potentially refreshed, on a **weekly basis** to capture recent market dynamics.
  - **Platform:** Development, training, and deployment will primarily utilize **AWS SageMaker** and associated AWS services.

---

## 2. Data Strategy

### Data Sources

#### Source 1: Web Scraping

Publicly available data from online using web scraping techniques from multiple source websites

- **Assessment:**
  - _Availability_: Public, but requires robust web scraping.
  - _Freshness_: Scraping frequently - weekly.
  - _Volume_: Potentially lakhs of listings nationally over time.
  - _Quality_: Core features are reliable with less missing values. Since multiple sources are used, these data needs to be combined in proper manner
  - _Cost_:
    - A minimal compute instance (EC2) with runtime of 3 hours can fetch all the data.
    - Only about 20Mb per week thus we can use free tier S3 storage.
    - IP block protection from proxy service provider at 8$ per GB of data.
  - _Privacy/Ethics_:
    - Respect `robots.txt`.
    - Avoid PII.
    - Adhere to website Terms of Service where possible.

### Data Acquisition & Processing

- **Process Flow:** Batch scraping (**Dataset_Gathering**).
- **ETL Pipeline:**
  - _Extraction_: Scheduled scraping jobs (e.g., AWS Lambda triggered by EventBridge, or SageMaker Processing Jobs) using Python libraries (Scrapy, BeautifulSoup, Selenium if needed). Store raw HTML/JSON snapshots in S3 (versioned).
  - _Transformation_: Parse raw data, extract key fields (price, make, model, year, mileage, location, VIN if available, etc.). Clean data (handle missing values, standardize units/categories - e.g., 'Ford F-150' vs 'F150'). Normalize location data. Use **SageMaker Processing Jobs** (Spark or Pandas/Dask).
  - _Loading_: Load cleaned, structured data into S3 (e.g., Parquet format). Load key features into **SageMaker Feature Store** for training/inference consistency.
- **Tools:** Python, Scrapy/BeautifulSoup, Pandas/Spark (via SageMaker Processing), SageMaker Feature Store, S3, potentially AWS Glue for ETL orchestration if complex.

### Data Filtering

- **Rules:**
  - Exclude listings missing core features (Price, Make, Model, Year, Mileage).
  - Exclude listings with likely data entry errors (e.g., Mileage=1, Price=$1).
  - Remove duplicate listings (based on VIN or key features + location/seller).
  - Potentially filter extreme price outliers per Make/Model/Year group.
- **Justification:** Improve data quality and relevance for modeling. Document filtering impact (% data removed).

### Data Labeling

- **Strategy:** Supervised Learning. The **target variable (label)** is the **listing price** scraped from the website. This is inherent in the data (**Data_Labeling** is automatic).
- **Schema:** Target = Numerical Price (e.g., USD).
- **Quality Control:** Validate price extraction logic. Handle currency symbols/formats. Check for unreasonable price values during cleaning.

### Data Pipeline & Storage

- **Storage:**
  - Raw Scraped Data: S3 (Versioned buckets).
  - Processed Data: S3 (Parquet format, partitioned by date/region).
  - Features for Training/Inference: **SageMaker Feature Store** (Online Store for low-latency inference, Offline Store for training).
- **Tradeoffs:** Feature Store provides low latency and consistency but higher cost than direct S3 access for batch training.
- **Properties:** Pipeline needs **Reliability** (handle scrape failures, retries), **Monitoring** (success/failure rates, data volumes), **Scalability** (handle increasing data volume). Feature Store needs defined update **Latency**. (**Data_Pipeline_Properties**).

### Metadata Management

- **Tracked Metadata:** Scrape source URL, scrape timestamp, data processing script version (Git hash), Feature Store feature group versions, dataset statistical profiles (mean/median/min/max/missing% per feature), owner.
- **Importance:** **Reproducibility** (link model version to data version), **Debugging** (trace data issues), **Monitoring_and_Reliability** (detect schema/distribution drift), **Governance**. Use SageMaker Experiments/Model Registry tracking.

---

## 3. Modeling Approach

### Baseline Model

- **Definition:** Simple heuristic: Predict the average price for the specific Make/Model/Year combination observed in the training data. If unseen, predict overall average. Alternative: Simple Linear Regression on Year and Mileage.
- **Performance:** Measure MAPE/RMSE on the validation set. This establishes the minimum acceptable performance (**Baseline_Solution**).

### Proposed Model(s)

- **Choice:** Gradient Boosting Trees (e.g., **SageMaker Built-in XGBoost** or LightGBM).
- **Justification:** Strong performance on tabular data, handles numerical/categorical features well, robust to outliers (to some extent), readily available and optimized in SageMaker. Relatively interpretable feature importance.
- **Implications:** Moderate training time/cost (can be tuned), fast inference.
- **Alternatives:** Linear Models (likely too simple), Deep Neural Networks (potentially more accurate but higher complexity, data needs, harder tuning/infra).
- **Hyperparameters:** Tune using **SageMaker Automatic Model Tuning** (Hyperparameter Optimization jobs) targeting validation set MAPE/RMSE. Key parameters: learning rate, tree depth, number of estimators, regularization terms.
- **Risks:** Potential bias if training data is geographically skewed or doesn't represent all vehicle types equally (**Risks_and_Consequences_in_ML_Systems**).

### Feature Engineering

- **Plan:**
  - Direct Features: Year, Mileage (numeric).
  - Categorical: Make, Model, Location (e.g., State or ZIP-code prefix), Transmission Type. Use **SageMaker Feature Store** definitions.
- **Transformations:**
  - Log-transform Mileage and Price (target variable) if distributions are highly skewed.
  - Standard Scaling for numerical features.
  - Categorical Encoding: Target encoding or embedding layers (if using DNNs) or rely on GBT's native handling (e.g., XGBoost/LightGBM). One-hot encoding if feature cardinality is low.
- **Feature Store:** Use **SageMaker Feature Store** to ensure consistency between features used in training (batch transformation) and features retrieved for online inference.
- **Iteration:** Based on **Error_Analysis**, potentially add features like: Engine Size/Type, Trim Level (requires robust parsing from text), Time-on-Market (if available), distance-to-major-city.

### Feature Analysis & Selection

- **Importance:** Use **Feature_Importance_Analysis** (e.g., SHAP integrated with SageMaker Clarify, or built-in GBT importance metrics) during development to understand drivers and debug.
- **Selection:** Initially include all core engineered features. **Feature_Selection** (e.g., removing low-importance features) might be considered later if model complexity or latency becomes an issue, but carefully validate impact on accuracy.

---

## 4. Evaluation Strategy

### Metrics

- **Offline (Model Development):**
  - Primary Metric: **MAPE** (Mean Absolute Percentage Error) - intuitive business interpretation.
  - Secondary Metric: **RMSE** (Root Mean Squared Error) - sensitive to large errors.
  - Optimization Target (**Loss_Functions_and_Metrics**): RMSE or Huber Loss during training.
- **Online (Business Impact):**
  - Primary: API Latency (p95), API Error Rate (5xx).
  - Secondary: Prediction request volume, potentially user feedback if UI includes it. (Proxy metrics for user satisfaction/utility - **Offline_vs_Online_Metrics**).
- **Fairness/Robustness:** Track MAPE across different regions or car value segments if concerns about bias arise.

### Validation Schema

- **Schema:** **Time-based split**. Train on data scraped up to `T1`, validate on data scraped between `T1` and `T2` (e.g., `T2 = T1 + 1 week`). This mimics production deployment where the model predicts on future, unseen data.
- **Justification:** Prevents **Data_Leakage** by ensuring the model doesn't learn from future price information. Crucial for time-sensitive data like market prices.
- **Leakage Prevention:** Ensure any aggregate features (e.g., average price for a model in a region) used during **Feature_Engineering** are calculated _only_ using data from _before_ the timestamp of the record being processed. Apply preprocessing steps _within_ each fold if using cross-validation during development (though time-split is primary).
- **Updating:** Use a rolling forward window for retraining and evaluation (e.g., train on last 6 months, validate on next 1 week).

### Error Analysis Plan

- **Approach:** Systematically analyze prediction errors on the validation set.
- **Techniques:**
  - **Residual_Analysis**: Plot residuals (actual - predicted price) vs. predicted price, vs. key features (Year, Mileage, Location). Look for patterns (e.g., underpredicting expensive cars, high variance for older cars).
  - Analyze largest errors: Identify specific types of cars or listings where the model performs poorly.
  - Segment performance: Compare MAPE/RMSE across different Makes, Models, Locations, or price buckets.
- **Feedback Loop:** Insights guide further **Feature_Engineering**, data cleaning, or model selection. E.g., "Model struggles with rare models -> needs more data or specific handling."

### Business Impact Evaluation

- **Methodology:** Primarily track **Online Metrics** (latency, errors, usage volume) as proxies for user value (**Measuring_Results**). Direct business impact (e.g., effect on car sales) is hard to measure for this tool.
- **A/B Testing Plan:**
  - If comparing major model versions (e.g., GBT vs. DNN) or significant feature changes:
  - Hypothesis: New model version B will reduce offline MAPE by X% compared to current production model A, while maintaining online latency and error rate targets.
  - Primary Metric: Offline MAPE (for decision), Online Latency/Error Rate (guardrails).
  - Randomization: User ID or session ID, split traffic (e.g., 50/50) via SageMaker Endpoint production variants.
  - Duration: Run long enough to capture sufficient prediction requests for stable offline metric comparison on recent data.

---

## 5. Training & Experimentation

### Training Pipeline

- **Architecture:** Use **SageMaker Pipelines** to define and orchestrate the end-to-end workflow.
- **Stages:** Data Extraction/Processing (SageMaker Processing Job) -> Feature Engineering/Update Feature Store -> Model Training (SageMaker Training Job) -> Model Evaluation (SageMaker Processing Job) -> Conditional Model Registration (SageMaker Model Registry based on evaluation results).
- **Tools:** SageMaker Pipelines, SageMaker Python SDK, SageMaker Processing, SageMaker Training, SageMaker Experiments (for tracking), SageMaker Model Registry, Docker, Git.
- **Scalability:** Leverage SageMaker's distributed processing/training capabilities if data volume grows significantly. Select appropriate instance types.
- **Artifact Management:** Models stored in S3 via SageMaker, registered in SageMaker Model Registry. Evaluation reports saved to S3 and linked in SageMaker Experiments/Pipelines. Logs go to CloudWatch.

### Reproducibility

- **Mechanisms:**
  - Code: Version control pipeline definition and scripts using Git (commit hash).
  - Data: Use specific timestamps or version IDs for S3 datasets and **SageMaker Feature Store** feature groups.
  - Environment: Use specific SageMaker Docker image URIs (built-in or custom). Pin library versions (`requirements.txt`).
  - Randomness: Set fixed random seeds in training scripts.
  - Parameters: Log all pipeline parameters, hyperparameters, and instance types via SageMaker Experiments.
- **Challenges:** Potential non-determinism in distributed computing or underlying libraries (mitigate with seeds where possible).

### Configuration

- **Management:** Use SageMaker Pipelines parameters for dynamic settings (e.g., instance types, date ranges). Store static config (e.g., feature lists, model settings) in YAML/JSON files versioned with the code in Git.
- **Experiment Tracking:** SageMaker Experiments automatically logs parameters passed to pipeline runs. Hyperparameter tuning jobs manage configurations internally.

### Pipeline Testing

- **Strategy:**
  - _Unit Tests:_ For helper functions (e.g., cleaning logic, feature transformations) using `pytest`.
  - _Integration Tests:_ Test interactions between pipeline steps locally or using **SageMaker Local Mode** (e.g., ensure output schema of processing matches training input).
  - _End-to-End Smoke Tests:_ Run the full SageMaker Pipeline on a small, well-defined data subset to catch major errors quickly.
  - _Data Validation:_ Use tools like `pydeequ` (on Spark) or Great Expectations within SageMaker Processing Jobs to assert data quality (nulls, ranges, distributions).

---

## 6. Deployment & Operations

### Integration Plan

- **Architecture:** React UI -> AWS API Gateway -> AWS Lambda (running FastAPI app) -> **SageMaker Endpoint**.
- **Dependencies:** Lambda needs permissions to invoke SageMaker Endpoint. Endpoint needs access to **SageMaker Feature Store** (Online Store) if real-time feature lookup is used (though primary features likely come from user input via API).
- **Deployment:** Use AWS CDK or CloudFormation for infrastructure-as-code deployment across dev/staging/prod environments. SageMaker Pipelines handles model deployment/update to the endpoint.

### Serving & Inference

- **API Design:** Defined by FastAPI in the Lambda function.
  - Protocol: REST (via API Gateway).
  - Endpoint: e.g., `/predict`.
  - Request Schema (JSON): `{ "make": "Toyota", "model": "Camry", "year": 2018, "mileage": 50000, "zip_code": "90210" }`
  - Response Schema (JSON): `{ "predicted_price": 21500.50 }` or `{ "error": "message" }`. Use OpenAPI spec.
  - Authentication: API Gateway handles auth (e.g., API Keys, Cognito).
- **Inference Pipelines:** Real-time inference via **SageMaker Endpoint**.
- **Serving_and_Inference_Optimization:**
  - Target: p95 latency < 500ms. Throughput needs based on expected traffic (start small, configure auto-scaling).
  - Hardware: CPU-based instances likely sufficient for GBTs (e.g., `ml.m5.large`).
  - Frameworks: Use SageMaker's built-in XGBoost serving container or a custom container with the trained model.
  - Optimizations: SageMaker Endpoint Auto-Scaling based on CPU utilization or invocation rate. Consider model quantization if latency is critical (trade-off with accuracy). Caching at API Gateway level for identical requests (short TTL).
- **Model Versioning:** Use SageMaker Endpoint Production Variants to deploy new models with traffic shifting (e.g., canary release) for safe rollout and A/B testing.

### Monitoring & Reliability

- **Strategy:** Comprehensive monitoring using **SageMaker Model Monitor** and **AWS CloudWatch**.
- **Metrics & Tools:**
  - _System Health (CloudWatch):_ Endpoint Latency (p50, p90, p99), Invocation Count, Error Rates (4xx, 5xx), Instance CPU/Memory Utilization. Lambda function metrics (duration, errors, throttles). API Gateway metrics (latency, errors).
  - _Data Quality/Integrity (SageMaker Model Monitor - Data Quality):_ Schedule jobs to compare live prediction input data distribution against the training dataset baseline. Detect schema violations, missing features, **Training-Serving_Skew**. Monitor feature distributions (PSI, KS tests).
  - _Model Performance (SageMaker Model Monitor - Model Quality/Bias):_ If ground truth (actual sale price) becomes available later, track prediction accuracy drift (**Concept_Drift**). Initially, monitor prediction output distribution for significant shifts compared to validation set predictions (proxy for drift).
- **Alerting Strategy:** CloudWatch Alarms on critical metrics: High p99 latency, high 5xx error rate, significant data drift detected by Model Monitor, high resource utilization. Define severity levels and notification channels (e.g., SNS -> PagerDuty/Slack). Link alerts to runbooks.

### Fallbacks & Overrides

- **Failure Behavior:**
  - If SageMaker Endpoint invocation fails (error/timeout): Lambda function returns a specific error code/message to the UI ("Prediction service unavailable").
  - If input validation in Lambda fails (e.g., missing required feature): Return 4xx error with details.
- **Fallback Strategy:** No model-based fallback initially (e.g., reverting to baseline). Focus on endpoint reliability. If Model Monitor triggers critical data quality alerts, manual intervention might be needed to pause predictions or investigate.
- **Overrides:** No business rule overrides planned for V1.

---

## 7. Maintenance & Sustainability

### Ownership & Accountability

- **Accountability:** Clearly define owners for:
  - Scraping Pipeline: ML Team / Data Engineering.
  - **SageMaker Pipelines** (Training/Deployment): ML Team.
  - **SageMaker Endpoint** & Monitoring: ML Team / Ops Team.
  - FastAPI Backend: Backend Team.
  - React Frontend: Frontend Team.
- **On-call:** Document rotation schedule and escalation paths via PagerDuty/OpsGenie.
- **Cost:** Track SageMaker and related AWS service costs via Cost Explorer and tagging.

### Knowledge Sharing & Risk

- **Bus_Factor:** Assess dependency on specific individuals for SageMaker expertise or scraping logic.
- **Mitigation:** Use Infrastructure-as-Code (CDK/CFN). Maintain thorough **Documentation\_(ML_Systems)** (this doc, runbooks, code comments). Regular internal demos and knowledge sharing sessions. Encourage code reviews.

### Documentation

- **Artifacts:** This Design Document (**Living_Document_Concept**), API Documentation (OpenAPI via FastAPI), Operational Runbooks (common alerts, debugging steps), READMEs for code repositories, SageMaker Model Cards (auto-generated/manual).
- **Maintenance:** Assign owners for each document. Store centrally (e.g., Confluence, shared Git repo). Keep documentation updated as system evolves.

### Complexity Management

- **Strategy:** Leverage managed AWS services (**SageMaker**) extensively to reduce operational **Complexity\_(ML_Systems)**. Start with the simplest viable model (GBT). Introduce complexity (e.g., more complex features, DNNs) only when justified by significant performance gains. Periodically review pipeline/architecture for refactoring opportunities.

### Retraining/Update Strategy

- **Trigger:** Primarily Schedule-based (e.g., weekly or monthly via SageMaker Pipelines schedule). Supplement with Performance-based trigger if monitoring shows significant **Data_Drift_and_Concept_Drift** or performance degradation.
- **Process:** Fully automated via **SageMaker Pipelines**. Includes data processing, training, evaluation, and comparison against the current production model metric (in Model Registry).
- **Validation:** Automatic evaluation on a held-out time-based test set within the pipeline. Requires comparison metric (e.g., MAPE) to meet or exceed production model's metric + a threshold before proceeding to deployment. Potential for manual approval gate before deployment.
- **Rollback:** Use SageMaker Endpoint's production variants. If the new model underperforms (monitored closely post-deployment), shift 100% traffic back to the previous stable variant via SageMaker console or API/SDK.

---

## 8. Review & Approval

### Review Process

- **Plan:** Circulate this document for asynchronous review using collaborative tools (e.g., Google Docs, Confluence comments).
- **Reviewers:** ML Engineering Lead, Backend Lead, Frontend Lead, Senior ML Engineer, potentially Product Owner/Manager.
- **Timeline:** Allow X days for review. Schedule 1-2 follow-up meetings to discuss feedback, prioritize changes, and reach consensus. Track feedback resolution. (**Design_Document_Review**)

### Approval

- **Authority:** Engineering Manager or Tech Lead holds final sign-off authority.
- **Criteria:** Goals are clear and measurable, approach is technically feasible, risks are identified and adequately addressed, dependencies understood, SageMaker utilization plan is sound.

---

_This document serves as the initial design. It is intended to be a **Living_Document_Concept**, updated collaboratively as the project progresses, requirements evolve, and insights are gained during implementation and operation._
