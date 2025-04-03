## Title of Project:
 Predicting Microfinance Loan Default Risk

**Value Proposition:**  
Microfinance institutions and small banks frequently serve loan applicants with limited or no credit history. Manually assessing these applications is slow, subjective, and often inconsistent. We propose a machine learning system to **automatically assess the default risk of loan applicants**, providing an “approval likelihood” score (High, Medium, Low).  

The ML system will reduce the time to screen applicants, improve approval consistency, and proactively manage portfolio risk.  

**Current Status Quo:**  
Traditionally, underwriters rely on basic rule-based heuristics (like income cutoffs, past payment delays), which do not capture the complexity of multi-factor risk indicators.  

**Business Metrics:**  
- **Classification accuracy / F1-score** on test set  
- **Recall of high-risk (Low likelihood) applicants**  
- **Turnaround time for evaluation**

---

### Contributors

| Name            | Responsible for                                  | Link to their commits in this repo |
|-----------------|---------------------------------------------------|------------------------------------|
| All team members| Design, development, deployment, and evaluation   |                                    |
| Dhruv Sridhar   | Model training, preprocessing, dataset curation   |                                    |
| Barath Rama Shankar | Backend + model API integration, data pipeline    |                                    |
| Sampreeth Avvari | Frontend (form input + result visualization)      |                                    |

---

┌────────────────────┐
│     User Input     │  ←─ Loan application form (Gradio or React)
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Frontend (UI)    │
│  - React/Gradio    │
└────────┬───────────┘
         │ calls API
         ▼
┌────────────────────────┐
│   FastAPI Backend API  │
│ - Validates input      │
│ - Loads ML model       │
│ - Returns prediction   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────────┐
│  Trained ML Model (XGBoost)│
│ - Classifies as Low/Med/High│
│ - Loaded from saved model  │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│ Prediction + Explanation   │
│ - Score breakdown          │
│                            │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│ Output to UI (Approval Risk│
│ Score + Visualization)     │
└────────────────────────────┘


---

### Summary of Outside Materials

|               | How it was created                                                                 | Conditions of use |
|---------------|--------------------------------------------------------------------------------------|-------------------|
| LendingClub dataset | Collected and shared by LendingClub, real-world loan performance data (~1M+ rows) | Publicly available for academic research via Kaggle |
| scikit-learn   | Open-source ML library for classical tabular models                               | BSD-3 license     |
| XGBoost        | Boosted tree model implementation                                                  | Apache 2.0        |

---

### Summary of Infrastructure Requirements

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | Data preprocessing, model training, frontend/backend hosting |
| `gpu_mi100`     | 4-hour block once a week (optional)               | Speeding up model training during hyperparameter tuning |
| Floating IPs    | 1 for entire project duration                     | For exposing web interface to users |
| Persistent storage | 100GB                                          | Store original + cleaned datasets, model artifacts, logs |

---

### Detailed Design Plan

#### Model Training and Training Platforms
1. **Strategy:**  
   - Use `XGBoost` as core model (binary/multiclass classification)
   - Map `loan_status` into 3 categories:
     - **High chance of approval**: Fully Paid, Current  
     - **Medium**: Late (31-120), Grace Period, Late (16-30)  
     - **Low**: Default, Charged Off, Does Not Meet Credit Policy  
   - Handle class imbalance using weighted loss or SMOTE

2. **Platform**:  
   - Training on Chameleon `m1.medium` or `gpu_mi100` with tracked experiments in MLFlow

3. **Difficulty Points (Unit 4)**:  
   - [✓] Use distributed hyperparameter tuning with Ray Tune or Optuna  
   - [✓] Log all training runs using hosted MLFlow (Unit 5)

---

#### Model Serving and Monitoring Platforms
1. **Strategy:**  
   - Wrap trained model into a FastAPI service
   - Frontend calls backend via REST API
   - Optimize model (e.g., convert to ONNX if needed) for inference speed

2. **Monitoring Plan:**
   - Log predictions + user feedback (optional)
   - Use Gradio dashboard or Prometheus + custom logs

3. **Difficulty Points (Unit 6 & 7)**:  
   - [✓] Quantize model for performance optimization  
   - [✓] Track online model degradation based on user feedback logs

---

#### Data Pipeline

1. **Strategy:**  
   - Initial ingestion of LendingClub CSVs
   - Build ETL pipeline for:
     - Cleaning (e.g., mapping `loan_status` to 3 classes)
     - Feature encoding (categoricals)
     - Normalization of numerical fields
   - Store as versioned, cleaned `.parquet` files on mounted volume

2. **Simulated Online Data:**  
   - Simulate real-time applicant data via a streaming script (Unit 8)

3. **Difficulty Points (Unit 8)**:  
   - [✓] Persistent storage for all data artifacts  
   - [✓] Stream ingestion + auto-cleaning pipeline with monitoring hooks

---

#### Continuous X (DevOps)

1. **Infrastructure as Code**  
   - Provision all Chameleon components with Terraform + Ansible  
   - Model container built with Docker; deployed via Helm

2. **CI/CD Pipeline:**  
   - GitHub Actions to:  
     - Run training jobs  
     - Package updated model  
     - Deploy to staging → promote to prod

3. **Environments:**  
   - Staging → Canary → Production  
   - Load test and feedback scoring before full deployment

4. **Difficulty Points (Unit 3)**:  
   - [✓] Full CI/CD pipeline for model retraining and redeployment  
   - [✓] Canary rollout of new models with performance comparison
