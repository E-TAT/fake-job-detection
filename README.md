# 📸 Application Screenshots

## Home Page

![Home Page](screenshots/home_page.png)

The main interface where users can enter a job description and initiate fraud analysis.

---

## Fake Job Detection Example

![Fake Job Detection](screenshots/fake_job_detection.png)

Example showing detection of a fraudulent job posting with fraud probability score.

---

## Real Job Detection Example

![Real Job Detection](screenshots/real_job_detection.png)

Example showing detection of a legitimate job posting with confidence score.

---

## Model Summary Page

![Model Summary](screenshots/model_summary_page.png)

Displays model configuration, dataset information, and performance details.

---

# 📊 Dataset Analysis

## Dataset Distribution

![Dataset Distribution](screenshots/dataset_distribution.png)

Visualization showing the imbalance between real and fraudulent job postings in the dataset.

---

# 🤖 Model Evaluation

## SVM Performance

![SVM Results](screenshots/svm_results.png)

Performance metrics of the final Support Vector Machine model used in the application.

### Results
- Accuracy: ~97.7%
- Fake Job Recall: ~83%
- Fake Job Precision: ~76%

---

## Random Forest Comparison

![Random Forest Results](screenshots/random_forest_results.png)

Comparison model used during experimentation and model selection.

The SVM model was selected because it achieved better fraud detection performance while maintaining high overall accuracy.

---

# 🏗️ System Design

## Use Case Diagram

![Use Case Diagram](screenshots/use_case_diagram.png)

Shows interactions between users, administrators, and the Fake Job Detection System.

---

## Activity Diagram

![Activity Diagram](screenshots/activity_diagram.png)

Illustrates the complete workflow from user input to final prediction output.

---

## Data Flow Diagram

![Data Flow Diagram](screenshots/data_flow_diagram.png)

Represents the movement of data between users, the application, and the trained machine learning model.

---

## System Architecture

![System Architecture](screenshots/system_architecture.png)

High-level architecture showing preprocessing, TF-IDF vectorization, SVM classification, probability calculation, and result generation.

---

## Workflow Diagram

![Workflow Diagram](screenshots/workflow_diagram.png)

Detailed machine learning pipeline from dataset preprocessing to final prediction.
