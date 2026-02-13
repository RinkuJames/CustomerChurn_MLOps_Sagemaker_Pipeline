# CustomerChurn_MLOps_Sagemaker_Pipeline
End-to-end MLOps pipeline for customer churn prediction using XGBoost and Amazon SageMaker Pipelines (Processing, Training, Evaluation, and Automated Retraining).

This project implements an end-to-end machine learning pipeline for customer churn prediction using XGBoost and Amazon SageMaker Pipelines. The solution demonstrates a production-style MLOps workflow including data preprocessing, model training, evaluation, and automated pipeline execution using SageMaker SDK.
# Problem Statement
Customer churn prediction helps businesses identify customers likely to discontinue services. Early detection enables targeted retention strategies and revenue protection.
# Architecture - Pipeline Workflow
Data ingestion - Amazon S3

Preprocessing using SageMaker Processing Step

Model training (XGBoost Classifier-binary classification)

Evaluation with performance metrics (Accuracy & F1 Score)

Automated pipeline execution (Re-runnable/Retrainable and production-ready structure)
