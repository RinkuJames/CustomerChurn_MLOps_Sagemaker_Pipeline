import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.properties import PropertyFile


pipeline_session = PipelineSession()
region = pipeline_session.boto_region_name
role = "<IAM_ROLE_ARN>"
bucket = pipeline_session.default_bucket()

raw_data_uri = f"s3://{bucket}/churn/raw"

# -------------------
# Processing Step
# -------------------
sklearn_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1"
    ),
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session
)

processing_step = ProcessingStep(
    name="PreprocessStep",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=raw_data_uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            destination=f"s3://{bucket}/churn/processed/train"
        ),
        ProcessingOutput(
            output_name="test",
            destination=f"s3://{bucket}/churn/processed/test"
        )
    ],
    code="src/preprocess.py"
)

# -------------------
# Training Step
# -------------------
xgb_estimator = XGBoost(
    entry_point="src/train.py",
    framework_version="1.7-1",
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session
)

training_step = TrainingStep(
    name="TrainingStep",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
        ),
        "test": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
        )
    }
)

# Evaluation Processor
evaluation_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1"
    ),
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session
)

# Evaluation Report File
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

#Evaluation Step
evaluation_step = ProcessingStep(
    name="EvaluationStep",
    processor=evaluation_processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            destination=f"s3://{bucket}/churn/evaluation"
        )
    ],
    code="src/evaluate.py",
    property_files=[evaluation_report]
)

# Only continue if accuracy > 0.75
condition_step = ConditionStep(
    name="AccuracyCondition",
    conditions=[
        ConditionGreaterThan(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.accuracy.value"
            ),
            right=0.75
        )
    ],
    if_steps=[],
    else_steps=[]
)

# -------------------
# Pipeline
# -------------------
pipeline = Pipeline(
    name="CustomerChurnPipeline",
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=pipeline_session
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()

print("Pipeline execution started.")
