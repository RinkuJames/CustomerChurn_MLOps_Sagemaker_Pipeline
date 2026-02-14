import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model import Model
from sagemaker.transformer import Transformer

# -------------------------
# Setup
# -------------------------
session = sagemaker.Session()
region = session.boto_region_name
role = "<IAM_ROLE_ARN>"

# Replace with your actual model artifact path
model_artifact = "s3://bucket/path-to-trained-model/model.tar.gz"

bucket = session.default_bucket()

# -------------------------
# Create Model Object
# -------------------------
model = Model(
    image_uri=sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1"
    ),
    model_data=model_artifact,
    role=role,
    sagemaker_session=session
)

# -------------------------
# Create Transformer
# -------------------------
transformer = Transformer(
    model_name=model.name,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/churn/batch-output",
    sagemaker_session=session
)

# -------------------------
# Run Batch Transform
# -------------------------
transformer.transform(
    data=f"s3://{bucket}/churn/processed/test/test.csv",
    content_type="text/csv",
    split_type="Line"
)

transformer.wait()

print("Batch Transform Completed.")
