# Use the official AWS Lambda Python 3.8 base image
FROM public.ecr.aws/lambda/python:3.8

# Copy the application and model files to the container
COPY app.py ${LAMBDA_TASK_ROOT}
COPY fingerprint_data.csv ${LAMBDA_TASK_ROOT}
COPY normalized_data.csv ${LAMBDA_TASK_ROOT}
COPY train_normalize_user.py ${LAMBDA_TASK_ROOT}
# Install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# RUN echo "numpy==1.21.0\npandas==1.5.3\nscikit-learn==1.3.2\njoblib==1.4.2" > requirements.txt


# Run the training script at build time to create joblib files
RUN python train_normalize_user.py

# Copy the generated joblib files to the Lambda task root
# COPY one_hot_encoder.joblib ${LAMBDA_TASK_ROOT}
# COPY scaler.joblib ${LAMBDA_TASK_ROOT}

# Set the Lambda handler
CMD ["app.lambda_handler"]
