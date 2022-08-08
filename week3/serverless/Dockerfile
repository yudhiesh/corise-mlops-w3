# Base layer comes from here: https://github.com/aws/aws-lambda-base-images/blob/python3.8/Dockerfile.python3.8
FROM public.ecr.aws/lambda/python:3.8

# LAMBDA_TASK_ROOT is defined as `/var/task` in the base layer
# and is used by the lambda to figure out where to find the code
COPY . ${LAMBDA_TASK_ROOT}

# Install all Python dependencies
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
RUN pip install boto3 --target "${LAMBDA_TASK_ROOT}"

# Run the handler function
CMD ["service.handler"]
