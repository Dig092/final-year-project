import os
import json
import boto3
import base64
import logging

from functools import partial
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError

logging_level = os.environ.get('LOGGING_LEVEL', 'INFO')

if logging_level == 'DEBUG':
   logging_level = logging.DEBUG
elif logging_level == 'INFO':
    logging_level = logging.INFO
elif logging_level == 'WARNING':
    logging_level = logging.WARNING
elif logging_level == 'ERROR':
    logging_level = logging.ERROR
elif logging_level == 'CRITICAL':
    logging_level = logging.CRITICAL
else:
    logging_level = logging.INFO

def read_secret(secret_name, region):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region
    )

    try:
        # Attempt to get the secret value
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # Handle the exception if the secret is not found or access is denied
        raise e
    else:
        # If there's no error, the secret value is retrieved
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            # In case the secret is a binary blob, we decode it
            secret = base64.b64decode(get_secret_value_response['SecretBinary'])
        return json.loads(secret)

def get_secret(secret_name: str, region_name: str) -> str:
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except Exception as e:
        # Handle exceptions
        print(f"Error retrieving secret: {e}")
        return None

    return json.loads(get_secret_value_response['SecretString'])

environment = os.environ.get('ENVIRONMENT', "staging")

if environment == "staging":
    aws_region = "ap-south-1" # Mumbai
    ADMIN_AUTH_TOKEN = read_secret("staging/mapi-adm-auth", aws_region)["authToken"]
    JWT_SECRET = read_secret("staging/jwtsecret", aws_region)["secretKey"]
    GPT_JWT_SECRET = "customGPT"
elif environment == "prod" or environment == "production":
    aws_region = "us-east-2" # Ohio
    ADMIN_AUTH_TOKEN = read_secret("prod/mapi-adm-auth", aws_region)["authToken"]
    JWT_SECRET = read_secret("prod/jwtsecret", aws_region)["secretKey"]
    GPT_JWT_SECRET = "customGPT"
else:
    raise ValueError(f"Invalid environment {environment}")


dynamodb = boto3.resource('dynamodb', region_name=aws_region)
user_token_map = dynamodb.Table('user_token_map_v1')

JSONResponse = partial(JSONResponse, headers={"Access-Control-Allow-Origin": "*"})

CREDIT_CHECK_THRESHOLD = None

THROTTLE_LIMITS = {
    "freeplan": 10,
    "wolf": 20,
    "beast": 40,
    "monster": 60
}