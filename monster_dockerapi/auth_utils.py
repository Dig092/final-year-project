import jwt
import time
import json
import hashlib
import logging

from decimal import Decimal
from botocore.exceptions import ClientError
from fastapi import Security, HTTPException, status
from boto3.dynamodb.conditions import Key, And, Attr
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from limits import RateLimitItemPerMinute, storage
from limits.strategies import MovingWindowRateLimiter

from config import JWT_SECRET, \
                    user_token_map, \
                    ADMIN_AUTH_TOKEN, \
                    CREDIT_CHECK_THRESHOLD, \
                    THROTTLE_LIMITS, \
                    JSONResponse, \
                    GPT_JWT_SECRET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

security = HTTPBearer()

memory_storage = storage.MemoryStorage()

rate_limiter = MovingWindowRateLimiter(memory_storage)

def rate_limit_check(user_id: str, plan_name: str):
    limit = RateLimitItemPerMinute(100)  # Default to 100 requests per minute
    key = f"{user_id}:{plan_name}"
    if not rate_limiter.hit(limit, key):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            # Convert decimal instances to strings
            return str(obj)
        # Let the base class default method raise the TypeError
        return super(DecimalEncoder, self).default(obj)

def verify_api_key(api_key, jwt_secret=JWT_SECRET):
    try:
        decoded_payload = jwt.decode(api_key, jwt_secret, algorithms=["HS256"], options={"verify_signature": True})
        print(f"API Key: {api_key}")
        print(f"Decoded: {decoded_payload}")
    except Exception as decodeExe:
        logger.error(f"Error occurred while decoding auth token: {decodeExe}")
        logger.error("Invalid Auth Token. Please generate a new token to proceed")
        return None
    else:
        if "username" in decoded_payload.keys():
            logger.info("Username found.. Will validate against cache/DB")
            user_id = decoded_payload['username']
        else:
            logger.info(f"Invalid payload: {decoded_payload}")
            return None

    if True:
        try:
            useridresponse = user_token_map.query(KeyConditionExpression=Key('user_id').eq(user_id))['Items']
            if len(useridresponse) > 0:
                logger.info("User present in DB, caching and proceeding.")
                user_info = useridresponse[0]
                if user_info['is_valid']:
                    if api_key == user_info['api_key']:
                        keys_to_consider = ["user_id","user_email","api_key","available_credit","overage","is_valid","plan_name"]
                        filtered_user_info = {k: user_info[k] for k in keys_to_consider if k in user_info}
                        user_info = filtered_user_info
                    else:
                        logger.info(api_key)
                        logger.info(user_info)
                        logger.info("Invalid API Key.")    
                else:
                    logger.info("User is blocked. Please contact support.")
                    return None
            else:
                logger.info("No user found in DB.")
                return None
        except ClientError as err:
            logger.error(f"Error occurred while fetching user details: {err}")
            return None


    if user_info and user_info['is_valid']:
        rate_limit_check(user_info["user_id"], user_info["plan_name"])
        if user_info["api_key"] == api_key:
            return user_info
        else:
            return None
    else:
        return None

def user_auth_dependency(credentials: HTTPAuthorizationCredentials = Security(security)):
    throttle_limits = THROTTLE_LIMITS
    st = time.time()
    user_api_key = credentials.credentials  # Extract the API key from the credentials
    user_info = verify_api_key(user_api_key)  
    if user_info is None:
        verified = False
    else:
        verified = True

    vt = time.time()
    logger.info(100*'#')
    logger.info(f"Key verification time: {vt - st}")
    logger.info(100*'#')
    if not verified:  # If the API key is not verified
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    user_id = user_info.get("user_id")
    user_email = user_info.get("user_email")
    available_credit = user_info.get("available_credit")
    user_plan = user_info.get("plan_name")

    
    # Verify min credit threshold
    if CREDIT_CHECK_THRESHOLD is not None:
        if float(available_credit) < CREDIT_CHECK_THRESHOLD:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User does not have enough credits, please maintain 100 Credits atleast!"
            )

    # If the API key is verified, return the user_id and plan
    et = time.time()
    logger.info(100*'#')
    logger.info(f"Authentication Time: {et-st}")
    logger.info(100*'#')
    return {"user_id": user_id, "plan": user_plan, "user_email": user_email}


def verify_jwt_token(token, secret_key = GPT_JWT_SECRET):
    if not secret_key:
        raise ValueError('JWT secret key is not defined. Please set it in your environment variables.')

    try:
        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
        return decoded
    except jwt.ExpiredSignatureError:
        raise ValueError('Token has expired')
    except jwt.InvalidTokenError:
        raise ValueError('Invalid user jwt session token')

def hash_user_email(data):
    try:
        hash_object = hashlib.md5(data.encode())
        hash_object = hash_object.hexdigest()
        return hash_object
    except:
        raise ValueError(f"Invalid user email: {data}")

def parse_user_session_token(user_session_token):
    if user_session_token == None:
        return JSONResponse(content = {"message": f"USER Session token authorization failed! Error: please feed user_session_token in payload."}, status_code = 404)
    else:
        try:
            user_email = verify_jwt_token(user_session_token)['email']
            user_id = hash_user_email(user_email)
            return {"user_email": user_email, "user_id": user_id}
        except ValueError as e:
            return JSONResponse(content = {"message": f"USER Session token authorization failed! Error: {e}"}, status_code = 401)
        except Exception as e:
            return JSONResponse(content = {"message": "Unexpected USER Session token authorization fail, Please contact support at support@monsterapi.ai!"}, status_code = 500)