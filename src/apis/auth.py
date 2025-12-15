from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
import os

class SimpleAuth:
    """
    Simple Token-based Authentication Middleware.
    In a real scenario, this would integrate with Cognito/IAM or similar.
    For this prototype, we check against a fixed environment variable or allow if not set (dev mode).
    """
    def __init__(self, token_env_var: str = "API_AUTH_TOKEN"):
        self.token_env_var = token_env_var
        self.security = HTTPBearer()

    async def __call__(self, request: Request):
        # Allow health checks and root without auth
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
             return

        expected_token = os.environ.get(self.token_env_var)
        
        # Dev mode: if no token is set in env, allow all
        if not expected_token:
            return

        # Check Authorization header
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=403, detail="Not authenticated")
        
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or token != expected_token:
             raise HTTPException(status_code=403, detail="Invalid token")

# Instance to be used as dependency
auth_dependency = SimpleAuth()
