# Backend API Architecture Manifesto

## Overview

This document defines the architecture for a student portal backend built with FastAPI. The system includes:

- User authentication with MFA (multi-factor authentication)
- Class enrollment management
- Payment processing
- Custom captcha verification for sensitive operations

---

## JWT Types

This system uses **5 distinct JWTs**:

| JWT Type                     | Location                                                                      | Purpose                                                          | TTL    |
| ---------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------ |
| `mfa_required_auth_token`    | Response body from `/login`                                                   | Restricted token; required for `/mfa/initiate` and `/mfa/submit` | 5 min  |
| `encrypted_mfa_code_token`   | Response body from `/mfa/initiate`                                            | Contains encrypted MFA code; verified in `/mfa/submit`           | 5 min  |
| `auth_token`                 | Response body from `/mfa/submit`, then sent in `Authorization: Bearer` header | Full access token for all authenticated endpoints                | 30 min |
| `mfa_authenticated_token`    | Response body from `/mfa/submit` + HTTP-only cookie                           | Proves user completed MFA recently; allows skipping MFA          | 30 min |
| `captcha_solved_token`       | Response body from `/captcha/submit`                                          | Proof of successful captcha solve; scoped to a specific purpose  | 1 min  |

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration and environment variables
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py           # Main API router aggregator
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── utils.py        # Utility endpoints for hackathon testing (POST /user, /reset)
│   │       ├── auth.py         # Auth endpoints (POST /login, /mfa/initiate, /mfa/submit)
│   │       ├── users.py        # User endpoints (GET /user-info, DELETE /class, POST /payment-method, /payment, /dropout)
│   │       └── captcha.py      # Captcha endpoints (GET /captcha/challenge, POST /captcha/submit)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── users.py            # User business logic (auth, payments, classes)
│   │   └── captcha.py          # Captcha business logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py             # User, Class, Finance, PaymentMethod models
│   │   └── captcha.py          # Captcha request/response models
│   │
│   └── core/
│       ├── __init__.py
│       ├── security.py         # JWT creation/validation, encryption utilities, decorators
│       └── exceptions.py       # Custom exception handlers
│
├── data/
│   └── users.json              # User data persistence
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Data Persistence

User data is stored in a JSON file (`data/users.json`).

### Data Structure

```json
{
  "username1": {
    "username": "student_sun",
    "email": "sun@example.com",
    "password": "securepassword",
    "status": "active",
    "full_name": "Sun Student",
    "classes": [
      {
        "class_id": "CS-101",
        "name": "Intro to Computer Science",
        "semester": "Fall 2024",
        "cost": 100
      },
      {
        "class_id": "MATH-200",
        "name": "Linear Algebra",
        "semester": "Fall 2024",
        "cost": 100
      }
    ],
    "finance": {
      "base_balance": 100,
      "amount_paid": 150.00,
      "payment_methods": [
        {
          "id": "4242",
          "type": "credit_card",
          "credit_card_number": "4242424242424242",
          "cvv": "424",
          "expiry": "12/26"
        }
      ]
    }
  },
  "username2": { ... }
}
```

### Key Fields

| Field               | Description                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| `status`            | `"active"` or `"dropped"`                                                |
| `classes`           | Array of enrolled classes (hardcoded defaults on registration)           |
| `base_balance`      | Base tuition amount every student owes, regardless of classes            |
| `amount_paid`       | Total amount user has paid                                               |
| `balance`           | Calculated dynamically: `base_balance + sum(classes.cost) - amount_paid` |
| `payment_method.id` | Always the last 4 digits of the credit card number                       |

---

## API Endpoints

### 1. Utility Endpoints (Hackathon Testing)

These endpoints are **not part of the main application flow**. They exist to help hackathon participants easily set up and reset test accounts via Postman. They are not called from the frontend.

#### POST /user

**Purpose:** Register a new user for testing.

**Authentication:** None required.

**Request Body:**

```json
{
  "username": "student_sun",
  "email": "sun@example.com",
  "password": "securepassword",
  "full_name": "Sun Student"
}
```

**Response:** Confirmation of user creation.

**Notes:**

- Classes are initialized to hardcoded defaults (to be defined later)
- Payment methods array starts empty
- Status set to `"active"`
- Called via Postman by hackathon participants to create test accounts

---

#### POST /reset

**Purpose:** Reset a user account to initial state for re-testing.

**Authentication:** None required (validates username + password in body, similar to login).

**Request Body:**

```json
{
  "username": "student_sun",
  "password": "securepassword"
}
```

**Response:** Reset success confirmation.

**Logic:** Validates credentials like `/login` does, but instead of returning a token, resets the user's data.

**Side Effects:**

- Removes all payment methods
- Resets classes to hardcoded defaults
- Sets status to `"active"`
- Resets `amount_paid` to 0

**Notes:** This allows hackathon participants to quickly reset their test accounts without manually editing the JSON file.

---

### 2. Authentication

#### POST /login

**Purpose:** Authenticate user credentials (step 1 of login flow).

**Authentication:** None required.

**Request Body:**

```json
{
  "username": "student_sun",
  "password": "securepassword"
}
```

**Response:**

```json
{
  "mfa_required_auth_token": "jwt-token-5min-ttl"
}
```

**Notes:** This token can ONLY be used for `/mfa/initiate`. Contains claim `credentials_valid=true`.

---

#### POST /mfa/initiate

**Purpose:** Trigger MFA code to be sent via email.

**Authentication:** `Authorization: Bearer <mfa_required_auth_token>`

**Response:**

```json
{
  "encrypted_mfa_code_token": "jwt-containing-encrypted-otp"
}
```

**Side Effects:** Sends OTP code to user's email.

**Notes:** The `encrypted_mfa_code_token` contains the encrypted OTP code that will be verified in `/mfa/submit`.

---

#### POST /mfa/submit

**Purpose:** Verify OTP code and complete login.

**Authentication:** `Authorization: Bearer <mfa_required_auth_token>`

**Request Body:**

```json
{
  "encrypted_mfa_code_token": "jwt-from-mfa-initiate",
  "code": "123456"
}
```

**Response:**

```json
{
  "auth_token": "jwt-full-access-30min-ttl"
}
```

**Notes:** The `auth_token` grants access to all authenticated endpoints.

---

### 3. User Operations

#### GET /user-info

**Purpose:** Retrieve all user information including calculated balance.

**Authentication:** `Authorization: Bearer <auth_token>`

**Response:**

```json
{
  "username": "student_sun",
  "email": "sun@example.com",
  "full_name": "Sun Student",
  "status": "active",
  "classes": [...],
  "finance": {
    "base_balance": 100,
    "amount_paid": 150.00,
    "balance": 50.00,
    "payment_methods": [...]
  }
}
```

**Notes:** `balance` is calculated as: `base_balance + sum(classes.cost) - amount_paid`. Can be negative (credit to the student).

---

#### DELETE /class

**Purpose:** Remove a single class from enrollment.

**Authentication:** `Authorization: Bearer <auth_token>`

**Request Body:**

```json
{
  "class_id": "CS-101"
}
```

**Response:** Confirmation of class removal.

**Notes:** Only one class can be removed per request.

---

#### POST /payment-method

**Purpose:** Add a new payment method.

**Authentication:** `Authorization: Bearer <auth_token>`

**Request Body:**

```json
{
  "credit_card_number": "4242424242424242",
  "cvv": "424",
  "expiry": "12/26"
}
```

**Response:** Confirmation with payment method ID (last 4 digits of card number).

---

#### POST /payment

**Purpose:** Execute a payment transaction.

**Authentication:** `Authorization: Bearer <auth_token>`

**Request Body:**

```json
{
  "payment_method_last_4": "4242",
  "amount": 50.0
}
```

**Response:** Payment confirmation.

**Side Effects:** Increases `amount_paid` by the specified amount.

---

#### POST /dropout

**Purpose:** Finalize user dropout from the program.

**Authentication:** `Authorization: Bearer <auth_token>`

**Request Body:**

```json
{
  "captcha_solved_token": "jwt-from-captcha-submit"
}
```

**Validation:**

- User's balance MUST be 0 or less (no outstanding debt)
- `captcha_solved_token` must be valid and not expired

**Response:** `"User has dropped out!"`

**Side Effects:**

- Updates user status to `"dropped"`
- Sends email notification to "The Deck team" (email service TBD)

---

### 4. Captcha

> **Note:** Image fetching and generation logic is TBD. The request/response structure below is finalized; implementation details for image sourcing will be provided later.

#### GET /captcha/challenge

**Purpose:** Generate and return a captcha challenge.

**Authentication:** `Authorization: Bearer <auth_token>`

**Response:**

```json
{
  "images": [
    { "id": "uuid-1", "url": "https://cdn.example.com/captcha/img1.png" },
    { "id": "uuid-2", "url": "https://cdn.example.com/captcha/img2.png" },
    { "id": "uuid-3", "url": "https://cdn.example.com/captcha/img3.png" }
  ],
  "encrypted_answer": "base64-encoded-encrypted-string"
}
```

**Note:** Image URLs are short-lived (~1 minute). The frontend must download and display images promptly. This is a frontend concern; the backend does not track URL expiration.

**Logic Flow:**

1. Validate `auth_token` from headers
2. Generate captcha challenge (select images, determine correct answers) - TBD
3. Encrypt list of correct URLs
4. Return image URLs, IDs, and encrypted answer

---

#### POST /captcha/submit

**Purpose:** Validate user's captcha selection.

**Authentication:** None required.

**Request Body:**

```json
{
  "selected_urls": [
    "https://cdn.example.com/captcha/img1.png",
    "https://cdn.example.com/captcha/img3.png"
  ],
  "encrypted_answer": "base64-encoded-encrypted-string",
  "purpose": "auth" | "payment" | "dropout"
}
```

**Success Response (200):**

```json
{
  "captcha_solved_token": "jwt-with-1min-ttl-containing-purpose"
}
```

**Failure Response (400):**

```json
{
  "error": "Invalid captcha selection"
}
```

**Logic Flow:**

1. Decrypt the encrypted answer (contains list of correct URLs)
2. Compare selected URLs against correct URLs from decrypted answer
3. If match: return `captcha_solved_token` with embedded `purpose` (1 min TTL)
4. If mismatch: return error response

**Note:** Any captcha challenge type (`pretty_faces`, `logos`, or `sun`) can be solved and submitted with any `purpose` value. The challenge type does not restrict what purpose the resulting token can have.

---

## Core Components

### Models (`app/models/user.py`)

```python
from pydantic import BaseModel
from typing import List

class PaymentMethod(BaseModel):
    id: str  # Last 4 digits of credit card number
    type: str = "credit_card"
    credit_card_number: str
    cvv: str
    expiry: str

class Class(BaseModel):
    class_id: str
    name: str
    semester: str
    cost: float

class Finance(BaseModel):
    base_balance: float
    amount_paid: float
    payment_methods: List[PaymentMethod]

class User(BaseModel):
    username: str
    email: str
    password: str
    status: str  # "active" or "dropped"
    full_name: str
    classes: List[Class]
    finance: Finance

# Request/Response models
class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str

class LoginRequest(BaseModel):
    username: str
    password: str

class MFASubmitRequest(BaseModel):
    encrypted_mfa_code_token: str
    code: str

class DeleteClassRequest(BaseModel):
    class_id: str

class PaymentMethodRequest(BaseModel):
    credit_card_number: str
    cvv: str
    expiry: str

class PaymentRequest(BaseModel):
    payment_method_last_4: str
    amount: float

class DropoutRequest(BaseModel):
    captcha_solved_token: str
```

### Models (`app/models/captcha.py`)

```python
from pydantic import BaseModel
from typing import List

class CaptchaImage(BaseModel):
    id: str
    url: str

class CaptchaChallenge(BaseModel):
    images: List[CaptchaImage]
    encrypted_answer: str

class CaptchaSubmitRequest(BaseModel):
    selected_urls: List[str]
    encrypted_answer: str
```

### Services (`app/services/users.py`)

Responsibilities:

- User CRUD operations (create, read, update)
- Class management (delete)
- Payment method management (add, lookup by last 4)
- Payment processing (update `amount_paid`)
- Balance calculation: `base_balance + sum(classes.cost) - amount_paid`
- User reset functionality
- Dropout processing
- JSON file read/write operations

### Services (`app/services/captcha.py`)

Responsibilities:

- Generate captcha challenges (image selection, answer determination) - TBD
- Use `create_encrypted_token()` and `decrypt_token()` from security.py for answer encryption
- Validate submitted answers by comparing decrypted URLs

### Security (`app/core/security.py`)

#### Header Tokens (validated via decorators)

These tokens are sent in `Authorization: Bearer` header and validated using decorators:

- `create_mfa_required_token(username)` - JWT with `credentials_valid=true`
- `create_auth_token(username)` - Full access JWT
- `@require_mfa_required_token` - Decorator to validate `mfa_required_auth_token`
- `@require_auth_token` - Decorator to validate `auth_token`

#### Encrypted Payload Tokens (validated via utility function)

These tokens contain encrypted values and are sent in request body. All use a single utility function:

- `create_encrypted_token(payload)` - Creates JWT with encrypted payload
- `decrypt_token(token)` - Decrypts and returns payload from any encrypted token

**Used for:**
| Token | Encrypted Payload | Usage |
|-------|------------------|-------|
| `encrypted_mfa_code_token` | OTP code (string) | Verify MFA code in `/mfa/submit` |
| `encrypted_answer` | List of correct URLs | Verify captcha selection in `/captcha/submit` |
| `captcha_solved_token` | `{"success": true, "purpose": "..."}` | Verify captcha for specific action (`/dropout`, `/payment`, etc.) |

#### Other Utilities

- `generate_otp()` - Generate 6-digit OTP code

---

## Authentication Strategy

### Login Flow

```
1. POST /login (username, password)
   └─> Returns: mfa_required_auth_token (5 min TTL, claim: credentials_valid=true)

2. POST /mfa/initiate (Bearer: mfa_required_auth_token)
   └─> Sends OTP to email
   └─> Returns: encrypted_mfa_code_token (5 min TTL, contains encrypted OTP)

3. POST /mfa/submit (Bearer: mfa_required_auth_token, body: encrypted_mfa_code_token + code)
   └─> Verifies OTP matches decrypted value from encrypted_mfa_code_token
   └─> Returns: auth_token (30 min TTL, full access)

4. All authenticated endpoints (Bearer: auth_token)
   └─> User Info, Class deletion, Payments, Captcha, Dropout
```

### Decorator Pattern for Header Tokens

```python
# app/core/security.py
from functools import wraps
from fastapi import HTTPException, Request

def require_mfa_required_token(func):
    """Decorator for /mfa/initiate and /mfa/submit - validates mfa_required_auth_token"""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        payload = decode_jwt(token)
        if not payload or not payload.get("credentials_valid"):
            raise HTTPException(status_code=401, detail="Invalid token")
        request.state.token_data = payload
        return await func(request, *args, **kwargs)
    return wrapper

def require_auth_token(func):
    """Decorator for all authenticated endpoints - validates auth_token"""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        payload = decode_jwt(token)
        if not payload or not payload.get("username"):
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        request.state.current_user = payload
        return await func(request, *args, **kwargs)
    return wrapper
```

### Usage in Endpoints

```python
@router.post("/mfa/initiate")
@require_mfa_required_token
async def mfa_initiate(request: Request):
    username = request.state.token_data["username"]
    ...

@router.post("/mfa/submit")
@require_mfa_required_token
async def mfa_submit(request: Request, body: MFASubmitRequest):
    # Decrypt the encrypted_mfa_code_token from body
    otp_from_token = decrypt_token(body.encrypted_mfa_code_token)
    if otp_from_token != body.code:
        raise HTTPException(status_code=400, detail="Invalid MFA code")
    ...

@router.get("/user-info")
@require_auth_token
async def get_user_info(request: Request):
    username = request.state.current_user["username"]
    ...

@router.post("/dropout")
@require_auth_token
async def dropout(request: Request, body: DropoutRequest):
    # Decrypt captcha_solved_token and check success boolean
    captcha_result = decrypt_token(body.captcha_solved_token)
    if not captcha_result.get("success"):
        raise HTTPException(status_code=400, detail="Invalid captcha token")
    ...
```

---

## Configuration (`app/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Student Portal API"
    DEBUG: bool = False

    # JWT Settings
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"

    # Token TTLs
    MFA_REQUIRED_TOKEN_EXPIRE_SECONDS: int = 300   # 5 min
    MFA_TOKEN_EXPIRE_SECONDS: int = 300            # 5 min
    AUTH_TOKEN_EXPIRE_SECONDS: int = 1800          # 30 min
    CAPTCHA_TOKEN_EXPIRE_SECONDS: int = 60         # 1 min

    # Encryption
    ENCRYPTION_KEY: str  # Fernet key for all encrypted tokens (MFA code, captcha answer, captcha solved)

    # Data
    USERS_FILE_PATH: str = "data/users.json"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Error Handling

Custom exceptions in `app/core/exceptions.py`:

- `UserNotFoundError` - User doesn't exist
- `InvalidCredentialsError` - Wrong username/password
- `InvalidMFACodeError` - Wrong OTP code
- `UserAlreadyExistsError` - Username taken during registration
- `InvalidCaptchaAnswerError` - Wrong captcha selection
- `DecryptionError` - Tampered or invalid encrypted payload
- `OutstandingBalanceError` - Cannot dropout with balance > 0
- `PaymentMethodNotFoundError` - Payment method doesn't exist
- `ClassNotFoundError` - Class ID doesn't exist in user's enrollment

Global exception handlers registered in `main.py` to return consistent error responses.

---

## Security Considerations

1. **Multi-Factor Authentication:** Login requires OTP sent to email
2. **Token Separation:** Different JWTs for each auth stage with appropriate TTLs
3. **Encrypted Payloads:** MFA codes and captcha answers encrypted with Fernet
4. **Short-lived Tokens:** Captcha token (1 min), MFA tokens (5 min), Auth token (30 min)
5. **Captcha for Sensitive Actions:** Dropout requires proof of human verification
6. **Rate Limiting:** Consider adding rate limiting middleware (future iteration)

---

## Dependencies

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-jose[cryptography]>=3.3.0
cryptography>=42.0.0
python-multipart>=0.0.6
resend>=0.8.0
```

---

## Email Service Implementation

**Provider:** [Resend](https://resend.com/) - Modern email API for developers

**Dependencies:**
```
resend>=0.8.0
```

**Configuration Required** (`app/config.py`):
```python
# Email Settings
SENDGRID_API_KEY: str  # API key from SendGrid dashboard
EMAIL_FROM: str = "onboarding@resend.dev"  # or your verified domain
TEAM_EMAIL: str = "team@thedeck.com"  # For dropout notifications
```

**Service Implementation** (`app/services/email.py`):
```python
import resend
from app.config import settings

resend.api_key = settings.SENDGRID_API_KEY

def send_mfa_code(to_email: str, otp_code: str):
    """Send OTP code to user's email via Resend"""
    try:
        params = {
            "from": settings.EMAIL_FROM,
            "to": [to_email],
            "subject": "Your MFA Code",
            "text": f"Your verification code is: {otp_code}\n\nThis code will expire in 5 minutes.",
        }
        email = resend.Emails.send(params)
        print(f"[EMAIL] Sent MFA code to {to_email}, ID: {email['id']}")
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send MFA code: {e}")
        raise

def send_dropout_notification(username: str, user_email: str):
    """Notify team of user dropout via Resend"""
    try:
        params = {
            "from": settings.EMAIL_FROM,
            "to": [settings.TEAM_EMAIL],
            "subject": "User Dropout Notification",
            "text": f"User {username} ({user_email}) has completed dropout process.",
        }
        email = resend.Emails.send(params)
        print(f"[EMAIL] Sent dropout notification for {username}, ID: {email['id']}")
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send dropout notification: {e}")
        raise
```

**Integration Points:**
1. `/mfa/initiate` endpoint - Replace console print with `send_mfa_code()`
2. `/dropout` endpoint - Add `send_dropout_notification()` after successful dropout

**Setup Notes:**
- Sign up at [resend.com](https://resend.com/)
- Get API key from dashboard
- For production: Verify your domain
- For testing: Use `onboarding@resend.dev` (Resend's test address)

---

## MFA 30-Minute Exemption

**Strategy:** Cookie-based "remember this device" token that allows users to skip MFA for 30 minutes after successful authentication.

### New Token Type: `mfa_authenticated_token`

| JWT Type | Location | Purpose | TTL |
|----------|----------|---------|-----|
| `mfa_authenticated_token` | Set as HTTP-only cookie after `/mfa/submit` | Proves user completed MFA recently; allows skipping MFA on next login | 30 min |

### Updated Authentication Flow

```
1. POST /login (username, password)
   ├─ Check for mfa_authenticated_token cookie
   ├─ If valid AND username matches:
   │  └─> Return: auth_token (full access, skip MFA)
   └─ Else:
      └─> Return: mfa_required_auth_token (proceed with MFA)

2. POST /mfa/initiate (Bearer: mfa_required_auth_token)
   └─> Send OTP to user's email
   └─> Return: encrypted_mfa_code_token

3. POST /mfa/submit (Bearer: mfa_required_auth_token, body: encrypted_mfa_code_token + code)
   └─> Verify OTP
   └─> Set mfa_authenticated_token cookie (30 min, HTTP-only)
   └─> Return: { auth_token, mfa_authenticated_token }
```

### Implementation Changes

**A. New Security Functions** (`app/core/security.py`):
```python
def create_mfa_authenticated_token(username: str) -> str:
    """Create token proving user completed MFA (valid for 30 minutes)"""
    payload = {
        "username": username,
        "type": "mfa_authenticated",
        "exp": datetime.utcnow() + timedelta(seconds=settings.MFA_AUTHENTICATED_TOKEN_EXPIRE_SECONDS)
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def verify_mfa_authenticated_token(token: str, username: str) -> bool:
    """Verify MFA authenticated token is valid and matches username"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("type") == "mfa_authenticated" and payload.get("username") == username
    except:
        return False
```

**B. Configuration** (`app/config.py`):
```python
MFA_AUTHENTICATED_TOKEN_EXPIRE_SECONDS: int = 1800  # 30 minutes
```

**C. Updated `/login` Endpoint** (`app/api/endpoints/auth.py`):
```python
@router.post("/login")
async def login(request: LoginRequest, request_obj: Request):
    # Validate credentials
    users_service.validate_credentials(request.username, request.password)

    # Check for MFA exemption cookie
    mfa_auth_token = request_obj.cookies.get("mfa_authenticated_token")
    if mfa_auth_token and verify_mfa_authenticated_token(mfa_auth_token, request.username):
        # User recently completed MFA, skip to full access
        auth_token = create_auth_token(request.username)
        return {"auth_token": auth_token}

    # Require MFA
    token = create_mfa_required_token(request.username)
    return {"mfa_required_auth_token": token}
```

**D. Updated `/mfa/submit` Endpoint** (`app/api/endpoints/auth.py`):
```python
from fastapi import Response

@router.post("/mfa/submit")
@require_mfa_required_token
async def mfa_submit(request: Request, body: MFASubmitRequest, response: Response):
    username = request.state.token_data["username"]

    # Verify OTP (existing logic)
    otp_from_token = decrypt_token(body.encrypted_mfa_code_token)
    if otp_from_token != body.code:
        raise HTTPException(status_code=400, detail="Invalid MFA code")

    # Generate tokens
    auth_token = create_auth_token(username)
    mfa_authenticated_token = create_mfa_authenticated_token(username)

    # Set 30-minute exemption cookie
    response.set_cookie(
        key="mfa_authenticated_token",
        value=mfa_authenticated_token,
        max_age=1800,  # 30 minutes
        httponly=True,
        secure=True,  # Set False for local dev
        samesite="lax"
    )

    return {
        "auth_token": auth_token,
        "mfa_authenticated_token": mfa_authenticated_token
    }
```

### Security Considerations

1. **HTTP-Only Cookie:** Prevents JavaScript access to MFA token
2. **Secure Flag:** Ensures cookie only sent over HTTPS (disable for local dev)
3. **SameSite:** Protects against CSRF attacks
4. **Username Verification:** Token must match the login username
5. **Short TTL:** 30-minute window balances security and convenience

---

## TBD / Future Work

- **Captcha Image Generation:** How images are sourced, stored, and served. Request/response structure is finalized.

---

## Next Steps

1. Implement base project structure
2. Create Pydantic models (user.py, captcha.py)
3. Implement security utilities (JWT creation, encryption, decorators)
4. Implement users service (auth, payments, classes)
5. Implement captcha service (placeholder for image generation)
6. Create all API endpoints
7. Write tests
8. Integrate email service (when decided)
9. Implement captcha image generation (when decided)
