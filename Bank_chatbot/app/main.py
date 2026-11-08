from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.auth import validate_token
from app.database import (
    get_safe_account,
    get_user_by_id,
    get_user_transactions,
)
from app.guardrails import (
    check_input_guardrail,
    check_output_guardrail,
)
from app.llm import generate_answer


app = FastAPI(title="Bank Account Chatbot")

security = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security)
):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    user_id = validate_token(
        f"{credentials.scheme} {credentials.credentials}"
    )

    user = get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


@app.post("/login")
def login(payload: LoginRequest):
    from app.auth import login_user

    return login_user(
        payload.username,
        payload.password,
    )


@app.get("/account")
def account(user: dict = Depends(get_current_user)):
    return get_safe_account(user)


@app.get("/transactions")
def transactions(user: dict = Depends(get_current_user)):
    return {
        "transactions": get_user_transactions(
            user["user_id"]
        )
    }


@app.post("/chat")
def chat(
    payload: ChatRequest,
    user: dict = Depends(get_current_user),
):
    # Check the user's question
    blocked = check_input_guardrail(payload.message)

    if blocked:
        return {
            "answer": blocked
        }

    # Safe account information given to the LLM
    transactions = get_user_transactions(user["user_id"])

    transaction_text = ""

    if transactions:
        for tx in transactions:
            transaction_text += (
                f"Transaction ID: {tx.get('transaction_id', 'N/A')}\n"
                f"Date: {tx.get('date', 'N/A')}\n"
                f"Type: {tx.get('type', 'N/A')}\n"
                f"Amount: ₹{tx.get('amount', 'N/A')}\n"
                f"Description: {tx.get('description', 'N/A')}\n"
                f"Status: {tx.get('status', 'N/A')}\n"
                f"---\n"
        )
    else:
        transaction_text = "No transaction records found for this user."

    safe_context = (
        f"Name: {user['name']}\n"
        f"Account Type: {user['account_type']}\n"
        f"Balance: ₹{int(float(user['balance'])):,}\n"
        f"Account Number: {user['account_number']}\n"
        f"Branch: {user['branch']}\n"
        f"IFSC: {user['ifsc']}\n"
        f"Account Status: {user['account_status']}\n\n"
        f"Transaction Information:\n"
        f"{transaction_text}"
)

    print("BEFORE LLM")

    answer = generate_answer(
        user,
        payload.message,
        safe_context,
    )

    print("AFTER LLM:", answer)

    safe_answer = check_output_guardrail(answer)

    safe_answer = safe_answer.replace("\\n", " ").replace("\n", " ")

    print("FINAL ANSWER:", safe_answer)

    return {
    "answer": safe_answer
}


@app.get("/")
def root():
    return {
        "message": "Bank Account Chatbot API is running."
    }