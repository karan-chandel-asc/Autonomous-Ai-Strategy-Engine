import re
from pydantic import BaseModel, field_validator, model_validator


def success_response(data=None, message=""):
    return {
        "success": True,
        "message": message,
        "data":    data,
    }


def error_response(message="", data=None):
    return {
        "success": False,
        "message": message,
        "data":    data,
    }


class LoginSchema(BaseModel):
    email: str
    password: str

    @field_validator("email")
    def validate_email(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Email is required")
        return value.strip().lower()

    @field_validator("password")
    def validate_password(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Password is required")
        return value

class ForgotPasswordSchema(BaseModel):
    email: str

    @field_validator("email")
    def validate_email(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Email is required")
        return value.strip().lower()


class ResetPasswordSchema(BaseModel):
    uid: str
    token: str
    password: str
    confirm_password: str

    @field_validator("password")
    def validate_password(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Password is required")
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        return value

    @field_validator("confirm_password")
    def validate_confirm_password(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Confirm password is required")
        return value

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class SignupSchema(BaseModel):
    full_name: str
    email: str
    password: str
    confirm_password: str

    @field_validator("full_name")
    def validate_full_name(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Full name is required")
        return value.strip()

    @field_validator("email")
    def validate_email(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Email is required")
        email = value.strip().lower()
        if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", email):
            raise ValueError("Enter a valid email address")
        return email

    @field_validator("password")
    def validate_password(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Password is required")
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        return value

    @field_validator("confirm_password")
    def validate_confirm_password(cls, value):
        if not value or str(value).strip() == "":
            raise ValueError("Confirm password is required")
        return value

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self
