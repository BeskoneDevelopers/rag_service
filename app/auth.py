import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from app.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


class InvalidAuthTokenError(Exception):
    pass


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            key=settings.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            options={"require": ["sub", "company", "exp"]},
        )
        return payload

    except jwt.InvalidTokenError as exc:
        raise InvalidAuthTokenError("Авторизация не пройдена") from exc


async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> dict:
    try:
        payload = decode_token(token)

        if (
            not isinstance(payload.get("sub"), str)
            or not payload["sub"].strip()
            or not isinstance(payload.get("company"), str)
            or not payload["company"].strip()
        ):
            raise HTTPException(
                status_code=401,
                detail="Некорректные данные токена",
                headers={"WWW-Authenticate": "Bearer"},
            )

    except InvalidAuthTokenError as exc:
        raise HTTPException(
            status_code=401,
            detail="Недействительный токен",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return payload
