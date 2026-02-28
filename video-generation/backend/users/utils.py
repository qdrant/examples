import os
from cryptography.fernet import Fernet, InvalidToken

def get_fernet():
    key = os.getenv('DJANGO_ENCRYPTION_KEY')
    print("🔐 Loaded Fernet key:")
    if not key:
        raise ValueError("Missing DJANGO_ENCRYPTION_KEY in environment.")
    return Fernet(key)

def encrypt_value(value):
    if not value:
        return None
    f = get_fernet()
    return f.encrypt(value.encode()).decode()

def decrypt_value(value):
    if not value:
        return None
    try:
        f = get_fernet()
        return f.decrypt(value.encode()).decode()
    except InvalidToken:
        return "[DECRYPTION_FAILED]"
