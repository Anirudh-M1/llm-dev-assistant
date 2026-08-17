import re

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
URL_PATTERN = re.compile(r"^https?://[^\s]+\.[^\s]+$")

def is_valid_email(s):
    return bool(EMAIL_PATTERN.match(s))

def is_valid_url(s):
    return bool(URL_PATTERN.match(s))
