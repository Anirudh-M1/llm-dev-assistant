def reverse_string(s):
    return s[::-1]

def capitalize_words(text):
    return text.title()

def is_palindrome(s):
    normalized = s.lower().replace(" ", "")
    return normalized == normalized[::-1]

def count_vowels(s):
    return sum(1 for ch in s.lower() if ch in "aeiou")