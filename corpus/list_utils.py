def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def unique(items):
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen

def chunk_list(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]
