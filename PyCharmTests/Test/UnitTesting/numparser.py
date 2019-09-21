def parse(txt):
    result = []
    for num in txt.split(','):
        result.append(int(num))
    return result

