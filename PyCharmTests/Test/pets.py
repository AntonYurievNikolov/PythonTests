def remove_pet(kennel, name):
    for other in kennel:
        if name == other:
            kennel.remove(other)
    return kennel


kennel = ['Snoopy', 'Fido', 'Fido', 'Pluto']
result = remove_pet(kennel, 'Fido')
print(result)
