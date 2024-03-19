import os
raw_dir = os.getcwd()
# Step 1: Create a Python dictionary
relations_dict = {
    '/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor': 0,
    '/people/person/nationality': 1,
    '/people/person/profession': 2
}

with open(os.path.join(raw_dir, 'relations.dict'), 'w') as f:
    for key, value in relations_dict.items():
        # Note that the original reading code assumes value is the original key, and key is the original value
        f.write(f"{value}\t{key}\n")


entities_dict = {
    '/m/017dcd': 0,
    '/m/06v8s0': 1,
    '/m/09c7w0': 2,
    '/m/0np9r': 3
}

with open(os.path.join(raw_dir, 'entities.dict'), 'w') as f:
    for key, value in entities_dict.items():
        # Note that the original reading code assumes value is the original key, and key is the original value
        f.write(f"{value}\t{key}\n")