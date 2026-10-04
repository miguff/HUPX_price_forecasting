import json

notebook_path = r"c:\Users\local_user\Documents\Programozás\AIEnergyPrices\statistic_analize.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Clear all cell outputs
for cell in nb['cells']:
    if 'outputs' in cell:
        cell['outputs'] = []
    if 'execution_count' in cell:
        cell['execution_count'] = None

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook outputs cleared")
