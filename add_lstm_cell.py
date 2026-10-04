import json

notebook_path = r"c:\Users\local_user\Documents\Programozás\AIEnergyPrices\statistic_analize.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Add new cell at the end with LSTM visualization
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": "%run viz_lstm_only.py"
}

nb['cells'].append(new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook updated: LSTM cell added")
