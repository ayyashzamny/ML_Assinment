import json
import os

def add_demo_cell(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    # Read the notebook
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Remove existing demo cell if it exists to update it
    nb['cells'] = [cell for cell in nb['cells'] if not (cell['cell_type'] == 'markdown' and "Interactive Prediction Demo (Viva)" in "".join(cell['source']))]
    nb['cells'] = [cell for cell in nb['cells'] if not (cell['cell_type'] == 'code' and "VIVA DEMO RESULTS" in "".join(cell.get('source', [])))]

    # Create the demo cell
    demo_markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Interactive Prediction Demo (Viva)\n",
            "Use this section to demonstrate the model's prediction on a single game scenario."
        ]
    }

    demo_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "# 1. Select a random game from the test set for demonstration\n",
            "random_idx = np.random.randint(0, len(X_test))\n",
            "sample_game = X_test.iloc[random_idx]\n",
            "actual_result = y_test.iloc[random_idx]\n",
            "\n",
            "print('--- Game Scenario Features ---')\n",
            "print(sample_game)\n",
            "print('-' * 30)\n",
            "\n",
            "# 2. Transform the sample for prediction\n",
            "# Note: scaler.transform expects a 2D array-like input\n",
            "sample_scaled = scaler.transform([sample_game.values])\n",
            "\n",
            "# 3. Make prediction\n",
            "# Automatically find the available model variable\n",
            "model = None\n",
            "for var_name in ['best_rf', 'best_svm', 'knn_best', 'best_knn', 'rf_model', 'svm_model', 'knn_model']:\n",
            "    if var_name in globals():\n",
            "        model = globals()[var_name]\n",
            "        break\n",
            "\n",
            "if model is None:\n",
            "    print(\"Error: No trained model found in global scope.\")\n",
            "else:\n",
            "    prediction = model.predict(sample_scaled)[0]\n",
            "    proba = model.predict_proba(sample_scaled)[0]\n",
            "\n",
            "    # 4. Display Results\n",
            "    print(f'\\n[VIVA DEMO RESULTS]')\n",
            "    if prediction == 1:\n",
            "        print('>>> PREDICTION: HOME TEAM WINS! 🏀')\n",
            "        print(f'Confidence Score: {proba[1]*100:.2f}%')\n",
            "    else:\n",
            "        print('>>> PREDICTION: AWAY TEAM WINS! 🏆')\n",
            "        print(f'Confidence Score: {proba[0]*100:.2f}%')\n",
            "\n",
            "    result_text = \"Home Win\" if actual_result == 1 else \"Away Win\"\n",
            "    status = \"✅ CORRECT\" if prediction == actual_result else \"❌ INCORRECT\"\n",
            "    print(f'Actual Result: {result_text} ({status})')"
        ]
    }

    # Append to the end of the notebook
    nb['cells'].append(demo_markdown_cell)
    nb['cells'].append(demo_code_cell)

    # Write back
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"Updated live demo cell in {filename}")

if __name__ == "__main__":
    add_demo_cell('notebook1_random_forest.ipynb')
    add_demo_cell('notebook2_svm.ipynb')
    add_demo_cell('notebook3_knn.ipynb')
