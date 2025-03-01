# Mushroom Classification

This project applies machine learning models to classify mushrooms, focusing on decision trees, information gain, pruning, and random forests.

## Project Structure
```
Mushroom Classification/
│── data/               # Data files
│   ├── mushrooms-example.csv      # Sample dataset
│   ├── mushrooms-full_data.csv    # Full dataset
│── src/                # Python scripts
│   ├── 2.3final_decision_tree.py
│   ├── 2.3information_gain.py
│   ├── 3Pruning.py
│   ├── 6.3an_example_of_random_forest.py
│── README.md           # Project documentation
│── requirements.txt    # Dependency list
│── .gitignore          # Git ignore file
```

## Dataset
This project uses a publicly available dataset.

- **mushrooms-example.csv**: A small sample of the dataset for quick testing.
- **mushrooms-full_data.csv**: The complete dataset.

## How to Run

**Install dependencies**
```
pip install -r requirements.txt
```

**Run Decision Tree Model**
```
- python src/2.3final_decision_tree.py
```

**Calculate Information Gain**
```
- python src/2.3information_gain.py
```

**Run Pruning Algorithm**
```
- python src/3Pruning.py
```

**Run Random Forest Model**
```
- python src/6.3an_example_of_random_forest.py
```

## Project Goals
- **Data Preprocessing**: Clean and transform the mushroom dataset.
- **Decision Tree Implementation**: Apply decision trees for classification.
- **Information Gain Calculation**: Analyze feature importance.
- **Pruning Methods**: Optimize tree structures.
- **Random Forest Model**: Improve classification performance.

## Dependencies
This project uses the following Python libraries:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## License
This project is open-source under the MIT License. Contributions are welcome!

