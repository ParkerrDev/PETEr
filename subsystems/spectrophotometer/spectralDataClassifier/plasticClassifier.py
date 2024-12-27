import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from joblib import dump, load


def load_csv_data(filename):
    """Load data from CSV file."""
    try:
        data = pd.read_csv(filename)
        return data.to_dict('records')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []

def load_model(model_filename):
    """Check for a saved model and load it."""
    try:
        model = load(model_filename)
        print("Model loaded successfully.")
        return model, True
    except FileNotFoundError:
        print("No pre-trained model found. Proceeding to train a new model.")
        return None, False

script_dir = os.path.dirname(os.path.abspath(__file__))

model_filename = f"{script_dir}/trained_model.joblib"
best_model, model_loaded = load_model(model_filename)

# Load data from CSV files
non_plastics_list = load_csv_data(
    os.path.join(script_dir, "training_data/with_glass/non_plastic_data.csv")
)
plastics_list = load_csv_data(
    os.path.join(script_dir, "training_data/with_glass/plastic_data.csv")
)

# Convert lists to DataFrames to easily drop duplicates
non_plastics_df = pd.DataFrame(non_plastics_list).drop_duplicates()
plastics_df = pd.DataFrame(plastics_list).drop_duplicates()

# Check lengths and balance if necessary
if len(plastics_df) < len(non_plastics_df):
    length_difference = len(non_plastics_df) - len(plastics_df)
    print(f"Balancing non_plastics to plastics: Ignoring {length_difference} samples.")
    # Note: This reduces the non_plastics_df without considering the potential importance of each record
    non_plastics_df = non_plastics_df.iloc[: len(plastics_df)]

# If you need to work with lists later on
non_plastics_list = non_plastics_df.to_dict("records")
plastics_list = plastics_df.to_dict("records")

# Continuing with DataFrames for further operations is recommended if possible
non_plastics = non_plastics_df
plastics = plastics_df

non_plastics["is_plastic"] = 0
plastics["is_plastic"] = 1

data = pd.concat([non_plastics, plastics], ignore_index=True)

imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(
    imputer.fit_transform(data.drop(columns="is_plastic")),
    columns=data.drop(columns="is_plastic").columns,
)
data_imputed["is_plastic"] = data["is_plastic"].values

X = data_imputed.drop("is_plastic", axis=1)
y = data_imputed["is_plastic"]
# Apply row-wise mean centering
X_centered = X.sub(X.mean(axis=1), axis=0)

X_train_centered, X_test_centered, y_train, y_test = train_test_split(
    X_centered, y, test_size=0.2, random_state=42
)

if not model_loaded:
    # Hyperparameter optimization and training
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train_centered, y_train)

    best_model = grid_search.best_estimator_

    # Save trained model to
    dump(best_model, model_filename)
    print(f"Model trained and saved as {model_filename}")

y_pred = best_model.predict(X_test_centered)
accuracy = accuracy_score(y_test, y_pred)

feature_importances = pd.Series(
    best_model.feature_importances_, index=X.columns
).sort_values(ascending=False)


nanometer_dictionary = {
    "A": "410nm",
    "B": "435nm",
    "C": "460nm",
    "D": "485nm",
    "E": "510nm",
    "F": "535nm",
    "G": "560nm",
    "H": "585nm",
    "R": "610nm",
    "I": "645nm",
    "S": "680nm",
    "J": "705nm",
    "T": "730nm",
    "U": "760nm",
    "V": "810nm",
    "W": "860nm",
    "K": "900nm",
    "L": "940nm",
}

print("Predicted Feature Importances: ")
for feature, importance in feature_importances.items():
    nanometer = nanometer_dictionary.get(feature[0])
    print(f"{feature}: {importance}: {nanometer}")


confirmed_plastic = '{"A":0.87,"B":12.54,"C":118.20,"D":41.07,"E":27.13,"F":14.71,"G":3.53,"H":3.34,"R":7.03,"I":4.47,"S":1.96,"J":0.89,"T":0.79,"U":0.81,"V":0.86,"W":1.06,"K":0.73,"L":0.00}'
confirmed_non_plastic = None
samples = []


if confirmed_plastic != "" and confirmed_plastic is not None:
    sample1 = json.loads(confirmed_plastic)
    samples.append(sample1)

if confirmed_non_plastic != "" and confirmed_non_plastic is not None:
    sample2 = json.loads(confirmed_non_plastic)
    samples.append(sample2)

if samples:
    single_data_df = pd.DataFrame(samples)
    single_data_df_centered = single_data_df.sub(single_data_df.mean(axis=1), axis=0)
    print(single_data_df)

total_samples = int(len(plastics_list) + len(non_plastics_list))

print(f"Optimized Model Accuracy: {accuracy}")
print(f"Non-Plastic Samples: {len(non_plastics_list)}")
print(f"Plastic Samples: {len(plastics_list)}")
print(f"Total Samples: {total_samples}")
print(f"Minutes Training of Training Data: {int(total_samples/60)}")


probabilities = best_model.predict_proba(single_data_df_centered)
for i, prob in enumerate(probabilities[:, 1]):  # Index 1 for plastic
    if i == 0:
        print(
            f"The confirmed plastic item has a {prob*100:.3f}% predicted probability of being PET/PETE plastic."
        )
    if i == 1:
        print(
            f"The confirmed non-plastic item has a {prob*100:.3f}% predicted probability of being PET/PETE plastic."
        )
