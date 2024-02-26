import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from joblib import dump, load


def load_json_lines(filename):
    """Load JSON lines from file and skip improper lines."""
    data = []
    with open(filename, "r") as file:
        for line in file:
            try:
                line = line.strip().rstrip(",")
                if line and not line.startswith("#"):
                    json_obj = json.loads(line)
                    data.append(json_obj)
            except json.JSONDecodeError:
                print(f"Skipping line due to JSONDecodeError: {line}")
    return data


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

non_plastics_list = load_json_lines(
    os.path.join(script_dir, "training_data/non_plastic_data.json")
)
plastics_list = load_json_lines(
    os.path.join(script_dir, "training_data/plastic_data.json")
)

non_plastics = pd.DataFrame(non_plastics_list)
plastics = pd.DataFrame(plastics_list)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if not model_loaded:
    # Hyperparameter optimization and training
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 4, 7],
        "min_samples_leaf": [1, 2, 4],
    }
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Save trained model to
    dump(best_model, model_filename)
    print(f"Model trained and saved as {model_filename}")

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

feature_importances = pd.Series(
    best_model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print(feature_importances)

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
    print(single_data_df)

print(f"Optimized Model Accuracy: {accuracy}")
print(f"Non-Plastic Samples: {len(non_plastics_list)}")
print(f"Plastic Samples: {len(plastics_list)}")

probabilities = best_model.predict_proba(single_data_df)
for i, prob in enumerate(probabilities[:, 1]):  # Index 1 for plastic
    if i == 0:
        print(
            f"The confirmed plastic item has a {prob*100:.3f}% predicted probability of being PET/PETE plastic."
        )
    if i == 1:
        print(
            f"The confirmed non-plastic item has a {prob*100:.3f}% predicted probability of being PET/PETE plastic."
        )
