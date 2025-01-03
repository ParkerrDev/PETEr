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


def build_datasets(script_dir):
    """Load and configure training data, returning combined dataset."""

    non_plastics_list = load_csv_data(
        os.path.join(script_dir, "training_data/with_glass/non_plastic_data.csv")
    )
    plastics_list = load_csv_data(
        os.path.join(script_dir, "training_data/with_glass/plastic_data.csv")
    )

    non_plastics_df = pd.DataFrame(non_plastics_list).drop_duplicates()
    plastics_df = pd.DataFrame(plastics_list).drop_duplicates()

    if len(plastics_df) < len(non_plastics_df):
        length_difference = len(non_plastics_df) - len(plastics_df)
        print(f"Balancing non_plastics to plastics: Ignoring {length_difference} samples.")
        non_plastics_df = non_plastics_df.iloc[: len(plastics_df)]

    non_plastics_df["is_plastic"] = 0
    plastics_df["is_plastic"] = 1

    data = pd.concat([non_plastics_df, plastics_df], ignore_index=True)
    return data, non_plastics_df, plastics_df


def preprocess_data(data):
    """Apply imputation and prepare features and labels."""
    imputer = SimpleImputer(strategy="mean")
    data_imputed = pd.DataFrame(
        imputer.fit_transform(data.drop(columns="is_plastic")),
        columns=data.drop(columns="is_plastic").columns,
    )
    data_imputed["is_plastic"] = data["is_plastic"].values

    X = data_imputed.drop("is_plastic", axis=1)
    y = data_imputed["is_plastic"]
    return X, y


def train_model(X_train, y_train):
    """Train the model using GridSearchCV for hyperparameter tuning."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, X_columns):
    """Evaluate the trained model and print feature importances."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    feature_importances = pd.Series(model.feature_importances_, index=X_columns).sort_values(
        ascending=False
    )
    return accuracy, feature_importances


def print_feature_importances(feature_importances):
    """Print feature importances with associated nanometer dictionary info."""
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


def test_samples(model, plastics_list, non_plastics_list):
    """Test confirmed plastic and non-plastic samples, then print results."""
    confirmed_plastic = '{"A":0.87,"B":12.54,"C":118.20,"D":41.07,"E":27.13,"F":14.71,"G":3.53,"H":3.34,"R":7.03,"I":4.47,"S":1.96,"J":0.89,"T":0.79,"U":0.81,"V":0.86,"W":1.06,"K":0.73,"L":0.00}'
    confirmed_non_plastic = None
    samples = []

    if confirmed_plastic:
        sample1 = json.loads(confirmed_plastic)
        samples.append(sample1)

    if confirmed_non_plastic:
        sample2 = json.loads(confirmed_non_plastic)
        samples.append(sample2)

    if samples:
        single_data_df = pd.DataFrame(samples)
        print(single_data_df)
        probabilities = model.predict_proba(single_data_df)
        for i, prob in enumerate(probabilities[:, 1]):  # Index 1 for plastic
            if i == 0:
                print(f"The confirmed plastic item has a {prob*100:.3f}% predicted probability of being PET/PETE plastic.")
            if i == 1:
                print(f"The confirmed non-plastic item has a {prob*100:.3f}% predicted probability of being PET/PETE plastic.")

    total_samples = int(len(plastics_list) + len(non_plastics_list))
    print(f"Total Samples: {total_samples}")
    print(f"Minutes Training of Training Data: {int(total_samples/60)}")


def main():
    """Main function to run the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load or prepare model
    model_filename = f"{script_dir}/saved_models/trained_model.joblib"
    best_model, model_loaded = load_model(model_filename)

    # Build datasets
    data, non_plastics_df, plastics_df = build_datasets(script_dir)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train new model if none loaded
    if not model_loaded:
        best_model = train_model(X_train, y_train)
        dump(best_model, model_filename)
        print(f"Model trained and saved as {model_filename}")

    # Evaluate model
    accuracy, feature_importances = evaluate_model(best_model, X_test, y_test, X.columns)
    print(f"Optimized Model Accuracy: {accuracy}")

    # Print non-plastic and plastic sample counts
    print(f"Non-Plastic Samples: {len(non_plastics_df)}")
    print(f"Plastic Samples: {len(plastics_df)}")

    # Print feature importances
    print_feature_importances(feature_importances)

    # Test confirmed samples
    test_samples(best_model, plastics_df.to_dict("records"), non_plastics_df.to_dict("records"))


if __name__ == "__main__":
    main()