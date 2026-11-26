import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.svm import SVR
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, mean_squared_error, r2_score)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB CSV limit
DATA_PATH = os.path.join(os.path.dirname(__file__), "cleaned_for_ml.csv")
REQUIRED_COLUMNS = ["city_id", "pop_2010", "pop_2015", "pop_2020", "pop_2024",
                    "land_area_km2", "dens_2010", "dens_2015", "dens_2020"]

@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "The CSV file is too large. Maximum upload size is 10 MB."}), 413

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("cleaned_for_ml.csv was not found beside app.py")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    return df

df = load_data()

def validate_dataset(frame):
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    frame = frame.copy()
    numeric = REQUIRED_COLUMNS
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric).reset_index(drop=True)
    if len(frame) < 8:
        raise ValueError("Please upload at least 8 complete city rows to train and evaluate the models.")
    if (frame[["land_area_km2"]] <= 0).any().any():
        raise ValueError("Land area values must be greater than zero.")
    return frame

def train_models(frame):
    data = frame.copy()
    numeric = ["pop_2010", "pop_2015", "pop_2020", "pop_2024", "land_area_km2", "dens_2020"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=numeric).copy()

    # Regression model: predict 2024 population from earlier population and land area.
    features_reg = ["pop_2010", "pop_2015", "pop_2020", "land_area_km2"]
    Xr = data[features_reg]
    yr = data["pop_2024"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    sx, sy = StandardScaler(), StandardScaler()
    X_train_scaled = sx.fit_transform(X_train_r)
    X_test_scaled = sx.transform(X_test_r)
    y_train_scaled = sy.fit_transform(y_train_r.to_numpy().reshape(-1, 1)).ravel()
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train_scaled)
    pred_scaled = svr.predict(X_test_scaled)
    pred_r = sy.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_test_r, pred_r)))
    r2 = float(r2_score(y_test_r, pred_r))

    # Classification: predict whether 2024 population is above or below the dataset median.
    cat = data.copy()
    for col, labels in [
        ("pop_2020", ["Low", "Medium-Low", "Medium-High", "High"]),
        ("land_area_km2", ["Small", "Medium-Small", "Medium-Large", "Large"]),
        ("dens_2020", ["Sparse", "Moderate", "Dense", "Very Dense"])
    ]:
        cat[col + "_bin"] = pd.qcut(cat[col], q=4, labels=labels, duplicates="drop")
    cat["pop_2024_class"] = np.where(
        cat["pop_2024"] >= cat["pop_2024"].median(), "High Population", "Low Population"
    )
    feature_cols = ["pop_2020_bin", "land_area_km2_bin", "dens_2020_bin"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = enc.fit_transform(cat[feature_cols].astype(str))
    target_enc = LabelEncoder()
    y = target_enc.fit_transform(cat["pop_2024_class"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    estimators = {
        "Categorical Naive Bayes": CategoricalNB(alpha=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    results, model_objects = {}, {}
    for name, model in estimators.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "auc": float(auc(fpr, tpr)),
            "confusion_matrix": cm.tolist(),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        }
        model_objects[name] = model

    return {
        "regression": {"rmse": rmse, "r2": r2},
        "classification": results,
        "models": model_objects,
        "encoder": enc,
        "target_encoder": target_enc,
        "feature_cols": feature_cols
    }

try:
    ML = train_models(df)
    MODEL_ERROR = None
except Exception as exc:
    ML = None
    MODEL_ERROR = str(exc)

def forecast_population(frame):
    # Follows the notebook's recursive Decision Tree forecasting approach.
    X, y = [], []
    for _, row in frame.iterrows():
        vals = [row["pop_2010"], row["pop_2015"], row["pop_2020"], row["pop_2024"]]
        if all(pd.notna(v) for v in vals):
            X.append(vals[:3])
            y.append(vals[3])
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(np.asarray(X), np.asarray(y))
    output = []
    for _, row in frame.iterrows():
        history = [float(row["pop_2015"]), float(row["pop_2020"]), float(row["pop_2024"])]
        record = {"city_id": int(row["city_id"]), "actual_2024": float(row["pop_2024"])}
        for year in range(2025, 2031):
            prediction = max(0, float(model.predict([history])[0]))
            record[str(year)] = round(prediction)
            history = history[1:] + [prediction]
        output.append(record)
    return output

FORECAST = forecast_population(df)

def json_records(frame):
    return frame.replace({np.nan: None}).to_dict(orient="records")

@app.route("/")
def index():
    return render_template("index.html", model_error=MODEL_ERROR)

@app.route("/api/dashboard")
def dashboard():
    records = json_records(df)
    total = int(len(df))
    total_pop = int(df["pop_2024"].sum())
    average_pop = int(df["pop_2024"].mean())
    top = df.nlargest(10, "pop_2024")[["city_id", "pop_2024"]].to_dict("records")
    return jsonify({
        "total_cities": total, "total_population": total_pop,
        "average_population": average_pop,
        "years": ["2010", "2015", "2020", "2024"],
        "population_totals": [int(df[c].sum()) for c in ["pop_2010", "pop_2015", "pop_2020", "pop_2024"]],
        "top_cities": top, "records": records,
        "forecast": FORECAST,
        "models": ML["classification"] if ML else {},
        "regression": ML["regression"] if ML else None,
        "model_error": MODEL_ERROR
    })

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    global df, ML, MODEL_ERROR, FORECAST
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a CSV file to upload."}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file."}), 400
    try:
        candidate = validate_dataset(pd.read_csv(upload, encoding="utf-8-sig"))
        candidate_ml = train_models(candidate)
        candidate_forecast = forecast_population(candidate)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    # Replace the active dataset only after validation and model training succeed.
    df, ML, MODEL_ERROR, FORECAST = candidate, candidate_ml, None, candidate_forecast
    top = df.nlargest(10, "pop_2024")[["city_id", "pop_2024"]].to_dict("records")
    return jsonify({
        "message": f"Loaded {len(df)} complete city rows from {secure_filename(upload.filename)}.",
        "total_cities": int(len(df)), "total_population": int(df["pop_2024"].sum()),
        "average_population": int(df["pop_2024"].mean()),
        "years": ["2010", "2015", "2020", "2024"],
        "population_totals": [int(df[c].sum()) for c in ["pop_2010", "pop_2015", "pop_2020", "pop_2024"]],
        "top_cities": top, "records": json_records(df), "forecast": FORECAST,
        "models": ML["classification"], "regression": ML["regression"], "model_error": None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if not ML:
        return jsonify({"error": MODEL_ERROR or "Models are unavailable."}), 500
    body = request.get_json(silent=True) or {}
    try:
        pop2020 = float(body["pop_2020"])
        area = float(body["land_area_km2"])
        density = float(body["dens_2020"])
        if min(pop2020, area, density) < 0 or area == 0:
            raise ValueError("Values must be positive; land area must be greater than zero.")
        # Bin user values using the training dataset quartile boundaries.
        q_pop = pd.qcut(df["pop_2020"], q=4, retbins=True, duplicates="drop")[1]
        q_area = pd.qcut(df["land_area_km2"], q=4, retbins=True, duplicates="drop")[1]
        q_den = pd.qcut(df["dens_2020"], q=4, retbins=True, duplicates="drop")[1]
        def bin_value(value, edges, names):
            idx = int(np.clip(np.searchsorted(edges[1:-1], value, side="right"), 0, len(names)-1))
            return names[idx]
        features = [[
            bin_value(pop2020, q_pop, ["Low", "Medium-Low", "Medium-High", "High"]),
            bin_value(area, q_area, ["Small", "Medium-Small", "Medium-Large", "Large"]),
            bin_value(density, q_den, ["Sparse", "Moderate", "Dense", "Very Dense"])
        ]]
        encoded = ML["encoder"].transform(pd.DataFrame(features, columns=ML["feature_cols"]).astype(str))
        predictions = {}
        for name, model in ML["models"].items():
            label = ML["target_encoder"].inverse_transform(model.predict(encoded))[0]
            probability = float(max(model.predict_proba(encoded)[0]))
            predictions[name] = {"prediction": label, "confidence": probability}
        return jsonify({"predictions": predictions})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/api/forecast")
def forecast():
    return jsonify(FORECAST)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
