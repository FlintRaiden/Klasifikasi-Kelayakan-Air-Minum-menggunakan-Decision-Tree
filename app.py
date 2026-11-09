import json
import warnings
import os

import joblib
import numpy as np
from flask import Flask, render_template, request

warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

model = joblib.load(os.path.join(BASE_DIR, "model", "decision_tree_model.joblib"))
with open(os.path.join(BASE_DIR, "model", "model_meta.json")) as f:
    meta = json.load(f)

FEATURES = meta["features"]

# Label, satuan, dan contoh nilai untuk setiap fitur pada formulir
FIELD_INFO = {
    "ph": ("pH", "", "7.0"),
    "Hardness": ("Hardness (kesadahan)", "mg/L", "196"),
    "Solids": ("Solids (total padatan terlarut)", "ppm", "20900"),
    "Chloramines": ("Chloramines", "ppm", "7.1"),
    "Sulfate": ("Sulfate", "mg/L", "333"),
    "Conductivity": ("Conductivity (konduktivitas)", "μS/cm", "420"),
    "Organic_carbon": ("Organic Carbon", "ppm", "14.2"),
    "Trihalomethanes": ("Trihalomethanes", "μg/L", "66.6"),
    "Turbidity": ("Turbidity (kekeruhan)", "NTU", "3.9"),
}


def build_fields():
    fields = []
    for name in FEATURES:
        label, unit, example = FIELD_INFO.get(name, (name, "", ""))
        lo, hi = meta["ranges"][name]
        fields.append({"name": name, "label": label, "unit": unit, "example": example,
                       "min": round(lo, 2), "max": round(hi, 2)})
    return fields


@app.route("/", methods=["GET", "POST"])
def index():
    fields = build_fields()
    result, error, values = None, None, {}

    if request.method == "POST":
        values = {f["name"]: request.form.get(f["name"], "").strip() for f in fields}
        try:
            row = [float(values[name].replace(",", ".")) for name in FEATURES]
            if any(not np.isfinite(v) for v in row):
                raise ValueError
        except ValueError:
            error = "Semua kolom wajib diisi dengan angka yang valid."
        else:
            X = np.array([row])
            pred = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            result = {
                "label": "Layak Minum" if pred == 1 else "Tidak Layak Minum",
                "potable": pred == 1,
                "prob_layak": round(float(proba[list(model.classes_).index(1)]) * 100, 1),
                "prob_tidak": round(float(proba[list(model.classes_).index(0)]) * 100, 1),
            }

    return render_template("index.html", fields=fields, result=result, error=error,
                           values=values, acc=round(meta["accuracy_test"] * 100, 2),
                           depth=meta["max_depth"])


if __name__ == "__main__":
    app.run(debug=True)
