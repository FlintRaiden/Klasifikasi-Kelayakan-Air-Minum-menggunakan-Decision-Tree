# Klasifikasi Kelayakan Air Minum (Decision Tree + Flask

Muhamad Rafli Ardiansyah (301240028), Kelas 5B.

## Struktur
```
water-potability/
├── app.py                      # aplikasi Flask (entrypoint Vercel)
├── requirements.txt
├── .gitignore / .vercelignore
├── templates/index.html        # formulir input + hasil prediksi
├── model/
│   ├── decision_tree_model.joblib   # Model B (max_depth=4)
│   └── model_meta.json              # urutan fitur, rentang, akurasi
├── notebook/water_potability_decision_tree.ipynb   # EDA, Model A/B, evaluasi, simpan model
├── data/water_potability.csv
└── images/                     # hasil plot dari notebook
```

## Menjalankan lokal
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter pandas matplotlib seaborn
jupyter notebook notebook/water_potability_decision_tree.ipynb   # Run All (menghasilkan ulang model)
python app.py                                                    # buka http://127.0.0.1:5000
```
> Versi `scikit-learn` di `requirements.txt` harus sama dengan versi yang dipakai saat melatih model
> (`python -c "import sklearn; print(sklearn.__version__)"`), agar file `.joblib` bisa dimuat di Vercel.

## Deploy: GitHub → Vercel
```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```
1. Buka vercel.com → **Add New… → Project** → impor repositori GitHub tersebut.
2. Framework Preset: **Flask** (terdeteksi otomatis dari `app.py`) → **Deploy**.
3. Setiap `git push` ke `main` akan memicu deploy ulang otomatis.
