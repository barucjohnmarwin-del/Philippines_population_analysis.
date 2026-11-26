PHILIPPINE CITIES POPULATION ANALYTICS SYSTEM
================================================

REQUIREMENTS
- Python 3.10 or newer
- Internet connection for Chart.js and Google Fonts (the Flask app itself runs locally)

INSTALLATION (Windows)
1. Extract this ZIP file.
2. Open the extracted folder.
3. Open Command Prompt in this folder.
4. (Recommended) Create a virtual environment:
   python -m venv venv
   venv\Scripts\activate
5. Install dependencies:
   pip install -r requirements.txt
6. Start the website:
   python app.py
7. Open this address in your browser:
   http://127.0.0.1:5000

IMPORTANT
- Keep cleaned_for_ml.csv in the same folder as app.py.
- The website uses the uploaded cleaned dataset (146 records).
- Classification target: High Population / Low Population, based on the dataset median.
- Classification scores are recalculated from the current data and fixed random split.
- Forecasts are recursive Decision Tree estimates, not official population statistics.
- The notebook's model comparison percentages were static values; this website calculates metrics from the actual current split instead.
- City labels are shown as City ID because the supplied cleaned CSV does not contain city names.
- To stop the local server, press CTRL+C in Command Prompt.

CSV UPLOAD
- Use the "Upload a CSV dataset" panel in the City Dataset section.
- Upload a CSV with these columns: city_id, pop_2010, pop_2015, pop_2020,
  pop_2024, land_area_km2, dens_2010, dens_2015, dens_2020.
- Include at least 8 complete rows. Land area must be greater than zero.
- The upload is limited to 10 MB. The dashboard recalculates its charts,
  classification metrics, ROC curves, confusion matrix, prediction models,
  regression metrics, and forecasts after a valid upload.
- The uploaded data stays active until you stop/restart the Flask server. On
  restart, the app loads cleaned_for_ml.csv again. The upload does not overwrite
  your CSV file on disk.
- ROC curves use distinct colors for Categorical Naive Bayes, Decision Tree,
  Random Forest, and Logistic Regression. The dashed diagonal is the random
  classifier baseline.
