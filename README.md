# Traffic Congestion Classification: Comparing ML Algorithms

A 10-week summer research project for high school students.

Title: Comparing Classical Machine Learning and Deep Learning Models for Traffic Congestion Classification on Smart City Sensor Data.

Research question: How do classical ML algorithms (e.g. SVM, Random Forest, XGBoost) compare with deep learning models (neural networks) for classifying traffic congestion level (Low / Medium / High) from multi-sensor smart-city data, in terms of accuracy, training time, and model size?

## How to start

1. Open `traffic_congestion_classification.ipynb` -- or launch it directly in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VietNguyen705/smart-city-highschool/blob/main/traffic_congestion_classification.ipynb)
2. (Optional) Switch the runtime to GPU (Runtime -> Change runtime type -> T4 GPU). The neural networks are tiny, so a CPU runtime is fine too.
3. Fill in every TODO cell from top to bottom.

Ask your instructor if stuck.

## 4 Big Steps

1. **Setup + Download** -- Set up Colab/Drive/GitHub, download the smart-city sensor CSVs, and explore the four sensor tables (traffic, air quality, weather, energy).
2. **Data Preprocessing** -- Clean bad readings, time-align the sensors into one table, build a congestion label, split into train/val/test, run EDA, and build a feature table for classical ML plus scaled arrays for the neural networks.
3. **Training & Evaluation** -- Train and compare many algorithms: classical ML (Logistic Regression, KNN, Decision Tree, Naive Bayes, SVM, Random Forest, XGBoost) and deep learning (a small neural network plus deeper/wider variants), then consolidate metrics and error analysis.
4. **Write Report** -- Draft the paper, polish figures, build a prediction demo, and prepare the poster and presentation.

## 10-Week Plan

<table>
  <thead>
    <tr><th>Step</th><th>Week</th><th>Tasks</th></tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Setup + Download</b></td>
      <td>1</td>
      <td>Learn ML basics, tabular/time-series data, and the smart-city sensing problem. Set up Python + Colab + GitHub. Download the sensor CSVs and load your first table</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Explore all four sensor tables. Understand the columns, the zones, the status flags, and how the readings relate to each other</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Data Preprocessing</b></td>
      <td>3</td>
      <td>Clean bad/ERROR readings, time-align traffic with weather and air quality (one row per reading), build a Low/Medium/High congestion label, and split into train/val/test</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Build the classical-ML feature table (sensor values + zone), scale features for the neural networks, and save reusable feature arrays. EDA: class balance, feature correlations</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Training &amp; Evaluation</b></td>
      <td>5</td>
      <td>Classical round 1: Logistic Regression, KNN, Decision Tree, Naive Bayes. Record accuracy and confusion matrices</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Classical round 2: SVM (linear/RBF), Random Forest, XGBoost. Try hyperparameter variations</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Build a small neural network (MLP) from scratch. Plot training curves</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Deeper/wider network variants (more layers, dropout). Compare all models (accuracy, F1, training time, model size) + error analysis</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Write Report</b></td>
      <td>9</td>
      <td>Draft the report (intro, dataset, methods, results, discussion). Build a prediction demo (enter sensor readings -&gt; predicted congestion)</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Final paper, poster, presentation. Discuss limitations (small dataset!) and real-world use</td>
    </tr>
  </tbody>
</table>

## Algorithm Variety

- **Classical ML:** Logistic Regression, KNN, Decision Tree, Naive Bayes, SVM (linear & RBF), Random Forest, XGBoost -- each with hyperparameter variations.
- **Deep learning:** a small neural network (MLP) built from scratch, plus deeper/wider variants with dropout.

## Dataset

- **Smart City Sensor Data 2026** -- a simulated multi-sensor dataset with four hourly sensor feeds, each tagged by zone (Zone_A/B/C) and a status flag:
  - **Traffic:** `timestamp, device_id, location, traffic_count, status`
  - **Air quality:** `timestamp, device_id, location, pm25, co2, temperature, humidity, status`
  - **Weather:** `timestamp, device_id, location, temperature, humidity, wind_speed, precipitation, status`
  - **Energy:** `timestamp, device_id, location, energy_usage, power_factor, voltage, current, status`
- GitHub source: https://github.com/rauffauzanrambe/smart-city-sensor-data-2026 (use the CSVs in `data/raw`).
- **Note:** these CSVs are small (tens of rows each). That is fine for learning the full ML pipeline end to end, but it means results are noisy and deep models will tend to overfit. Understanding *why* small data limits deep learning is one of the goals of this project, so write about it in your report.
