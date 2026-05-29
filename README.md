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

1. **Setup + Download** -- Set up Colab/Drive/GitHub, download the Metro Interstate Traffic Volume dataset, and explore its traffic, weather, and time columns.
2. **Data Preprocessing** -- Clean bad readings, add time features (hour, day, month, holiday), build a congestion label, split into train/val/test, run EDA, and build a feature table for classical ML plus scaled arrays for the neural networks.
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
      <td>Learn ML basics, tabular/time-series data, and the smart-city sensing problem. Set up Python + Colab + GitHub. Download the traffic dataset and load it into a DataFrame</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Explore the columns (traffic volume, weather, time). Understand how traffic varies by hour of day, day of week, and weather</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Data Preprocessing</b></td>
      <td>3</td>
      <td>Clean bad/error readings (0 K temps, outliers, duplicate timestamps), add time features (hour, day, month, holiday), build a Low/Medium/High congestion label, and split into train/val/test</td>
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
      <td>Final paper, poster, presentation. Discuss limitations (single road sensor, label built from volume) and real-world use</td>
    </tr>
  </tbody>
</table>

## Algorithm Variety

- **Classical ML:** Logistic Regression, KNN, Decision Tree, Naive Bayes, SVM (linear & RBF), Random Forest, XGBoost -- each with hyperparameter variations.
- **Deep learning:** a small neural network (MLP) built from scratch, plus deeper/wider variants with dropout.

## Dataset

- **Metro Interstate Traffic Volume (UCI)** -- ~48,000 hourly readings of westbound I-94 traffic near Minneapolis-St Paul (2012-2018), combining a traffic sensor with weather and holiday data. Columns:
  - `date_time` -- hourly timestamp
  - `traffic_volume` -- vehicles in that hour (used to build the Low/Medium/High congestion label)
  - `temp` (Kelvin), `rain_1h`, `snow_1h`, `clouds_all` -- weather sensors
  - `weather_main`, `weather_description` -- weather category
  - `holiday` -- US national/regional holiday (or blank)
- Source: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume (the notebook downloads `Metro_Interstate_Traffic_Volume.csv.gz` automatically).
- **Engineered features:** temperature in Celsius, hour-of-day, day-of-week, month, an `is_holiday` flag, and one-hot `weather_main`. The congestion label comes from splitting `traffic_volume` into three equal-sized groups with `pd.qcut`.
- **Note:** this is a single road sensor, so it captures one location over many years rather than many zones. With tens of thousands of rows the neural networks can actually train, which makes the classical-vs-deep-learning comparison meaningful. Discuss the single-location limitation in your report.
