# Anxiety & Panic Attack Analysis

**Language:** Python    
**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `graphviz`, `scipy`  
**Dataset:** Panic Attacks — 1,200+ observations, ages 18–64, Kaggle  
**Target Variables:** Panic Attack Frequency, Panic Attack Severity  
**Group Project:** Completed as part of a collaborative team effort

---

## Objective

- Identify key factors that contribute to anxiety and panic attacks  
- Explore lifestyle choices and pre-existing conditions  
- Analyze predictive behaviors for frequency and severity of panic attacks  

---

## Data Description

- **Observations:** 1,200+  
- **Columns:** 34  
- **Gender Distribution:** Female ~45%, Male ~44%, Non-binary ~11%  
- **Target Variables:** Panic Attack Frequency, Panic Attack Severity  
- **Predictors:** Sleep hours, caffeine intake, exercise frequency, alcohol consumption, smoking, therapy, medical history (anxiety, depression, PTSD), triggers, and symptoms  

**Data Cleaning Steps:**  
1. Dropped irrelevant columns and reconstructed categorical features from one-hot encoding  
2. Converted numeric columns (`Symptom_Count`, `Sleep_Hours`)  
3. Categorized heart rate and high-risk individuals based on sleep, caffeine, and medical history  
4. Merged datasets for full analysis  

---

## Exploratory Data Analysis (EDA)

**Trigger Prevalence (~ evenly distributed):**  
- Caffeine ~17%  
- Stress ~16%  
- PTSD ~17%  
- Social Anxiety ~16%  
- Phobia ~17%  
- Unknown ~17%  

**Symptom Prevalence:**  
- Sweating ~25%  
- Shortness of Breath ~23%  
- Dizziness ~19%  
- Trembling ~18%  
- Chest Pain ~15%  

**High-Severity Patterns:**  
- Sweating and shortness of breath are the most common symptoms of panic attacks  
- History of PTSD, anxiety, or depression is linked to higher severity and frequency  
- High-risk individuals identified as: Sleep <5 hrs, Caffeine >3 cups/day, and history of anxiety/depression/PTSD  

**Lifestyle Insights:**  
- Slight tendency for increased sleep hours to reduce frequency  
- Individuals with anxiety history consume more caffeine (≈3 cups/day) vs. ≈2 cups/day without history  
- Recommended sleep: 7 hours  

**Visuals:**
- KDE of panic attack severity by gender  
- Count plot of sleep hours  
- Co-occurrence heatmap of triggers & symptoms  

---

## Chi-Squared Analysis

- Tested symptom co-occurrence and symptom-trigger relationships  
- Significant relationships identified between certain symptoms and triggers (e.g., chest pain and PTSD, caffeine, phobia, stress)  

---

## Regression Trees

### Severity Prediction

**Target:** Panic Attack Severity (Scale 1–5; 1=low, 5=high)  
**Predictors:** Caffeine Intake, Hours of Sleep, Exercise Frequency, Alcohol Consumption  
**Method:** Decision Tree Regressor (max depth=3)  
**Performance:** Root Mean Square Error ≈ 1.0  

**Key Observations:**  
- Higher caffeine intake, low sleep hours, and certain exercise/alcohol patterns increase severity  
- Patients with PTSD or anxiety histories tend to have higher severity  

### Frequency Prediction

**Target:** Panic Attack Frequency (Scale 1–5; 1=rarely, 5=regularly)  
**Predictors:** PTSD history, Sleep Hours, Alcohol Intake, Exercise Frequency, Caffeine Intake  
**Method:** Decision Tree Regressor (max depth=3)  
**Performance:** Root Mean Square Error ≈ 1.4  

**Key Observations:**  
- PTSD history strongly associated with higher frequency  
- Sleep hours and caffeine intake show clear correlation patterns  
- Helps identify individuals at higher risk based on lifestyle and medical history  

**Visuals:**
- Regression tree diagrams for severity and frequency  

---

## Key Findings

**Behavioral Insights:**  
- History of PTSD leads to higher frequency and severity of panic attacks  
- Sweating and shortness of breath are the most reliable symptom indicators  
- Lifestyle factors (sleep, caffeine, exercise, alcohol) influence both frequency and severity  

**Practical Implications:**  
- Provides doctors actionable insights to advise patients  
- Helps identify high-risk individuals based on lifestyle and medical history  

**Future Investigations:**  
- Stratify trees by age and gender to observe demographic-specific patterns  
- Explore more granular sleep, exercise, and caffeine patterns  
- Develop intervention strategies to mitigate high-risk factors  

