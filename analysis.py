import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fractions import Fraction
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import graphviz
from matplotlib.colors import LinearSegmentedColormap

# --- Load Data ---
df_extra = pd.read_csv('/content/Panic_Attack_Extra_Feature_Version.csv', encoding='unicode_escape')
df_ohe = pd.read_csv('/content/Panic_Attack_OHE_Enocde_Version.csv', encoding='unicode_escape')
df_label = pd.read_csv('/content/Panic_Attack_Label_Enocde_Version.csv', encoding='unicode_escape')

# --- Preprocess Extra Feature Data ---
df_extra.drop(['Gender', 'Trigger', 'Medical_History', 'Caffeine_Impact', 
               'Exercise_vs_Stress','Medication_Dependency', 
               'Gender_Binary', 'History_of_Mental_Illness'], axis=1, inplace=True)

# --- Preprocess One-Hot Encoded Data ---
df_ohe = df_ohe[['Gender_Female', 'Gender_Male', 'Gender_Non-binary',
                 'Trigger_Caffeine', 'Trigger_PTSD', 'Trigger_Phobia', 
                 'Trigger_Social_Anxiety', 'Trigger_Stress', 'Trigger_Unknown',
                 'Medical_History_Anxiety', 'Medical_History_Depression', 
                 'Medical_History_Missing', 'Medical_History_PTSD']]
df_ohe.rename(columns={'Trigger_Social Anxiety': 'Trigger_Social_Anxiety'}, inplace=True)

# --- Merge Extra & OHE Datasets ---
df = df_extra.join(df_ohe, how='left')

# --- Convert to Numeric ---
df['Symptom_Count'] = pd.to_numeric(df['Symptom_Count'], errors='coerce')
df['Sleep_Hours'] = pd.to_numeric(df['Sleep_Hours'], errors='coerce')

# --- Reconstruct Gender, Trigger, Medical History ---
df['Gender'] = df[['Gender_Female', 'Gender_Male', 'Gender_Non-binary']].idxmax(axis=1).map({
    'Gender_Female': 'Female', 'Gender_Male': 'Male', 'Gender_Non-binary': 'Non-binary'})
df['Trigger'] = df[['Trigger_Caffeine', 'Trigger_PTSD', 'Trigger_Phobia', 'Trigger_Social_Anxiety', 'Trigger_Stress', 'Trigger_Unknown']].idxmax(axis=1).map({
    'Trigger_Caffeine': 'Caffeine', 'Trigger_PTSD': 'PTSD', 'Trigger_Phobia': 'Phobia', 
    'Trigger_Social_Anxiety': 'Social Anxiety', 'Trigger_Stress': 'Stress', 'Trigger_Unknown': 'Unknown'})
df['Medical_History'] = df[['Medical_History_Anxiety', 'Medical_History_Depression', 'Medical_History_Missing', 'Medical_History_PTSD']].idxmax(axis=1).map({
    'Medical_History_Anxiety': 'Anxiety', 'Medical_History_Depression': 'Depression',
    'Medical_History_Missing': 'Missing', 'Medical_History_PTSD': 'PTSD'})
df['Symptoms'] = df[['Sweating', 'Shortness_of_Breath', 'Dizziness', 'Chest_Pain', 'Trembling']].idxmax(axis=1)

# --- Heart Rate & Severity ---
df['Heart_Rate_Category'] = df['Heart_Rate'].apply(lambda x: 'Low' if x < 100 else ('Normal' if x <= 120 else 'High'))
df['High_Risk_Individual'] = ((df['Sleep_Hours'] < 5) & (df['Caffeine_Intake'] > 3) & 
                               ((df['Medical_History_Anxiety'] == 1) | 
                                (df['Medical_History_Depression'] == 1) | 
                                (df['Medical_History_PTSD'] == 1))).astype(int)

# --- Chi-Squared Tests: Symptoms ---
df_symptoms = df[["Sweating", "Shortness_of_Breath", "Dizziness", "Chest_Pain", "Trembling"]].applymap(lambda x: 1 if x==1 else 0)
results = []
for s1, s2 in combinations(df_symptoms.columns, 2):
    table = pd.crosstab(df_symptoms[s1], df_symptoms[s2])
    if table.shape == (2,2):
        chi2, p, dof, expected = chi2_contingency(table)
        results.append({"Symptom 1": s1, "Symptom 2": s2, "Chi2": chi2, "p-value": p, "Degrees of Freedom": dof})
results_df = pd.DataFrame(results)
significant_results = results_df[results_df["p-value"] < 0.05]

# --- Chi-Squared Tests: Symptoms vs Triggers ---
df_triggers = df[["Trigger_Caffeine","Trigger_PTSD","Trigger_Phobia","Trigger_Social_Anxiety","Trigger_Stress","Trigger_Unknown"]].applymap(lambda x: 1 if x==1 else 0)
results = []
for symptom in df_symptoms.columns:
    for trigger in df_triggers.columns:
        table = pd.crosstab(df_symptoms[symptom], df_triggers[trigger])
        if table.shape == (2,2):
            chi2, p, dof, expected = chi2_contingency(table)
            results.append({"Symptom": symptom, "Trigger": trigger, "Chi2": chi2, "p-value": p, "Degrees of Freedom": dof})
results_df = pd.DataFrame(results)
significant_results_triggers = results_df[results_df["p-value"] < 0.05]

# --- Co-occurrence Heatmap: Triggers & Symptoms ---
trigger_types = ["Caffeine","PTSD","Phobia","Social Anxiety","Stress","Unknown"]
symptom_types = ["Sweating","Shortness of Breath","Dizziness","Chest Pain","Trembling"]

np.random.seed(10)
num_samples = 5000
triggers = np.random.choice(trigger_types, size=num_samples, p=prop_trigger)
prop_symptoms = np.array(prop_symptoms)/np.sum(prop_symptoms)
symptoms = np.random.choice(symptom_types, size=num_samples, p=prop_symptoms)
df_tvs = pd.DataFrame({'Trigger': triggers, 'Symptom': symptoms})

heatmap_data = pd.DataFrame(index=trigger_types, columns=symptom_types, dtype=float)
for trigger in trigger_types:
    for symptom in symptom_types:
        co_occurrence = len(df_tvs[(df_tvs['Trigger']==trigger) & (df_tvs['Symptom']==symptom)])/len(df_tvs)
        heatmap_data.loc[trigger, symptom] = co_occurrence

cmap = LinearSegmentedColormap.from_list("custom_gradient", ['#ffffff','#e69a8a'], N=100)
sns.heatmap(heatmap_data, annot=True, square=True, cmap=cmap, fmt=".2f")
plt.title('Co-Occurrence: Triggers & Symptoms')
plt.xlabel('Symptoms')
plt.ylabel('Triggers')
plt.show()

# --- Density Plots & Counts ---
sns.kdeplot(data=df, x='Panic_Attack_Severity', hue='Gender', fill=True, common_norm=False, palette=['#e69a8a','#6280ff','#fdcbbf'])
plt.title('Proportional Distribution: Panic Attack Severity by Gender')
plt.show()

sns.countplot(data=df, x='Sleep_Hours', hue='Sleep_Hours', legend=False, palette=['#fdcbbf'])
plt.title('Frequency: Hours of Sleep')
plt.show()

# --- Correlations ---
corr = df[['Sleep_Hours','Panic_Attack_Frequency']].corr().iloc[0,1]
print(f"Correlation Coefficient: {corr:.4f}")

# --- High-Severity Analysis ---
high_sev = df['Panic_Attack_Severity'] > df['Panic_Attack_Severity'].quantile(0.75)
history_cols = ['Medical_History_Anxiety','Medical_History_Depression','Medical_History_PTSD']
history_types = ["Anxiety","Depression","PTSD"]

# Proportions
for col in history_cols:
    prop = df[high_sev][col].sum()/df[col].sum()
    frac = Fraction(prop).limit_denominator()
    print(f"{col} Proportion: {prop:.4f}, Ratio: {frac.numerator}:{frac.denominator}")

# --- Plots: Triggers & Symptoms by Severity ---
colors = ['#fdcbbf','#6280ff','#c4bbe7']
plt.figure(figsize=(6,6))
plt.bar(history_types, [df[high_sev][col].sum()/df[col].sum() for col in history_cols], color=colors, edgecolor='black')
plt.title('High-Severity Proportions of Individual Medical Histories')
plt.xlabel('Medical History')
plt.ylabel('Proportion')
plt.show()

# --- Regression Tree: Severity ---
def categorize_risk(Severity):
    if Severity < 80: return 1
    elif Severity < 160: return 2
    elif Severity < 240: return 3
    elif Severity < 320: return 4
    elif Severity < 400: return 5
    else: return None

df['Sev_Risk_Level'] = df['Panic_Attack_Severity'].apply(categorize_risk)
X = df[['Caffeine_Intake','Exercise_Frequency','Sleep_Hours','Alcohol_Consumption','Smoking','Therapy',
        'Medical_History_Anxiety','Medical_History_Depression','Medical_History_PTSD']]
y = df['Sev_Risk_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_tree = DecisionTreeRegressor(max_depth=3)
reg_tree.fit(X_train, y_train)
y_pred = reg_tree.predict(X_test)
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test,y_pred)):.4f}")
data = export_graphviz(reg_tree, feature_names=X.columns, rounded=True, impurity=False)
graph = graphviz.Source(data)

# --- Regression Tree: Frequency ---
def categorize_risk(frequency):
    if frequency in [0,1]: return 1
    elif frequency in [2,3]: return 2
    elif frequency in [4,5]: return 3
    elif frequency in [6,7]: return 4
    elif frequency in [8,9]: return 5
    else: return None

df['Risk_Level'] = df['Panic_Attack_Frequency'].apply(categorize_risk)
X = df[['Caffeine_Intake','Exercise_Frequency','Sleep_Hours','Alcohol_Consumption','Smoking','Therapy',
        'Medical_History_Anxiety','Medical_History_Depression','Medical_History_PTSD']]
y = df['Risk_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
reg_tree = DecisionTreeRegressor(max_depth=3)
reg_tree.fit(X_train, y_train)
y_pred = reg_tree.predict(X_test)
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test,y_pred)):.4f}")
data = export_graphviz(reg_tree, feature_names=X.columns, rounded=True, impurity=False)
graph = graphviz.Source(data)
