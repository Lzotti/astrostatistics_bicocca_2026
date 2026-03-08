import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix






filename = '/home/leo/astroML_data/sample_2e7_design_precessing_higherordermodes_3detectors.h5'
f = h5py.File(filename, 'r')

plt.figure(figsize=(10, 8))

all_features = []
all_labels = np.array(f['det'][:]) 

for key in f.keys():
    if key == 'det' or key == 'snr':  
        continue

    print(f"Processing feature: {key}")
    X = f[key][:].reshape(-1, 1)  
    all_features.append(X)  

    idx = np.random.choice(X.shape[0], size=100, replace=False)
    X = X[idx]
    labels = all_labels[idx]
    # labels = all_labels  

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Calcola le curve ROC e AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

        # Calcola altre metriche
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Stampa le metriche
    print(f"Metrics for {key}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  Confusion Matrix:\n{conf_matrix}")


    plt.plot(fpr, tpr, label=f'{key} (AUC = {auc:.2f})')

print("Processing all features together")
all_features_combined = np.hstack(all_features)  # Combina tutte le feature in un unico array
idx = np.random.choice(all_features_combined.shape[0], size=100, replace=False)
X = all_features_combined[idx]
labels = all_labels[idx]
# labels = all_labels  

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

# Calcola le curve ROC e AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

    # Calcola altre metriche
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
    # Stampa le metriche
print(f"Metrics for {key}:")
print(f"  Accuracy: {accuracy:.2f}")
print(f"  Precision: {precision:.2f}")
print(f"  Recall: {recall:.2f}")
print(f"  F1-Score: {f1:.2f}")
print(f"  Confusion Matrix:\n{conf_matrix}")


plt.plot(fpr, tpr, label=f'All Features (AUC = {auc:.2f})', linestyle='--', color='black')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Features')
plt.legend()
plt.grid()
plt.show()



####decision tree

plt.figure(figsize=(10, 8))

all_features = []
all_labels = np.array(f['det'][:])  

for key in f.keys():
    if key == 'det' or key == 'snr':  
        continue

    print(f"Processing feature: {key}")
    X = f[key][:].reshape(-1, 1)  
    all_features.append(X)  

    idx = np.random.choice(X.shape[0], size=1000, replace=False)
    X = X[idx]
    labels = all_labels[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # Calcola le curve ROC e AUC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    # Calcola altre metriche
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Stampa le metriche
    print(f"Metrics for {key}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  Confusion Matrix:\n{conf_matrix}")

    plt.plot(fpr, tpr, label=f'{key} (AUC = {auc:.2f})')

print("Processing all features together")
all_features_combined = np.hstack(all_features)  # Combina tutte le feature in un unico array
idx = np.random.choice(all_features_combined.shape[0], size=1000, replace=False)
X = all_features_combined[idx]
labels = all_labels[idx]

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Calcola le curve ROC e AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Calcola altre metriche
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Stampa le metriche per tutte le feature combinate
print("Metrics for all features combined:")
print(f"  Accuracy: {accuracy:.2f}")
print(f"  Precision: {precision:.2f}")
print(f"  Recall: {recall:.2f}")
print(f"  F1-Score: {f1:.2f}")
print(f"  Confusion Matrix:\n{conf_matrix}")

plt.plot(fpr, tpr, label=f'All Features (AUC = {auc:.2f})', linestyle='--', color='black')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Features (Decision Tree)')
plt.legend()
plt.grid()
plt.show()




