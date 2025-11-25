import joblib

label_encoder = joblib.load("label_encoder.pkl")
print("Number of classes:", len(label_encoder.classes_))
print("Classes:")
for c in label_encoder.classes_:
    print("-", c)
