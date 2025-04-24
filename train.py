
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# تحميل البيانات
df = pd.read_csv("Final_Model_Datav2.csv")

# الميزات والهدف
features = [
    "Base_Price_Base", "Importance_Num", "Stage_Num", "Venue_Num",
    "Team1_Num", "Team2_Num", "Days_until_match", "Tickets_Sold", "Year"
]
X = df[features]
y = df["Final_Price"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تدريب المودل
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# تقييم الأداء
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

# حفظ النموذج
joblib.dump(model, "new_model.pkl")

# طباعة النتائج
print(f"R² التدريب: {train_r2:.4f}")
print(f"R² الاختبار: {test_r2:.4f}")
print(f"RMSE التدريب: {train_rmse:.2f}")
print(f"RMSE الاختبار: {test_rmse:.2f}")

# تجربة النموذج
model = joblib.load("new_model.pkl")

# بيانات تجربة
input_data = {
    "Base_Price_Base": 500,
    "Importance_Num": 2,
    "Stage_Num": 3,
    "Venue_Num": 5,
    "Team1_Num": 7,
    "Team2_Num": 12,
    "Days_until_match": 4,
    "Tickets_Sold": 75,
    "Year": 2034
}

df_input = pd.DataFrame([input_data])
predicted_price = model.predict(df_input)[0]
print(f"🎯 السعر المتوقع للتذكرة هو: {round(predicted_price, 2)} ريال")
