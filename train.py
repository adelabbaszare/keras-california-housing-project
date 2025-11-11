import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- توابع کمکی برای مصورسازی ---

def plot_feature_distributions(df, save_path):
    """رسم هیستوگرام برای تمام ویژگی‌ها"""
    print("در حال رسم نمودار توزیع ویژگی‌ها...")
    plt.figure(figsize=(15, 10))
    df.hist(bins=30, figsize=(15, 10), layout=(-1, 4))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_correlation_heatmap(df, save_path):
    """رسم هیت‌مپ (نقشه حرارتی) همبستگی"""
    print("در حال رسم هیت‌مپ همبستگی...")
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(save_path)
    plt.close()

def plot_geo_map(df, save_path):
    """رسم نقشه جغرافیایی قیمت‌ها"""
    print("در حال رسم نقشه جغرافیایی قیمت‌ها...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="Longitude",
        y="Latitude",
        size="MedHouseVal",
        hue="MedHouseVal",
        palette="viridis",
        alpha=0.6,
        sizes=(1, 500)
    )
    plt.title("Geographical Distribution of House Prices")
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_path):
    """رسم نمودار تاریخچه آموزش (Loss و MAE)"""
    print("در حال رسم تاریخچه آموزش...")
    df_history = pd.DataFrame(history.history)
    plt.figure(figsize=(12, 5))
    
    # نمودار Loss (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(df_history['loss'], label='Training Loss')
    plt.plot(df_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # نمودار MAE
    plt.subplot(1, 2, 2)
    plt.plot(df_history['mae'], label='Training MAE')
    plt.plot(df_history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_predictions_vs_actual(y_true, y_pred, save_path):
    """رسم نمودار مقایسه مقادیر واقعی و پیش‌بینی شده"""
    print("در حال رسم نمودار پیش‌بینی در برابر واقعیت...")
    plt.figure(figsize=(8, 8))
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel("Actual Prices (MedHouseVal)")
    plt.ylabel("Predicted Prices (MedHouseVal)")
    plt.title("Actual vs. Predicted Prices")
    plt.legend()
    plt.axis('equal') # مقیاس یکسان برای هر دو محور
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- تابع ساخت مدل ---

def build_model(input_shape):
    """ساخت مدل شبکه عصبی برای رگرسیون"""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1) # لایه خروجی رگرسیون
    ])
    
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

# --- اسکریپت اصلی ---

def main():
    # ایجاد پوشه برای ذخیره نمودارها
    output_plot_dir = "plots"
    os.makedirs(output_plot_dir, exist_ok=True)
    
    # ۱. بارگذاری دیتاست
    print("بارگذاری دیتاست مسکن کالیفرنیا...")
    (x_train, y_train), (x_test, y_test) = california_housing.load_data()
    
    # تعریف نام ستون‌ها (بر اساس مستندات Keras)
    feature_names = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
        'AveOccup', 'Latitude', 'Longitude'
    ]
    target_name = 'MedHouseVal'

    # ۲. تحلیل اکتشافی داده‌ها (EDA)
    print("\n--- شروع تحلیل اکتشافی داده‌ها (EDA) ---")
    
    # تبدیل داده‌های آموزشی به DataFrame برای تحلیل و مصورسازی
    df_train = pd.DataFrame(x_train, columns=feature_names)
    df_train[target_name] = y_train
    
    # نمایش اطلاعات پایه دیتاست
    print("اطلاعات دیتاست (Info):")
    df_train.info()
    
    print("\nآمار توصیفی (Describe):")
    # .transpose() برای نمایش بهتر
    print(df_train.describe().transpose().to_string())

    # مصورسازی‌های EDA (ذخیره در پوشه plots)
    plot_feature_distributions(df_train[feature_names], os.path.join(output_plot_dir, "1_feature_distributions.png"))
    plot_correlation_heatmap(df_train, os.path.join(output_plot_dir, "2_correlation_heatmap.png"))
    plot_geo_map(df_train, os.path.join(output_plot_dir, "3_geo_price_map.png"))
    
    print(f"نمودارهای EDA در پوشه '{output_plot_dir}' ذخیره شدند.")

    # ۳. آماده‌سازی داده‌ها برای SandDance
    print("\n--- آماده‌سازی داده‌ها برای SandDance ---")
    sanddance_csv_path = "california_housing_for_sanddance.csv"
    df_train.to_csv(sanddance_csv_path, index=False)
    print(f"داده‌های آموزشی در فایل '{sanddance_csv_path}' ذخیره شدند.")
    print(f">>> لطفاً این فایل را در {os.path.abspath(sanddance_csv_path)}")
    print(">>> در وب‌سایت https://sanddance.js.org/app/ آپلود کنید تا مصورسازی تعاملی انجام دهید.")

    # ۴. پیش‌پردازش داده‌ها برای مدل
    print("\n--- پیش‌پردازش داده‌ها برای مدل ---")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print("استانداردسازی داده‌ها (StandardScaler) انجام شد.")

    # ۵. ساخت و آموزش مدل
    print("\n--- ساخت و آموزش مدل ---")
    model = build_model(input_shape=x_train.shape[1])
    model.summary()
    
    history = model.fit(
        x_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2, # استفاده از داده‌های اعتبارسنجی
        verbose=1
    )
    
    print("آموزش مدل به پایان رسید.")
    
    # ذخیره مدل
    model_save_path = "california_housing_model.keras"
    model.save(model_save_path)
    print(f"مدل آموزش‌دیده در '{model_save_path}' ذخیره شد.")

    # ۶. ارزیابی و مصورسازی نتایج مدل
    print("\n--- ارزیابی نتایج مدل ---")
    
    # ارزیابی نهایی روی داده‌های تست
    test_loss, test_mae = model.evaluate(x_test_scaled, y_test, verbose=0)
    print(f"\nنتایج ارزیابی نهایی روی داده‌های تست:")
    print(f"  Loss (MSE): {test_loss:.4f}")
    print(f"  Mean Absolute Error (MAE): {test_mae:.4f}")

    # مصورسازی نتایج آموزش
    plot_training_history(history, os.path.join(output_plot_dir, "4_training_history.png"))
    
    # پیش‌بینی روی داده‌های تست
    y_pred = model.predict(x_test_scaled).flatten()
    
    # مصورسازی پیش‌بینی در برابر واقعیت
    plot_predictions_vs_actual(y_test, y_pred, os.path.join(output_plot_dir, "5_predictions_vs_actual.png"))
    
    print(f"نمودارهای ارزیابی مدل در پوشه '{output_plot_dir}' ذخیره شدند.")
    print("\n--- پروژه با موفقیت انجام شد ---")

if __name__ == "__main__":
    main()