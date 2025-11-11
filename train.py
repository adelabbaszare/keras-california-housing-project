import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
import numpy as np

def build_model(input_shape):

    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1) 
    ])
    

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

def main():
    print("بارگذاری دیتاست مسکن کالیفرنیا...")
    # ۱. بارگذاری دیتاست
    (x_train, y_train), (x_test, y_test) = california_housing.load_data()
    
    print(f"تعداد نمونه‌های آموزشی: {x_train.shape[0]}")
    print(f"تعداد نمونه‌های تست: {x_test.shape[0]}")
    print(f"تعداد ویژگی‌ها: {x_train.shape[1]}")

    # ۲. پیش‌پردازش داده‌ها (استانداردسازی)
    # شبکه‌های عصبی به مقیاس ویژگی‌ها حساس هستند.
    # ما از StandardScaler برای نرمال‌سازی داده‌ها استفاده می‌کنیم.
    
    scaler = StandardScaler()
    
    # مهم: Scaler را فقط روی داده‌های آموزشی fit می‌کنیم
    x_train_scaled = scaler.fit_transform(x_train)
    
    # و از همان scaler برای تبدیل داده‌های تست استفاده می‌کنیم
    x_test_scaled = scaler.transform(x_test)
    
    print("استانداردسازی داده‌ها انجام شد.")

    # ۳. ساخت مدل
    model = build_model(input_shape=x_train.shape[1])
    model.summary()

    # ۴. آموزش مدل
    print("\nشروع آموزش مدل...")
    history = model.fit(
        x_train_scaled,
        y_train,
        epochs=100,       # تعداد دورهای آموزش (می‌توانید افزایش دهید)
        batch_size=32,
        validation_split=0.2, # استفاده از ۲۰٪ داده‌های آموزشی برای اعتبارسنجی
        verbose=1          # نمایش پیشرفت آموزش
    )
    print("آموزش مدل به پایان رسید.")

    # ۵. ارزیابی مدل
    print("\nارزیابی مدل بر روی داده‌های تست...")
    # مدل را با داده‌های تست ارزیابی می‌کنیم (که قبلا استاندارد شده‌اند)
    test_loss, test_mae = model.evaluate(x_test_scaled, y_test, verbose=0)
    
    print(f"\nنتایج ارزیابی نهایی:")
    print(f"  Loss (MSE) on Test Data: {test_loss:.4f}")
    print(f"  Mean Absolute Error (MAE) on Test Data: {test_mae:.4f}")
    
    # MAE به ما می‌گوید که به طور متوسط، تخمین‌های مدل چقدر با قیمت‌های واقعی فاصله دارد.
    # (توجه: قیمت‌ها در این دیتاست معمولا در مقیاس صدها هزار دلار هستند)

if __name__ == "__main__":
    main()