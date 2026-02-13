import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. تحميل البيانات
df = pd.read_csv('Chocolate_Sales.csv')

# 2. استكشاف البيانات الأولي (EDA)
print("معلومات البيانات الأساسية:")
print(df.info())
print("\nأول 5 صفوف من البيانات:")
display(df.head())

# 3. تنظيف البيانات (Data Cleaning)
# تحويل عمود 'Amount' إلى أرقام (حذف علامة $ والفواصل)
df['Amount'] = df['Amount'].str.replace('$', '').str.replace(',', '').astype(float)

# تحويل عمود 'Date' إلى صيغة تاريخ
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# التحقق من القيم المفقودة
print("\nالقيم المفقودة في كل عمود:")
print(df.isnull().sum())

# 4. تحليل البيانات
# حساب إجمالي المبيعات لكل منتج
product_sales = df.groupby('Product')['Amount'].sum().sort_values(ascending=False)

# 5. التصور البياني (Visualization) باستخدام Seaborn و Matplotlib
plt.figure(figsize=(12, 6))
sns.barplot(x=product_sales.values, y=product_sales.index, palette='viridis')

plt.title('Total Revenue by Product', fontsize=15)
plt.xlabel('Total Sales ($)', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()

# 6. تحليل إضافي: اتجاه المبيعات عبر الزمن
df['Month_Year'] = df['Date'].dt.to_period('M')
monthly_trend = df.groupby('Month_Year')['Amount'].sum()

plt.figure(figsize=(12, 5))
monthly_trend.plot(kind='line', marker='o', color='orange')
plt.title('Monthly Sales Trend', fontsize=15)
plt.ylabel('Sales ($)')
plt.xlabel('Month')
plt.grid(True)
plt.show()