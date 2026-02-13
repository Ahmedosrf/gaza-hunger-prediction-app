# Gaza Hunger Prediction System

## English

### Project Overview
This project aims to predict household hunger severity in the Gaza Strip using machine learning models. The system leverages a dataset collected from 1,209 households between May and July 2024 to analyze various socio-economic and environmental factors that contribute to food insecurity. The primary goal is to provide a predictive tool that can assist in identifying households at risk.

### Features
- **Data Analysis:** Comprehensive exploratory data analysis (EDA) of household data from Gaza.
- **Machine Learning Models:** Implementation and evaluation of Logistic Regression, Random Forest, and Gradient Boosting models for multi-class classification.
- **Interactive Web Application:** A Streamlit-based application for real-time predictions and data visualization.
- **Key Indicators:** Prediction based on various factors including household demographics, displacement status, health conditions, income, education, housing, and water availability.

### Dataset
The dataset `GazaHungerData.xlsx` contains information from 1,209 households in the Gaza Strip, collected from May to July 2024. It includes 50 features covering a wide range of indicators such as food access questions (Q1-Q7), frequency questions (Q8-Q10), household food security indicators (Q11-Q28), and various demographic and living condition questions (Q29-Q49). The target variable for prediction is `Q50: Water availability`.

### Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn (for machine learning models)
- Streamlit (for web application)
- Matplotlib, Seaborn, Plotly (for data visualization)

### How to Run the Application
To run the Streamlit application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ahmedosrf/gaza-hunger-prediction-app.git
    cd gaza-hunger-prediction-app
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not currently in the repository, but typically includes `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`.)*
3.  **Run the Streamlit app:**
    ```bash
    streamlit run gaza_hunger_app.py
    ```
    The application will open in your web browser.

### Project Structure
```
.gitignore
README.md
GazaHungerData.xlsx
gaza_hunger_app.py
gaza_hunger_prediction.ipynb
كود تشغيلstreamlit.txt
```

## العربية

### نظرة عامة على المشروع
يهدف هذا المشروع إلى التنبؤ بشدة الجوع لدى الأسر في قطاع غزة باستخدام نماذج التعلم الآلي. يستفيد النظام من مجموعة بيانات تم جمعها من 1,209 أسرة بين مايو ويوليو 2024 لتحليل العوامل الاجتماعية والاقتصادية والبيئية المختلفة التي تساهم في انعدام الأمن الغذائي. الهدف الأساسي هو توفير أداة تنبؤية يمكن أن تساعد في تحديد الأسر المعرضة للخطر.

### الميزات
-   **تحليل البيانات:** تحليل استكشافي شامل للبيانات الأسرية من غزة.
-   **نماذج التعلم الآلي:** تطبيق وتقييم نماذج الانحدار اللوجستي (Logistic Regression)، الغابات العشوائية (Random Forest)، وتعزيز التدرج (Gradient Boosting) للتصنيف متعدد الفئات.
-   **تطبيق ويب تفاعلي:** تطبيق مبني على Streamlit للتنبؤات في الوقت الفعلي وتصور البيانات.
-   **المؤشرات الرئيسية:** التنبؤ بناءً على عوامل مختلفة بما في ذلك التركيبة السكانية للأسر، حالة النزوح، الظروف الصحية، الدخل، التعليم، السكن، وتوفر المياه.

### مجموعة البيانات
تحتوي مجموعة البيانات `GazaHungerData.xlsx` على معلومات من 1,209 أسرة في قطاع غزة، تم جمعها من مايو إلى يوليو 2024. تتضمن 50 ميزة تغطي مجموعة واسعة من المؤشرات مثل أسئلة الوصول إلى الغذاء (Q1-Q7)، أسئلة التكرار (Q8-Q10)، مؤشرات الأمن الغذائي للأسر (Q11-Q28)، ومختلف الأسئلة الديموغرافية وظروف المعيشة (Q29-Q49). المتغير المستهدف للتنبؤ هو `Q50: توفر المياه`.

### التقنيات المستخدمة
-   بايثون (Python)
-   بانداس (Pandas)، نامباي (NumPy)
-   سايكت ليرن (Scikit-learn) (لنماذج التعلم الآلي)
-   ستريم ليت (Streamlit) (لتطبيق الويب)
-   مات بلوت ليب (Matplotlib)، سيبورن (Seaborn)، بلوتلي (Plotly) (لتصور البيانات)

### كيفية تشغيل التطبيق
لتشغيل تطبيق Streamlit محليًا، اتبع الخطوات التالية:

1.  **استنساخ المستودع (Clone the repository):**
    ```bash
    git clone https://github.com/Ahmedosrf/gaza-hunger-prediction-app.git
    cd gaza-hunger-prediction-app
    ```
2.  **تثبيت التبعيات (Install dependencies):**
    يوصى باستخدام بيئة افتراضية.
    ```bash
    pip install -r requirements.txt
    ```
    *(ملاحظة: ملف `requirements.txt` غير موجود حاليًا في المستودع، ولكنه عادةً ما يتضمن `streamlit`، `pandas`، `numpy`، `scikit-learn`، `matplotlib`، `seaborn`، `plotly`.)*
3.  **تشغيل تطبيق Streamlit:**
    ```bash
    streamlit run gaza_hunger_app.py
    ```
    سيتم فتح التطبيق في متصفح الويب الخاص بك.

### هيكل المشروع
```
.gitignore
README.md
GazaHungerData.xlsx
gaza_hunger_app.py
gaza_hunger_prediction.ipynb
كود تشغيلstreamlit.txt
```
