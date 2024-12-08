import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fastapi import FastAPI
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Greetings and basic information','Graphs','Add your new data',"Conclusion.Thanks for using!"])

if page == 'Greetings and basic information':
    st.title('Students Depression Data Analysis')
    st.subheader('Hello! I am the creator of this site. Miroshnichenko Arina Valeryevna, DSBA, group - 246')
    st.header('Welcome to the Students Depression Analysis App!')
    st.write("Welcome to the Students' Depression Analysis Platform! Here, you can explore detailed data visualizations and insights on student mental health, with easy navigation to other pages for a deeper understanding.")

    @st.cache_data
    def load_data():
        df = pd.read_csv(r"C:\Users\arina\OneDrive\Документы\Visual Studio 2022\Student Depression Dataset.csv")
        return df
    df = load_data() 

    st.title("Overview of Depression Data")

    st.write("The first 5 rows of data:")
    st.dataframe(df.head())

# Добавим возможность пользователю выбрать количество строк для отображения
    num_rows = st.slider('The number of lines to display', min_value=5, max_value=len(df), value=5)
    st.dataframe(df.head(num_rows)) 

if page == 'Graphs':

    @st.cache_data
    def load_data():
        df = pd.read_csv(r"C:\Users\arina\OneDrive\Документы\Visual Studio 2022\Student Depression Dataset.csv")
        return df
    df = load_data()
    st.title('Basic Graphs based on the data')
    st.subheader("Gender Distribution Analysis")
    st.write('This graph shows the ratio of the number of men and women surveyed, where pink indicates the number of women (as a percentage), and blue indicates the number of men (also as a percentage)')
    gender_counts = df['Gender'].value_counts()
    fig = px.pie(
    values=gender_counts.values,
    names=gender_counts.index,
    color=gender_counts.index,
    color_discrete_map={'Male': 'skyblue', 'Female': 'pink'}
)

    fig.update_layout(title='Gender Distribution')

# Отображение диаграммы в Streamlit
    st.plotly_chart(fig)

    st.subheader("Interactive Correlation Matrix")
    st.write('In this graph, we can trace the dependence of our data. To do this, I used the correlation graph. The correlation graph shows how one variable changes depending on another. This allows you to understand whether there is a linear relationship between the variables.')
    @st.cache_data
    def load_data():
        df = pd.read_csv(r"C:\Users\arina\OneDrive\Документы\Visual Studio 2022\streamlit\updated_dataset.csv")
        return df
    df = load_data()
    temp_df = df.copy()
    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].replace({'Yes': 1, 'No': 0})
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].replace({'Yes': 1, 'No': 0})
    columns_to_include = ['Age', 'CGPA', 'Have you ever had suicidal thoughts ?',
                          'Work/Study Hours', 'Financial Stress',
                          'Family History of Mental Illness', 'Depression',
                          'Overall Pressure', 'Overall Satisfaction']
    
    # Проверка наличия колонок в данных
    temp_df = df[columns_to_include]
    
        
        # Вычисление корреляционной матрицы
    correlation_matrix = temp_df.corr(method='pearson')
        
        # Создание интерактивного графика
    fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            labels={'color': "Pearson's Correlation"},
            x=correlation_matrix.columns,
            y=correlation_matrix.index
        )
        
    fig.update_layout(
    paper_bgcolor='white',
    plot_bgcolor='lightgray',
    showlegend=False,
    margin=dict(l=100, r=8, t=50, b=80),  # Настройка отступов: уменьшите 'r', чтобы подвинуть график влево
    width=900,  # Увеличение ширины холста
    height=600,  # Увеличение высоты холста
    xaxis=dict(domain=[1.0, 1.0]),  # Использовать всю ширину для оси X
    yaxis=dict(domain=[1.0, 1.0])  # Сужение или сдвиг графика по горизонтальной оси
)
        
        # Отображение графика
    st.plotly_chart(fig)
    
    st.header('The relationship of people suffering from depression and not')
    st.write('This graph shows the percentage of people who suffer from depression and those who do not')
    color_map = {0: 'brown', 1: 'pink'}

# Построение диаграммы пирога

    fig = px.pie(
    values=df['Depression'].value_counts(),
    names=df['Depression'].value_counts().index.map({0: 'No Depression', 1: 'Have Depression'}),
    color=df['Depression'].value_counts().index,
    color_discrete_map= color_map)

# Настройка внешнего вида графика
    fig.update_layout(title='Depression %')

# Отображение графика в Streamlit
    st.plotly_chart(fig)

    selected_columns = ['Age', 'Have you ever had suicidal thoughts ?',
                    'Work/Study Hours', 'Financial Stress',]

# Фильтрация данных
    df_temp = df[selected_columns]

# Отображение всех гистограмм
    st.title('Histograms for selected columns')
    st.write('The histogram data shows the number of people belonging to a particular category')

    for column in selected_columns:
       st.subheader(f'Histograms for: {column}')
       fig = px.histogram(df_temp, x=column, nbins=10, color_discrete_sequence=['purple'])
       st.plotly_chart(fig)

    st.header('Histograms for non-numeric values')
    st.write('These columns had data other than numeric, and it became interesting for me to count people in each category')
    temp_df = df.copy()

# Указание колонок для анализа
    columns = ['Sleep Duration', 'Dietary Habits', 'Degree']

# Создание подграфиков
    fig = make_subplots(rows=1, cols=3, subplot_titles=columns, horizontal_spacing=0.1)

# Цвета для графиков
    colors = ['purple', 'lightgreen', 'cyan']

# Построение графиков для каждой колонки
    for i, column in enumerate(columns):
        if column in temp_df.columns:
        # Подсчет уникальных значений
            data = temp_df[column].value_counts().sort_values(ascending=False)

        # Добавление данных на график
            fig.add_trace(
            go.Bar(
                x=data.index,  # Уникальные значения
                y=data.values,  # Частоты
                marker=dict(color=colors[i]),  # Цвет столбцов
                name=column
            ),
            row=1, col=i+1  # Расположение в сетке
        )

# Настройка макета
        fig.update_layout(
    height=500,
    width=1200,
    title_text="Distribution of Sleep Duration, Dietary Habits, and Degree",
    showlegend=False,
    paper_bgcolor='white',
    plot_bgcolor='white',
)

# Интеграция с Streamlit
    st.plotly_chart(fig, use_container_width=True)

    data = df.groupby(['Gender', 'Depression']).size().reset_index(name='Count')

# Создание списка значений для оси X и Y
    x_labels = ['Male - Have depression', 'Male - No depression',
            'Female - Have depression', 'Female - No depression']
    counts = [
    data[(data['Gender'] == 'Male') & (data['Depression'] == 1)]['Count'].values[0] if not data[(data['Gender'] == 'Male') & (data['Depression'] == 1)].empty else 0,
    data[(data['Gender'] == 'Male') & (data['Depression'] == 0)]['Count'].values[0] if not data[(data['Gender'] == 'Male') & (data['Depression'] == 0)].empty else 0,
    data[(data['Gender'] == 'Female') & (data['Depression'] == 1)]['Count'].values[0] if not data[(data['Gender'] == 'Female') & (data['Depression'] == 1)].empty else 0,
    data[(data['Gender'] == 'Female') & (data['Depression'] == 0)]['Count'].values[0] if not data[(data['Gender'] == 'Female') & (data['Depression'] == 0)].empty else 0
]

# Создание графика
    fig = go.Figure()

    fig.add_trace(go.Bar(
    x=x_labels,
    y=counts,
    marker_color=['lightblue', 'lightblue', 'pink', 'pink'],  # Цвет для мужчин и женщин
    text=counts,
    textposition='auto',
    name='Depression Status'
))

# Настройка оформления
    fig.update_layout(
    title="Comparison of Gender and Depression",
    xaxis=dict(
        title="Gender and Depression Status",
        tickvals=x_labels
    ),
    yaxis_title="Count",
    paper_bgcolor='white',
    plot_bgcolor='white',
    width=900,
    height=600
)

# Визуализация в Streamlit
    st.title("Gender and Depression Analysis")
    st.write('In my study, the number of men suffering from depression is higher than the number of women, but this is based on this graph. According to the graph, which is at the very beginning, it is clear that there were more men interviewed, so next I will investigate the percentage of people with depression from the total number (men and women separately)')
    st.plotly_chart(fig, use_container_width=True)

    male_total = len(df[df['Gender'] == 'Male'])  # Общее количество мужчин
    male_with_depression = len(df[(df['Gender'] == 'Male') & (df['Depression'] == 1)])  # Мужчины с депрессией

    female_total = len(df[df['Gender'] == 'Female'])  # Общее количество женщин
    female_with_depression = len(df[(df['Gender'] == 'Female') & (df['Depression'] == 1)])  # Женщины с депрессией

# Первый график - Male
    male_percentage = (male_with_depression / male_total) * 100 if male_total > 0 else 0

    fig_male = go.Figure(go.Pie(
    labels=['Have Depression', 'No Depression'],
    values=[male_with_depression, male_total - male_with_depression],
    marker=dict(colors=['lightblue', 'lightyellow']),
    textinfo='label+percent',
    pull=[0.1, 0]  # Выделяем "Have Depression"
))

    fig_male.update_layout(
    title="Male Depression Percentage",
    paper_bgcolor='white'
)

# Второй график - Female
    female_percentage = (female_with_depression / female_total) * 100 if female_total > 0 else 0

    fig_female = go.Figure(go.Pie(
    labels=['Have Depression', 'No Depression'],
    values=[female_with_depression, female_total - female_with_depression],
    marker=dict(colors=['pink', 'lightyellow']),
    textinfo='label+percent',
    pull=[0.1, 0]  # Выделяем "Have Depression"
))

    fig_female.update_layout(
    title="Female Depression Percentage",
    paper_bgcolor='white'
)

# Отображение в Streamlit
    st.title("Gender-Based Depression Analysis")

    st.subheader("Male Depression Percentage")
    st.plotly_chart(fig_male, use_container_width=True)

    st.subheader("Female Depression Percentage")
    st.plotly_chart(fig_female, use_container_width=True)

    city_depression = df[df['Depression'] == 1].groupby('City')['Depression'].count().reset_index()
    city_depression.columns = ['City', 'Count of Depression']

# Проверка пропущенных городов
    all_cities = df['City'].unique()
    missing_cities = set(all_cities) - set(city_depression['City'])

# Сортировка по количеству депрессий
    city_depression = city_depression.sort_values(by='Count of Depression', ascending=False)

# Создание графика
    fig = px.bar(
    city_depression,
    x='Count of Depression',
    y='City',
    title='Count of People with Depression by City (Horizontal)',
    text='Count of Depression',
    color='Count of Depression',
    color_continuous_scale='Reds',
    orientation='h'
)

# Настройка графика
    fig.update_layout(
    xaxis_title='Count of People with Depression',
    yaxis_title='City',
    paper_bgcolor='white',
    plot_bgcolor='whitesmoke',
    title_font=dict(size=18),
    height=800,  # Высота графика
)

# Streamlit
    st.title("Depression Analysis by City")
    st.plotly_chart(fig, use_container_width=True)

    df['Category'] = df.apply(
    lambda row:
    'Family History - no depression, No Depression' if row['Family History of Mental Illness'] == 0 and row['Depression'] == 0 else
    'Family History - had depression, No Depression' if row['Family History of Mental Illness'] == 1 and row['Depression'] == 0 else
    'Family History - no depression, Has Depression' if row['Family History of Mental Illness'] == 0 and row['Depression'] == 1 else
    'Family History - had depression, Has Depression',
    axis=1
)

# Подсчет значений по категориям
    category_counts = df['Category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']

# Упорядочение категорий
    category_order = [
    'Family History - no depression, No Depression',
    'Family History - had depression, No Depression',
    'Family History - no depression, Has Depression',
    'Family History - had depression, Has Depression'
]
    category_counts = category_counts.set_index('Category').reindex(category_order).reset_index()

# Построение линейного графика
    fig = go.Figure()

    fig.add_trace(go.Scatter(
    x=category_counts['Category'],
    y=category_counts['Count'],
    mode='lines+markers',
    line=dict(color='rebeccapurple', width=3),  # Линия сливового цвета
    marker=dict(size=10, color='orangered'),  # Точки красно-оранжевого цвета
    text=category_counts['Count'],  # Подписи точек
    name='Count'
))

# Настройка графика
    fig.update_layout(
    title='Mental Illness Family History vs Depression Status (Line Chart)',
    xaxis_title='Category',
    yaxis_title='Count',
    xaxis=dict(tickangle=45),
    paper_bgcolor='white',
    plot_bgcolor='whitesmoke',
    height=600,
    width=1000
)

# Отображение графика в приложении Streamlit
    st.header('A graph for investigating the question : Is depression a hereditary thing?')
    st.plotly_chart(fig)

    fig = px.histogram(
    df,
    x="Financial Stress",
    color="Depression",
    barmode="group",  # Группируем по цвету (Depression)
    color_discrete_sequence=["brown", "pink"],  # Цвета: коричневый - No Depression, розовый - Have Depression
    text_auto=True  # Отображаем значения на столбцах
)

# Настройка графика
    fig.update_layout(
    title="Comparison of Financial Stress and Depression",
    xaxis_title="Financial Stress",
    yaxis_title="Count",
    legend_title="Depression Status",
    legend=dict(
        itemsizing="constant",
        title_font=dict(size=14),
        font=dict(size=12),
        orientation="h",  # Легенда горизонтально
        yanchor="top",
        y=-0.2  # Позиция легенды
    ),
    paper_bgcolor="white",
    plot_bgcolor="white"
)

# Добавление пояснений к легенде
    fig.for_each_trace(
    lambda trace: trace.update(name="No Depression" if trace.name == "0" else "Have Depression")
)

# Отображение графика в приложении Streamlit
    st.header('Comparison of Financial Stress and Depression')
    st.plotly_chart(fig)

if page == 'Add your new data':
    columns = [
    'Age', 'City', 'AcademicPressure', 'WorkPressure', 'CGPA', 'SleepDuration', 
    'DietaryHabits', 'Degree', 'HaveYouEverHadSuicidalThoughts', 'WorkStudyHours',
    'FinancialStress', 'FamilyhistoryOfMentalIllness', 'Depression', 'OverallPressure', 'OverallSatisfaction'
]
    data = pd.DataFrame(columns=columns)
    # Заголовок страницы
    st.title("Add New Record to Depression Dataset")

# Создание формы для ввода данных
    with st.form(key='data_form'):
       age = st.number_input('Age', min_value=0.0, format="%.2f")
       city = st.text_input('City')
       academic_pressure = st.number_input('Academic Pressure', min_value=0, step=1)
       work_pressure = st.number_input('Work Pressure', min_value=0, step=1)
       cgpa = st.number_input('CGPA', min_value=0.0, format="%.2f")
    
       sleep_duration = st.selectbox('Sleep Duration', [
        'More than 8 hours', '7-8 hours', '5-6 hours', 'Less than 5 hours'
    ])
       
       dietary_habits = st.selectbox('Dietary Habits', [
        'Unhealthy', 'Moderate', 'Healthy'
    ])
       degree = st.text_input('Degree')
       suicidal_thoughts = st.selectbox('Have You Ever Had Suicidal Thoughts?', ['Yes', 'No'])
       work_study_hours = st.number_input('Work/Study Hours', min_value=0.0, format="%.2f")
       financial_stress = st.number_input('Financial Stress', min_value=0.0, format="%.2f")
       family_history = st.selectbox('Family History of Mental Illness?', ['Yes', 'No'])
       depression = st.number_input('Depression (1 for Yes, 0 for No)', min_value=0, max_value=1, step=1)
       overall_pressure = st.number_input('Overall Pressure', min_value=0.0, format="%.2f")
       overall_satisfaction = st.number_input('Overall Satisfaction', min_value=0.0, format="%.2f")
    
       submit_button = st.form_submit_button(label='Add Record')

       if submit_button:
        # Добавление нового экземпляра данных в DataFrame
           new_data = pd.DataFrame([{
            'Age': age, 'City': city, 'AcademicPressure': academic_pressure,
            'WorkPressure': work_pressure, 'CGPA': cgpa, 'SleepDuration': sleep_duration,
            'DietaryHabits': dietary_habits, 'Degree': degree,
            'HaveYouEverHadSuicidalThoughts': suicidal_thoughts,
            'WorkStudyHours': work_study_hours, 'FinancialStress': financial_stress,
            'FamilyhistoryOfMentalIllness': family_history, 'Depression': depression,
            'OverallPressure': overall_pressure, 'OverallSatisfaction': overall_satisfaction
        }])
        
        # Объединение с основным DataFrame (для демонстрации)
           data = pd.concat([data, new_data], ignore_index=True)

           st.success('Record added successfully!')
           st.dataframe(data)


if page == 'Conclusion.Thanks for using!':
    st.title('Conclusion')
    st.write('Thank you for visiting our website dedicated to the Depression dataset! We conducted an in-depth and fascinating analysis to shed light on important aspects of this topic. We hope you found the information valuable and inspiring. Your interest and support motivate us to continue exploring and sharing meaningful data. We appreciate your time and attention!')