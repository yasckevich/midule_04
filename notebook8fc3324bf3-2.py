#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, log_loss
from sklearn.metrics import accuracy_score
from pandas import Series
import pandas as pd
import numpy as np
import collections

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# client_id идентификатор клиента
# 
# education уровень образования категориальная переменная
# 
# sex пол заёмщика бинарная переменная
# 
# age возраст заёмщика числовая переменная
# 
# car флаг наличия автомобиля бинарная переменная
# 
# car_type флаг автомобиля-иномарки бинарная переменная
# 
# decline_app_cnt количество отказанных прошлых заявок числовая переменная
# 
# good_work флаг наличия «хорошей» работы бинарная переменная
# 
# bki_request_cnt количество запросов в БКИ числовая переменная
# 
# home_address категоризатор домашнего адреса категориальная переменная
# 
# work_address категоризатор рабочего адреса категориальная переменная
# 
# income доход заёмщика числовая переменная
# 
# foreign_passport наличие загранпаспорта бинарная переменная
# 
# default наличие дефолта

# In[ ]:


PATH_to_file = '/kaggle/input/sf-dst-scoring/'


# In[ ]:


# Чтобы различать обущающую и тестовые выборки, создадим столбец со значениями 1 и 0

df_train = pd.read_csv(PATH_to_file+'train.csv')
print('Размерность тренировочного датасета: ', df_train.shape)

df_test = pd.read_csv(PATH_to_file+'test.csv')
print('Размерность тестового датасета: ', df_test.shape)

sample_submission = pd.read_csv(PATH_to_file+'sample_submission.csv')


# In[ ]:


df_train['sample'] = 1   # помечаем где у нас трейн
df_test['sample'] = 0    # помечаем где у нас тест
df_test['default'] = -1  # в тесте у нас нет значения default, мы его должны предсказать, но его значения 0 или 1, поэтому заполняем его временно -1 для избежания ошибки

data = df_test.append(df_train, sort=False).reset_index(drop=True)   # объединяем

data


# In[ ]:


# разделяем признаки по группам

bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
cat_cols = ['education', 'home_address', 'work_address']
num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income','score_bki','region_rating','first_time','sna']
time_cols = ['app_date']

target = 'default'

id_data = data.client_id
data.drop(['client_id'], axis=1, inplace=True)


# In[ ]:


# проверяем пропуски
data.isnull().sum()


# In[ ]:


# Вариант 1 (удалить пропуски)
# train=train.dropna()
# но с таким вариантом результаты хуже

# поэтому
# Вариант 2 (заполнить наиболее частовстречаемым значением)
data.education = data.education.fillna(data.education.mode()[0])


# In[ ]:


data.default.value_counts()


# In[ ]:


# посмотрим на данные через boxplot
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.boxplot(x="default", y="age", data=data, palette='rainbow')


# In[ ]:


sns.boxplot(x="default", y="decline_app_cnt", data=data, palette='rainbow')


# In[ ]:


sns.boxplot(x="default", y="bki_request_cnt", data=data, palette='rainbow')


# In[ ]:


# меняем формат даты
data['app_date'] = pd.to_datetime(data.app_date)
data['app_date'].sample(3)

data['app_date'] = pd.to_datetime(data['app_date'], format='%d%b%Y')
data.head(3)


# In[ ]:


# отмеряем кол-во дней от первого в базе

start = data.app_date.min()
end = data.app_date.max()

data['app_date_delta'] = (data.app_date - start).dt.days.astype('int')

data


# In[ ]:


num_cols.append('app_date_delta')

# удаление временного ряда из датасета
data.drop(['app_date'], axis=1, inplace=True)
data


# In[ ]:


# смотрим распределение кол-ных признаков
fig, axes = plt.subplots(1, 3, figsize=(40,7))
for i,col in enumerate(['decline_app_cnt', 'bki_request_cnt', 'income']):
    data[col] = np.log(data[col] + 1)
    sns.distplot(data[col][data[col] > 0].dropna(), ax=axes.flat[i],kde = False, rug=False,color="g")


# In[ ]:


# логарифмируем эти признаки
data['age_log'] = np.log(data['age'] + 1)


# In[ ]:


# Для бинарных признаков мы будем использовать LabelEncoder

label_encoder = LabelEncoder()

for column in bin_cols:
    data[column] = label_encoder.fit_transform(data[column])
    
# убедимся в преобразовании    
data.head()


# In[ ]:


# преобразуем также категориальные признаки

data = pd.get_dummies(data, columns=['education'])
data = pd.get_dummies(data, columns=['home_address'])
data = pd.get_dummies(data, columns=['work_address'])
data


# In[ ]:


data[num_cols] = StandardScaler().fit_transform(data[num_cols].values)


# In[ ]:


# значимость признаков

a= data.columns.format() 

imp_cat = Series(mutual_info_classif(data, data['default'],
                                     discrete_features =True), index = a)
imp_cat.sort_values(inplace = True)
imp_cat.plot(kind = 'barh')


# In[ ]:


# Объединяем

train_data = data.query('sample == 1').drop(['sample'], axis=1)
test_data = data.query('sample == 0').drop(['sample'], axis=1)


# In[ ]:


X_train = train_data.drop(['default'], axis=1)
y_train = train_data.default.values
X_test = test_data.drop(['default'], axis=1)


# In[ ]:


RANDOM_SEED = 42

y = train_data.default.values            
x = train_data.drop(columns=['default'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)


# In[ ]:


# проверяем разбивку
test_data.shape, train_data.shape, X_train.shape, X_test.shape


# In[ ]:


model = LogisticRegression()
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)
probs = probs[:,1]


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()


# In[ ]:


y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Добавим типы регуляризации
penalty = ['l1', 'l2']

# Зададим ограничения для параметра регуляризации
C = np.logspace(0, 4, 10)

# Создадим гиперпараметры
hyperparameters = dict(C=C, penalty=penalty)

model = LogisticRegression()
model.fit(X_train, y_train)

# Создаем сетку поиска с использованием 5-кратной перекрестной проверки
clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)

best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Лучшее C:', best_model.best_estimator_.get_params()['C'])


# In[ ]:


# Обучим модель

model_new = LogisticRegression(penalty='l2', C=166.81005372000593, max_iter=800)
model_new.fit(X_train, y_train)

probs = model_new.predict_proba(X_test)
probs = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()


# In[ ]:


train_data = data.query('sample == 1').drop(['sample'], axis=1)
test_data = data.query('sample == 0').drop(['sample'], axis=1)


# In[ ]:


sample_submission


# In[ ]:


X_train = train_data.drop(['default'], axis=1)
y_train = train_data.default.values
X_test = test_data.drop(['default'], axis=1)


# In[ ]:


predict_submission = model_new.predict_proba(X_test)[:,1]
submission = pd.DataFrame(df_test.client_id)
submission['default'] = predict_submission
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission

