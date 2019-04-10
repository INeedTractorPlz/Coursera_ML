
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
get_ipython().magic(u'matplotlib inline')


# # Дискретное распределение

# Сгенерируем выборку объёма 100 из дискретного распределения с шестью равновероятными исходами.

# In[8]:


sample = np.random.choice([1,2,3,4,5,6], 100)


# Представим теперь, что эта выборка была получена не искусственно, а путём подбрасывания симметричного шестигранного кубика 100 раз. Оценим вероятности выпадения каждой из сторон с помощью частот:

# In[9]:


# посчитаем число выпадений каждой из сторон:
from collections import Counter

c = Counter(sample)

print("Число выпадений каждой из сторон:")    
print(c)

# теперь поделим на общее число подбрасываний и получим вероятности:
print("Вероятности выпадений каждой из сторон:")
print({k: v/100.0 for k, v in c.items()})


# Это и есть оценка функции вероятности дискретного распределения.

# # Непрерывное распределение

# Сгенерируем выборку объёма 100 из стандартного нормального распределения (с $\mu=0$ и $\sigma^2=1$):

# In[10]:


norm_rv = sts.norm(0, 1)
sample = norm_rv.rvs(100)


# Эмпирическая функция распределения для полученной выборки:

# In[11]:


x = np.linspace(-4,4,100)
cdf = norm_rv.cdf(x)
plt.plot(x, cdf, label='theoretical CDF')

# для построения ECDF используем библиотеку statsmodels
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(sample)
plt.step(ecdf.x, ecdf.y, label='ECDF')

plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.legend(loc='upper left')


# Гистограмма выборки:

# In[12]:


plt.hist(sample, normed=True)
plt.ylabel('fraction of samples')
plt.xlabel('$x$')


# Попробуем задавать число карманов гистограммы вручную:

# In[13]:


plt.hist(sample, bins=3, normed=True)
plt.ylabel('fraction of samples')
plt.xlabel('$x$')


# In[14]:


plt.hist(sample, bins=40, normed=True)
plt.ylabel('fraction of samples')
plt.xlabel('$x$')


# Эмпирическая оценка плотности, построенная по выборке с помощью ядерного сглаживания:

# In[18]:


# для построения используем библиотеку Pandas:
df = pd.DataFrame(sample, columns=['KDE'])
ax = df.plot(kind='density')
print(df)
get_ipython().magic(u'pinfo print')
get_ipython().magic(u'pinfo df')

# на том же графике построим теоретическую плотность распределения:
x = np.linspace(-4,4,100)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf, label='theoretical pdf', alpha=0.5)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

