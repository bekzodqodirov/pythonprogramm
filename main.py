import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#pandas csv filedan har bir columni ajratib olish un ishlatilgan
file = pandas.read_csv('cost_revenue_clean.csv')
x = DataFrame(file, columns=['production_budget_usd'])
y = DataFrame(file, columns=['worldwide_gross_usd'])


#Linear algebra orqali hsoblangan data "sklearn library" algaritmlani beradi
regression = LinearRegression()
regression.fit(x, y)

#matplotlib orqali datalani visualize qilingan
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.2)
plt.plot(x,regression.predict(x), color = 'red', linewidth = 1)
plt.title('Filmcost vs reveneu')
plt.xlabel('cost of film')
plt.ylabel('worldwide growth')
plt.ylim(0, 3000000000)
plt.xlim(0.450000000)
plt.show()
