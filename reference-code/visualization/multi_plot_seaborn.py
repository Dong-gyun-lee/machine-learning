import seaborn as sns
# 행과 열, 그림으로 category 구분하여 plot 한 번에 그리기

data_visual = sns.FacetGrid(data,row='행으로 구분할 column',row_order=['row_values'],col='열로 구분할 column',col_order=['col_values'],hue='그림 안에서 구분')
data_visual.map(sns.pointplot,'그림 안에 row_value','value')  #point plot
data.add_legend()

# barplot 등 다양한 plot도 비슷하게 그릴 수 있다.
porplot_os.map(sns.barplot,'buckets','proportion') #barplot

# seaborn package plot 은 대표적으로 아래와 같이 있다. 사용법은 https://seaborn.pydata.org/index.html# 로 가면 다 확인 가능
# distplot, countplot
# jointplot, pairplot
# heatmap
# barplot, boxplot, pointplot, violinplt, stripplot, swarmplot
