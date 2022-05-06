import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot()
ax.plot(x,y,marker='o',label='plot1')
ax.plot(x,y,marker='o',label='plot2')
ax.legend(fontsize=20)
plt.xlabel('y축',fontsize=15)
plt.ylabel('x축',fontsize=15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('제목',fontsize=20)
plt.show()
