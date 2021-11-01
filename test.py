import matplotlib.pyplot as plt


x = [i for i in range(24)]

plt.plot(x,x,alpha=0)
plt.plot([5,7], [1, 9], 'ro-')
plt.plot([5,8], [2, 10], 'go-')
plt.xlim(0,30)
plt.show()