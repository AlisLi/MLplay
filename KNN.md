### 一、KNN算法思路

最近邻居法（KNN算法，又译K-近邻算法）是一种用于分类和回归的非参数统计方法。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;例如：根据肿瘤大小和时间两个特征对肿瘤的良性和恶性的预测。其中绿色为良性，红色为恶性，现在要预测蓝色这一数据是良性还是恶性，那么对于kNN算法是怎么预测的呢？

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;大概思路：首先取一个k值（**要点一：K值的取法**），之后取离索要预测的点最近的k个数据（**要点二：数据距离的计算**），选取k个数据中所属最多的一类作为所预测的值。（**要点三：如何分类**）

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以上是一种简单的思路，对于不同的要点中，所选择的算法不同，则预测的准确度也不相同。下面进行详细的分析。
![kNN算法图解](https://github.com/AlisLi/MLplay/blob/master/MLplay/image/knn_1.png)

### 二、要点一：k的选取
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如何设置k使得效果最好？这里涉及到两个概念**超参数**和**模型参数**。k就是一个超参数。

 - 超参数：超参数是在开始学习过程之前设置值的参数。通常情况下，需要对超参数进行优化，给学习机选择一组最优超参数，以提高学习的性能和效果。
 - 模型参数：通过训练得到的参数数据。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于我们来讲，要找一个好的参数，一般是在一个有限的数值范围内进行选取，然后循环测试找出使得评价函数最高的k的值。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果k的值是临界值，比如：1或者10，则说明在这个范围外有可能还存在更好的k值，这时我们应该扩大范围进行选取。

```
best_score = 0.0
best_k = -1
for k in range(1,11):
    knn_cls = KNeighborsClassifier(n_neighbors=k)
    knn_cls.fit(X_train, y_train)
    score = knn_cls.score(X_test, y_test)
    if score > best_score:
        best_k = k
        best_score = score

print("best_k = ", best_k)
print("best_score = ", best_score)
结果：
best_k =  4
best_score =  0.9916666666666667
```
### 三、要点二：数据距离的计算

对于两个数据点之间的距离的计算，我们使用**欧氏距离**来进行计算，当然在这之前，需要进行数据归一化（见：）的处理。
距离公式如下：

d(x,y):=<a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{(x_1-y_1)^2&space;&plus;(x_2-y_2)^2&space;&plus;&space;\cdots&space;&plus;(x_n-y_n)^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{(x_1-y_1)^2&space;&plus;(x_2-y_2)^2&space;&plus;&space;\cdots&space;&plus;(x_n-y_n)^2}" title="\sqrt{(x_1-y_1)^2 +(x_2-y_2)^2 + \cdots +(x_n-y_n)^2}" /></a> = <a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{\sum_{i=1}^n(x_i-y_i)^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sqrt{\sum_{i=1}^n(x_i-y_i)^2}" title="\sqrt{\sum_{i=1}^n(x_i-y_i)^2}" /></a>

其中，a，b为两个样本点，n为样本特征数量。

### 四、要点三：如何分类

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1、最简单的分类方法就是按照样本个数最多的来进行分类。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2、但是，若是存在个数相等情况则1无法解决。（用到**网格搜索**）

![预测数据距离样本点](https://github.com/AlisLi/MLplay/blob/master/MLplay/image/knn_2.png)

若是存在三类，且预测数据距离样本点如图所示，这时候可以通过距离作为权重来进行分类。将距离的倒数大小作为分类的标准。红色：1；紫色：1/3；蓝色：1/4；则属于红色。



