
## 贝叶斯思维：统计建模的Python方法
#### 第一章 贝叶斯公式
联合概率公式:$P(AB) = P(A)\cdot P(B|A)$  

**对于联合概率公式有一个通俗的理解：**假设要将两枚硬币摞在一起，B硬币在上,A硬币在下。那么第一步是先放A，再在A已经放好的前提下把B放上去。

**历时诠释** ：历时意味着某些事情随着时间变化而发生。即假设的概率随着看到的新数据而变化。

根据数据集D更新假设概率H :$P(H | D) = \frac{P(H)P(D | H)}{P(D)}$

>$P(H)$称为先验概率，即得到新数据前某一假设的概率。

>$P(H|D)$称为后验概率，即看到新数据后我们要计算的该假设的概率。

>$P(D|H)$是该假设下得到这一数据的概率，称为**似然度**。

>$P(D)$是在任何假设下得到这一数据的概率，称为**标准化常量**

##### Monty Hall 难题

**先验概率**： 假设每个门后面有车的概率相等，为$\frac{1}{3}$

**似然度**：假设车在A门后，则主持人必须选择一个门后没有车的门，所以打开B门的概率为$\frac{1}{2}$；假设车在B门后，则主持人必须选择C门，所以打开B门的概率为0；假设车在C门后，则主持人必须选择B门，所以打开B门的概率为1。

计算可知$P(A|D) = \frac {1}{3}$, 而$P(C|D) =\frac{2}{3}$

但是如果主持人在可能的情况下总是选择打开B门，则可重新计算得后验概率均为$\frac{1}{2}$

#### 第二章 统计计算

**分布**：在统计上，分布是一组值及其对应的概率


```python
#example1
from thinkbayes.py import Pmf
pmf = Pmf() #创建一个分布对象
for x in list(range(1,7)):
  pmf.Set(x,1/6.0) #设置每个值的概率是1/6

#example2
word_list = ['name', 'daily', 'case', 'number']
pmf = Pmf()
for word in word_list:
  pmf.Incr(word,1)
pmf.Normalize() #归一化
pmf.Prob('name',0) #计算name出现的频率
```

其实`Pmf`是用Python字典来储存值及其概率，所以`Pmf`中的值是任意可被哈希的类型。概率可以是任意数值类型，但通常是浮点数。

##### 曲奇饼问题

在贝叶斯定理的语境下，可以很自然的使用一个``Pmf``映射每个假设和对应的概率。在曲奇饼问题里，该假设是B~1~ 和B~2~。 

~~~   python
pmf = Pmf()
pmf.Set('Bow1',0.5)
pmf.Set('Bow2',0.5)
pmf.Mult('Bow1',0.75)
pmf.Mult('Bow2',0.5)
pmf.Normalize()
0.625
pmf.Prob('Bow1')
0.6000000000000001
~~~

##### 贝叶斯框架

~~~python
from thinkbayes2 import Pmf

#self就是一个分布对象
class cookie(Pmf):

    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    mixes = {'Bow1': dict(vanilla=0.75, chocolate=0.25), 'Bow2': dict(vanilla=0.5, chocolate=0.5)}

    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        like = mix[data]
        return like
hypos = ['Bow1','Bow2']
pmf = cookie(hypos)
pmf.Update('vanilla')
for hypo, prob in pmf.Items():
    print(hypo, prob)

~~~

相比于上一节的代码，这段代码更加复杂。但是同样，这段代码也更加通用，它可以推广到**从同一只碗取一只一个曲奇饼的情形。** 

~~~python
hypos = ['Bow1','Bow2']
pmf = cookie(hypos)
dataset = ['vanilla', 'chocolate', 'vanilla']
for data in dataset:
    pmf.Update('vanilla')
    for hypo, prob in pmf.Items():
        print('{0}, {1:.2f}'.format(hypo, prob))
        
---------------
Bow1, 0.60
Bow2, 0.40
Bow1, 0.69
Bow2, 0.31
Bow1, 0.77
Bow2, 0.23

Process finished with exit code 0
~~~

##### Monty Hall 难题

~~~python
#假设H是车在A，B，C三个门之一中，数据D是主持人打开一扇没有车的门
from thinkbayes2 import Pmf


class monty(Pmf):
    def __init__(self,hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for typo in self.Values():
            like = self.Likelihood(data, typo)
            self.Mult(typo, like)
        self.Normalize()

    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        elif hypo == 'A':
            return 0.5
        else:
            return 1


hypos = ['A', 'B', 'C']
pmf = monty(hypos)
data = 'B'
pmf.Update(data)
for hypo, prob in pmf.Items():
    print("{0},{1:.2f}".format(hypo, prob))
--------------------------------
A,0.33
B,0.00
C,0.67

Process finished with exit code 0
~~~

##### 封装框架



