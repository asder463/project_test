# project_test
#心得：一开始不知道神经网络是什么，于是去b站及csdn上搜索了相关资料（包括后续的一些问题基本都是在csdn和b站上搜索才得以解决），在查阅相关资料以后我才了解原来神经网络没有那么神秘，本质上也就是一个包含了
很多很多参数的函数而已。
#然后开始着手做level1，第一步是搭建环境，因为很少使用命令行，也没有自己搭建环境的经验，我安装anaconda，cuda和pycharm就花费了大概一天的时间，并且因为pycharm的环境路径似乎没导入成功，我只能使用jupyter来写代码。
#在这之后，我先是跟着b站上的教程学习，然后发现内容太多了，如果按部就班的话时间来不及（无奈），所以在浅浅地找了一个针对mnist数据集的实战视频看了看大概流程以后，我就把学习重心放在了csdn的教程上，在阅读这些教程的过程中我逐渐了解了tensor，torch等等概念，在把这些模块的功能稍微熟悉之后，我开始正式写代码，当然也是面向b站和csdn编程（笑）。
#写的过程当然也遇到很多困难，比如只学过python的一点皮毛，对类和对象不了解，很多时候看不懂代码也不敢自己动手更改参数。不过尝试的多了也稍微摸到一点规律，加载数据集因为我发现mnist可以直接用datasets下载，于是就偷懒没有用学长们给的数据包。
#同时我也犹豫是否要使用Sequential来封装层级，在尝试的过程中我发现它确实会降低网络可塑性，比如不能在里面使用nn.Softmax函数（也可能是我不会用），所以我选择第一个level使用，第二个不使用。
#再就是loss的计算问题，因为给的label是一个数，而我的pred则是一个矩阵，所以我选择了onehot函数将label转化为矩阵，然后用mse计算pred和label的均方误差。
#还有网络的评估，因为mnist是一个多分类问题，我不知道如何定义正标签和负标签，所以我选择直接计算预测正确的个数与总个数之商来作为准确率评估整个网络。
#在训练的过程中我发现用cpu训练太慢了，所以我采用gpu进行训练，但是更改代码后报错不断，最后花了一个小时把net，loss和数据全部转移到gpu上（汗）。
#做RNN网络时，我效率明显快了很多，把第一个net的代码粘贴一下然后根据循环算法的原理更改一下网络结构就完事了，遇到最大的问题是因为网络的输出有两个量，即output和hidden，在计算loss时显示两个矩阵维度不同，因此无法计算，在搜索加操作一番后，也顺利解决了。

![Image text](
