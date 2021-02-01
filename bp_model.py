import numpy as np
import time
class Cyrus_BP(object):
    """
    layer 为神经网络各层神经元的个数,包括输出层神经元个数,传参形式以列表传入；
    activate:为各层的激活函数，传参形式为字符串或列表，
             若传入一个字符串，则各层激活函数相同，
             若传入一个列表，则列表元素代表各层激活函数
             可传参数有：（1）sigmoid：S型函数
                         （2）tanh：双曲正弦函数
                         （3）relu:max(0,x)函数
                         （4）purline：线性函数
                         （5）softsign：平滑函数

    lr:学习率，默认为0.01
    epoch：最大迭代次数 默认为1e4
    该模型具有的主要方法和属性如下：
    fit(X,Y):模型拟合方法
    predict(X):输出预测方法
    predict_label(X):分类标签输出预测方法
    activate:激活函数列表
    W：权值列表
    
    """
    def __init__(self,layer,**kargs):
        self.layer = np.array(layer).reshape(1,-1)
        if 'activate' in kargs.keys():
            if str(type(kargs["activate"])) == "<class 'str'>":    
                self.activate = [kargs["activate"]]*int(len(layer))
            else:
                self.activate = kargs["activate"]
        else:
            self.activate = ["sigmoid"]*int(len(layer))
        self.diff_activate = []
        if 'lr' in kargs.keys():
            self.lr = kargs["lr"]
        else:
            self.lr = 0.01
        if 'epoch' in kargs.keys():
            self.epoch = kargs["epoch"]
        else:
            self.epoch = int(1e4)
            
        self.X = None
        self.Y = None
        self.W = None
        self.output = []
        self.delta = []
        self.sum_input = []
    # 1、选择激活函数
    def activation_func(self):
        temp_func = []
        for i in range(len(self.activate)):
            if self.activate[i] == "sigmoid":
                temp_func.append(lambda x:1/(1+np.exp(-x)))
                self.diff_activate.append(lambda x:(1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x)))))
            if self.activate[i] == "tanh":
                temp_func.append(lambda x:(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))
                self.diff_activate.append(lambda x:((-np.exp(x) + np.exp(-x))*(np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))**2 + 1))
            if self.activate[i] == "softsign":
                temp_func.append(lambda x:x/(1+np.abs(x)))
                self.diff_activate.append(lambda x:1/((1+x/np.abs(x)*x)**2))
            if self.activate[i] == "relu":
                temp_func.append(lambda x:(x+np.abs(x))/(2*np.abs(x))*x)
                self.diff_activate.append(lambda x:(x+np.abs(x))/(2*np.abs(x)))
            if self.activate[i] == "purline":
                temp_func.append(lambda x:x)
                self.diff_activate.append(lambda x:1+x-x)
        self.activate = temp_func
    # 2、权值初始化函数
    def init_w(self):
        self.W = []
        for i in range(self.layer.shape[1]):
            if i == 0:
                w = np.random.random([self.X.shape[1]+1,self.layer[0,i]])*2-1
            else:
                w = np.random.random([self.layer[0,i-1]+1,self.layer[0,i]])*2-1
            self.W.append(w)
     
    # 3、权值调整函数
    def update_w(self):
        # 1 计算各层输出值
        self.output = []
        self.sum_input = []
        for i in range(self.layer.shape[1]):
            if i == 0:
                temp = np.dot(np.hstack((np.ones((self.X.shape[0],1)),self.X)),self.W[i])
                self.sum_input.append(temp)
                self.output.append(self.activate[i](temp))
            else:
                temp = np.dot(np.hstack((np.ones((self.output[i-1].shape[0],1)),self.output[i-1])),self.W[i])
                self.sum_input.append(temp)
                self.output.append(self.activate[i](temp))
        # 2 求每层的学习信号
        self.delta = [0 for i in range(len(self.output))]
        for i in range(len(self.output)):
            if i == 0:
                self.delta [-i-1] = ((self.Y-self.output[-i-1])*self.diff_activate[-i-1](self.sum_input[-i-1]))
            else:
                self.delta [-i-1] = ((self.delta[-i].dot(self.W[-i][1:,:].T))*self.diff_activate[-i-1](self.sum_input[-i-1]))
        # 3 更新权值
        for i in range(len(self.W)):
            if i == 0 :
                self.W[i] += self.lr * np.hstack((np.ones((self.X.shape[0],1)),self.X)).T.dot(self.delta[i])
            else:
                self.W[i] += self.lr * np.hstack((np.ones((self.output[i-1].shape[0],1)),self.output[i-1])).T.dot(self.delta[i])
                            
    def fit(self,X,Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        # 1 权值初始化
        self.init_w()

        # 2 选择激活函数
        self.activation_func()
        # 3 更新权值
        start_time = time.time()
        for i in range(int(self.epoch)):
            self.update_w()
            end_time = time.time()
            if end_time - start_time >= 5:
                print("Epoch%d:"%(i+1),np.mean(np.square(self.Y-self.output[-1])))
                print("\n")
                start_time = time.time()
    def predict(self,x):
        x = np.array(x)
        result = []
        for i in range(self.layer.shape[1]):
            if i == 0:
                result.append(self.activate[i](np.dot(np.hstack((np.ones((x.shape[0],1)),x)),self.W[i])))
            else:
                result.append(self.activate[i](np.dot(np.hstack((np.ones((result[i-1].shape[0],1)),result[i-1])),self.W[i])))
        return result[-1]
    def predict_label(self,x):
        x = np.array(x)
        result = []
        for i in range(self.layer.shape[1]):
            if i == 0:
                result.append(self.activate[i](np.dot(np.hstack((np.ones((x.shape[0],1)),x)),self.W[i])))
            else:
                result.append(self.activate[i](np.dot(np.hstack((np.ones((result[i-1].shape[0],1)),result[i-1])),self.W[i])))
        result = result[-1]   
        return np.array([result[i].argmax() for i in range(result.shape[0])]).reshape(-1,1)
    
