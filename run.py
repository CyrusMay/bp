from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
from bp_model import Cyrus_BP

bp = Cyrus_BP([50,10,3],lr=0.01,epoch = 2e5,activate = ["softsign","softsign","softsign"])
data = load_iris()
X = data["data"]
Y = data["target"]

# 用神经网络进行分类时，需把输出先进行独热编码
Y1 = pd.get_dummies(Y1) # 进行独热编码或将期望输出转换为哑变量
bp.fit(X,Y1)
Y_pre = bp.predict_label(X)
print("准确率为：",accuracy_score(Y,Y_pre))
