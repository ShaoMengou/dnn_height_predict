import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import cv2 as cv

class HNet(nn.Module):
    def __init__(self, input_dim):
        super(HNet, self).__init__()

        self.relu = nn.ReLU()

        dim_1 = int(math.sqrt(input_dim)) 
        self.fc1 = nn.Linear(input_dim, dim_1) # 第一个隐藏层
        dim_2 = int(math.sqrt(dim_1))
        self.fc2 = nn.Linear(dim_1,dim_2) # 第二个隐藏层
        dim_3 = int(math.sqrt(dim_2))
        self.fc3 = nn.Linear(dim_2,dim_3) # 第三个隐藏层
        self.out = nn.Linear(dim_3,1) # 输出层线性处理
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.out(out)
        return out
        
def train_model(model,data, num_epochs=10, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, height_target in data:
            optimizer.zero_grad()
            outputs = model(torch.tensor(inputs))
            loss = criterion(outputs, torch.tensor(height_target))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, running_loss))
    
def predict(model, input_tensor):
    output_tensor = model(input_tensor)

    return output_tensor.item()

info_train = list(map(float,open("height_info_train.txt").read().splitlines()))
info_test = list(map(float,open("height_info_test.txt").read().splitlines()))

cnt_frame_train:int = len(info_train)
cnt_frame_test:int = len(info_test)

print("训练集合有"+str(cnt_frame_train)+"条数据")
print("测试集合有"+str(cnt_frame_test)+"条数据")

cap = cv.VideoCapture("test.avi")

input_dim = cnt_frame_train

# 如果你想要重新训练，解除下一段内容的注释
# train_data = []
# 
# for i in range(cnt_frame_train):
#     ret,img=cap.read()
#     img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     img = img[:,0]
#     img=np.array(img,dtype=np.float32)
# 
#     for j in range(1,len(img)):
#         img[len(img)-j]-=img[len(img)-j-1]
#     img[0]=0.0
# 
#     train_data.append((img,float(info_train[i])))

my_model = HNet(648)

# 如果你想要重新训练，为下一段内容加上注释
my_model.load_state_dict(torch.load("model_35_5.model"))

# 如果你想要重新训练，解除下一段内容的注释
# train_model(my_model, train_data, num_epochs=cnt_frame_train, learning_rate=0.0001)
# torch.save(my_model.state_dict(),'model_35_5.model')

outputs = []

for i in range(cnt_frame_train):
    ret,img = cap.read()
    img_show = img
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = img[:,0]
    img=np.array(img,dtype=np.float32)

    for j in range(1,len(img)):
        img[len(img)-j]-=img[len(img)-j-1]
    img[0]=0.0

    output = predict(my_model,torch.tensor(img))
    outputs.append(output)

    img_show = cv.putText(img_show,"predict: "+f"{output:2.3f}",(100,100),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2,cv.LINE_AA,False)
    img_show = cv.putText(img_show,"real:    "+f"{info_train[i]:2.3f}",(100,128),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2,cv.LINE_AA,False)
    img_show = cv.putText(img_show,"loss:    "+f"{abs(info_train[i]-output):2.3f}",(100,156),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,165,255),2,cv.LINE_AA,False)
    img_show = cv.putText(img_show,"loss(%): "+f"{abs((info_train[i]-output)/info_train[i])*100.0:2.3f}",(100,188),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,165,255),2,cv.LINE_AA,False)
    cv.imshow("hey",img_show);

    if cv.waitKey(10) >=0:
        break

# 画点图出来
from matplotlib import pyplot as plt

plt.plot([abs((outputs[i] - info_train[i])/info_train[i])*100 for i in range(cnt_frame_train)])# 相对误差
plt.show()
