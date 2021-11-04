# abc
实践作业

项目说明：

在不使用封装好的LSTM 模型的情况下，实现单层和双层LSTM模型循环部分

LSTM.py为不使用封装好的LSTM的单层模型
doubleLSTMLM.py为不使用封装好的LSTM的双层模型

实现设计：

i_t = torch.sigmoid(x_t @ self.U_i + h_t0 @ self.V_i + self.b_i)  # 输入门
g_t = torch.tanh(x_t @ self.U_c + h_t0 @ self.V_c + self.b_c)  # 输入门
c_t = f_t * c_t0 + i_t * g_t  # 记忆更新
f_t = torch.sigmoid(x_t @ self.U_f + h_t0 @ self.V_f + self.b_f)  # 遗忘门
o_t = torch.sigmoid(x_t @ self.U_o + h_t0 @ self.V_o + self.b_o)  # 输出门
 h_t = o_t * torch.tanh(c_t)  # 输出门
