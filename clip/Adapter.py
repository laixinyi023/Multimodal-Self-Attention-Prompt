

import torch.nn as nn
import torch
class Adapter(nn.Module):
    def __init__(self, D_features,out_features, act_layer=nn.GELU,ratio = 0.25):
        super().__init__()
        D_hidden_features = int(D_features * ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, out_features)

    def forward(self, x):
        # x is expected (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        return x + xs
# class AdapterForPrompt(nn.Module):
#     def __init__(self, D_features,out_features, act_layer=nn.GELU,ratio = 0.25):
#         super().__init__()
#         #D_hidden_features = int(D_features * ratio)
#         #self.act = act_layer()
#         self.D_fc1 = nn.Linear(D_features, D_hidden_features)
#         self.D_fc2 = nn.Linear(D_hidden_features, out_features)
#
#     def forward(self, x):
#         # x is expected (BT, HW+1, D)
#         xs = self.D_fc1(x)
#         #xs = self.act(xs)
#         xs = self.D_fc2(xs)
#
#         return xs
class Adapter2Linear(nn.Module):
    def __init__(self, D_features,out_features, act_layer=nn.GELU,ratio = 1):
        super().__init__()
        D_hidden_features = int(D_features * ratio)
        # self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, out_features)
        # self.D_fc2 = nn.Linear(D_hidden_features, out_features)

    def forward(self, x):
        # x is expected (BT, HW+1, D)
        xs = self.D_fc1(x)
        # xs = self.act(xs)
        #xs = self.dropout(xs)

        # xs = self.D_fc2(xs)

        return xs
class Adapter2Prompt(nn.Module):
    def __init__(self, D_features,out_features, act_layer=nn.ReLU,ratio = 1):
        super().__init__()
        D_hidden_features = int(D_features * ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        # self.D_fc2 = nn.Linear(D_hidden_features, out_features)

    def forward(self, x):
        # x is expected (BT, HW+1, D)
        xs = self.D_fc1(x)
        # xs = self.act(xs)
        # xs = self.D_fc2(xs)

        return xs

class Adapter2Transformer(nn.Module):
    def __init__(self, D_features,out_features, act_layer=nn.GELU,ratio = 0.25):
        super().__init__()
        self.length = D_features
        D_hidden_features = int(D_features * ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, out_features)

    def forward(self, inputs):
        #x = [grid ** 2 +1 +n_ctx, batch_size,width]



        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        cls = x[0:1,:,:]

        xs = x[1:1+ self.length,:,:]
        suffix = x[1+self.length:,:,:]
        xs = xs.permute(1,2,0)

        # x = [batch_size,width, grid **2]



        xs = self.D_fc1(xs)
        xs = self.act(xs)
        xs = self.D_fc2(xs)


        xs = xs.permute(2,0,1)
        xs = torch.cat([cls,xs,suffix],dim=0)
        x = xs + x
        return [x, compound_prompts_deeper, counter]