import numpy as np
import torch
from torch import nn
Tensor = torch.Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn import functional as F
# from m2transformer.models.containers import Module
def linear(input: Tensor, weight: Tensor, bias: [Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """

    return torch._C._nn.linear(input, weight, bias)



class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        # '''
        # :param d_model: Output dimensionality of the model
        # :param d_k: Dimensionality of queries and keys
        # :param d_v: Dimensionality of values
        # :param h: Number of heads
        # '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


def ScaledDotProductAttention(queries, keys, values,in_proj_weight,in_proj_bias,out_proj,h,d_model,attention_mask):

    w_q,w_k,w_v = in_proj_weight.chunk(3)
    b_q,b_k,b_v = in_proj_bias.chunk(3)

    d_q = d_model // h
    d_k = d_model // h
    d_v = d_model // h

    queries = queries.permute(1,0,2)
    keys = keys.permute(1, 0, 2)
    values = values.permute(1, 0, 2)



    w_q = w_q.half()
    w_k = w_k.half()
    w_v = w_v.half()
    b_q = b_q.half()
    b_k = b_k.half()
    b_v = b_v.half()

    queries.half()
    keys.half()
    values.half()

    b_s, nq = queries.shape[:2]
    nk = keys.shape[1]
    q = linear(queries,w_q,b_q).view(b_s, nq, h, d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
    k = linear(keys,w_k,b_k).view(b_s, nk, h, d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
    v = linear(values,w_v,b_v).view(b_s, nk, h, d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

    att = torch.matmul(q, k) / np.sqrt(d_k)  # (b_s, h, nq, nk)
    # if attention_weights is not None:
    #     att = att * attention_weights
    if attention_mask is not None:
        att = att + attention_mask
    att = torch.softmax(att, -1)
    out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, h * d_v)  # (b_s, nq, h*d_v)
    out = out_proj(out)  # (b_s, nq, d_model)
    out = out.permute(1,0,2)
    return out







# class ScaledDotProductAttentionMemory(nn.Module):
#     '''
#     Scaled dot-product attention with memory
#     '''
#
#     def __init__(self, d_model, d_k, d_v, h, m):
#         '''
#         :param d_model: Output dimensionality of the model
#         :param d_k: Dimensionality of queries and keys
#         :param d_v: Dimensionality of values
#         :param h: Number of heads
#         :param m: Number of memory slots
#         '''
#         super(ScaledDotProductAttentionMemory, self).__init__()
#         self.fc_q = nn.Linear(d_model, h * d_k)
#         self.fc_k = nn.Linear(d_model, h * d_k)
#         self.fc_v = nn.Linear(d_model, h * d_v)
#         self.fc_o = nn.Linear(h * d_v, d_model)
#         self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
#         self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))
#
#         self.d_model = d_model
#         self.d_k = d_k
#         self.d_v = d_v
#         self.h = h
#         self.m = m
#
#         self.init_weights()
#
#     def init_weights(self):
#         nn.init.xavier_uniform_(self.fc_q.weight)
#         nn.init.xavier_uniform_(self.fc_k.weight)
#         nn.init.xavier_uniform_(self.fc_v.weight)
#         nn.init.xavier_uniform_(self.fc_o.weight)
#         nn.init.normal_(self.m_k, 0, 1 / self.d_k)
#         nn.init.normal_(self.m_v, 0, 1 / self.m)
#         nn.init.constant_(self.fc_q.bias, 0)
#         nn.init.constant_(self.fc_k.bias, 0)
#         nn.init.constant_(self.fc_v.bias, 0)
#         nn.init.constant_(self.fc_o.bias, 0)
#
#     def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
#         '''
#         Computes
#         :param queries: Queries (b_s, nq, d_model)
#         :param keys: Keys (b_s, nk, d_model)
#         :param values: Values (b_s, nk, d_model)
#         :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#         :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         :return:
#         '''
#
#
#
#         b_s, nq = queries.shape[:2]
#         nk = keys.shape[1]
#
#         m_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.m, self.h * self.d_k)
#         m_v = np.sqrt(self.m) * self.m_v.expand(b_s, self.m, self.h * self.d_v)
#
#         q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k = torch.cat([self.fc_k(keys), m_k], 1).view(b_s, nk + self.m, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
#         v = torch.cat([self.fc_v(values), m_v], 1).view(b_s, nk + self.m, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
#         if attention_weights is not None:
#             att = torch.cat([att[:, :, :, :nk] * attention_weights, att[:, :, :, nk:]], -1)
#         if attention_mask is not None:
#             att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(attention_mask, -np.inf)
#         att = torch.softmax(att, -1)
#         out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out = self.fc_o(out)  # (b_s, nq, d_model)
#         return out

class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()


    def forward(self, x):
        return x * torch.sigmoid(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, int(0.5 * input_size))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int (0.5 * input_size), hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MultiHeadAttention(nn.Module):
    # '''
    # # Multi-head attention layer with Dropout and Layer Normalization.
    # # '''

    def __init__(self, d_model, d_k, d_v, h, dropout=0.5, m=8,identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None,device=None, dtype=None,is_text_layer= False,layer = 0):
        super(MultiHeadAttention, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        embed_dim = d_model
        self.identity_map_reordering = identity_map_reordering
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.is_text_layer = is_text_layer

        # self.mlp_adapter = MLP(d_model,1)
        # if attention_module is not None:
        #     if attention_module_kwargs is not None:
        #         self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
        #     else:
        #         self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        # else:
        #     self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        #

        self.dropout = nn.Dropout(dropout)
        # self.swiglu = SwiGLU()
        self.swiglu = nn.GELU()
        self.layer = layer
        # self.dropout_add = nn.Dropout(0.5)
        # self.can_be_stateful = can_be_stateful
        # if self.can_be_stateful:
            # self.register_state('running_keys', torch.zeros((0, d_model)))
            # self.register_state('running_values', torch.zeros((0, d_model)))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        # self.adapter_k_weight = nn.Parameter(torch.HalfTensor(1,1).to(device)).half()
        # self.adapter_v_weight = nn.Parameter(torch.HalfTensor(1,1).to(device)).half()
        # self.adapter_k_weight = self.adapter_k_weight.to(device)
        # self.adapter_v_weight = self.adapter_v_weight.to(device)
        # self.m_k_adapter = nn.Parameter(torch.HalfTensor(1, m, h * d_k))
        # self.m_v_adapter = nn.Parameter(torch.HalfTensor(1, m, h * d_v))
        self.m = m
        # self.init_weight()
        # nn.init.normal_(self.m_k_adapter, 0, 1 / d_k)
        # nn.init.normal_(self.m_v_adapter, 0, 1 / self.m)

    # def init_weight(self):
    #     nn.init.ones_(self.adapter_k_weight)
    #     nn.init.ones_(self.adapter_v_weight)

    def forward(self, queries, keys, values,k = None,v = None,addk=None,addv=None,attn_mask=None, attention_weights=None,need_weights=None):


        # if self.can_be_stateful and self._is_stateful:
        #     self.running_keys = torch.cat([self.running_keys, keys], 1)
        #     keys = self.running_keys
        #
        #     self.running_values = torch.cat([self.running_values, values], 1)
        #     values = self.running_values

        # if self.identity_map_reordering:
        #     q_norm = self.layer_norm(queries)
        #     k_norm = self.layer_norm(keys)
        #     v_norm = self.layer_norm(values)
        #     out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
        #     out = queries + self.dropout(torch.relu(out))
        # else:
        # out = ScaledDotProductAttentionMemory(queries, keys, values,self.in_proj_weight,self.in_proj_bias,
        #                                       self.out_proj,self.h,self.d_model, attn_mask,k,v,self.m,self.adapter_k_weight,self.adapter_v_weight)
        if k !=  None:
            k = self.dropout(k)
            v = self.dropout(v)
            out = self.ScaledDotProductAttentionMemory(queries, keys, values, self.in_proj_weight, self.in_proj_bias,
                              self.out_proj, self.h, self.d_model, attn_mask, k, v, self.m,enhance=self.training,is_text_layer = self.is_text_layer,layer=self.layer)

        else:
            out = ScaledDotProductAttention(queries,keys,values,self.in_proj_weight,self.in_proj_bias,self.out_proj,self.h,self.d_model,attn_mask)
        # out = self.dropout(out)
        #out = self.layer_norm(queries + out)
        return out

    def ScaledDotProductAttentionMemory(self,queries, keys, values, in_proj_weight, in_proj_bias, out_proj, h, d_model,
                                        attention_mask, m_k,
                                        m_v, m, k_weight=1, v_weight=1, enhance=False, is_text_layer=False, layer=0):
        queries = queries.permute(1, 0, 2)
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        w_q, w_k, w_v = in_proj_weight.chunk(3)
        b_q, b_k, b_v = in_proj_bias.chunk(3)

        # d_q = d_model // h
        d_k = d_model // h
        d_v = d_model // h

        # m_k = drop(m_k)
        # m_v = drop(m_v)

        # m_k =  m_k.expand(b_s, m, h * d_k)
        # m_v =  m_v.expand(b_s, m, h * d_v)

        # if not self.training:
        # m = 1
        # m_k = m_k.view(m_v.shape[0],m,-1,m_k.shape[-1])
        # m_k,_ = torch.max(m_k,dim=2)
        # # m_k = torch.mean(m_k,dim=2)
        # m_v = m_v.view(m_v.shape[0],m,-1,m_v.shape[-1])
        # # m_v = torch.mean(m_v, dim=2)
        # m_v,_= torch.max(m_v, dim=2)

        # m_q = np.sqrt(m) *m_q.expand(b_s,m,h * d_v)

        w_q = w_q.half()
        w_k = w_k.half()
        w_v = w_v.half()
        b_q = b_q.half()
        b_k = b_k.half()
        b_v = b_v.half()

        queries.half()
        keys.half()
        values.half()
        q = linear(queries, w_q, b_q)  # (b_s, h, nq, d_k)[
        # print(queries.shape)
        # print(m_v.shape)

        # print(m_k.shape) #(1,m,512)
        # exit(0)


        # padding = torch.zeros(1,2,m_k.shape[-1]).half().to(m_k.device)
        # m_k = torch.cat([m_k,padding],dim=1)
        # m_v = torch.cat([m_v, padding], dim=1)
        m = m_k.shape[1]
        # origin

        m_k = np.sqrt(d_k * h) * k_weight * m_k.expand(b_s, m, h * d_k)
        m_v = np.sqrt(m) * v_weight * m_v.expand(b_s, m, h * d_v)

        # [22, 4, 512]


        q = q.view(b_s, nq, h, d_k).permute(0, 2, 1, 3)





        k = linear(keys, w_k, b_k)
        v = linear(values, w_v, b_v)

        if (not is_text_layer and layer < 3) or layer >= 11 :
            m = 0
        else:
            k = torch.cat([k, m_k], 1)
            v = torch.cat([v, m_v], 1)
        # if layer >= 4:
        #     m = 0
        # else:
        #     k = torch.cat([k, m_k], 1)
        #     v = torch.cat([v, m_v], 1)

        k = k.view(b_s, nk + m, h, d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = v.view(b_s, nk + m, h, d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(d_k)  # (b_s, h, nq, nk)

        # if attention_weights is not None:
        #     att = att * attention_weights
        if attention_mask is not None:
            att[:, :, :, :nk] = att[:, :, :, :nk] + attention_mask
        att = torch.softmax(att, -1)
        # if(not self.training):
        #     print(torch.mean(torch.mean(att[:,:,:,-4:],dim=-2),dim=1))
        # exit(0)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, h * d_v)  # (b_s, nq, h*d_v)
        out = out_proj(out)  # (b_s, nq, d_model)
        out = out.permute(1, 0, 2)

        return out
