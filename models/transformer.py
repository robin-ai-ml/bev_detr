# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        #the d_model must be divisable by nhead
        assert d_model % nhead == 0

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()


        self.d_model = d_model
        self.nhead = nhead
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # TODO: 实现 Transformer 模型的前向传播逻辑
        # 1. 将输入展平，将形状从 (bs, c, h, w) 变为 (hw, bs, c)
        #Given the input tensor of shape (batch_size, channel, height, width), 
        #the goal is to convert it to (height * width, batch_size, channel)
        bs, c, h, w = src.shape
        shape = src.shape
        src = src.view(shape[0], shape[1], -1)
        src = src.permute(2, 0, 1)

        shape = pos_embed.shape
        pos_embed = pos_embed.view(shape[0], shape[1], -1)
        pos_embed = pos_embed.permute(2, 0, 1)

        #convert the key padding mask (batch_size, height, width) to  (batch_size, height * width)
        mask = mask.flatten(1)

        # 2. 初始化需要预测的目标 query embedding  ????
        #nn.init.normal_(query_embed.weight, mean=0.0, std=1)
  
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (num_queries, batch_size, d_model),[100,2,256]

        # 3. 使用编码器处理输入序列，得到具有全局相关性（增强后）的特征表示
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        # 4. 使用解码器处理目标张量和编码器的输出，得到output embedding
        tgt = torch.zeros_like(query_embed)

        decoder_output = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        
        # 5. 对输出结果进行形状变换，并返回
        # decoder输出 [1, 100, bs, 256] -> [1, bs, 100, 256]        
        # encoder输出 [bs, 256, H, W]
        
        # Step 5: Reshape the output if necessary
        # hs: [num_layers, num_queries, batch_size, d_model]
        return decoder_output.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w) 



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # TODO: 实现 Transformer 编码器的前向传播逻辑
        # 1. 遍历$num_layers层TransformerEncoderLayer
        output_seq = src
        for encoder_layer in self.layers:
            output_seq = encoder_layer(output_seq, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        # 2. layer norm
        if self.norm is not None:
            output_seq = self.norm(output_seq)
        # 3. 得到最终编码器的输出
        return output_seq

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # 是否返回中间层输出结果
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        output_seq = tgt
        intermediate = []

        # TODO: 实现 Transformer 解码器的前向传播逻辑
        # 1. 遍历$num_layers层TransformerDecoderLayer，对每一层解码器进行前向传播，并处理return_intermediate为True的情况
        for decoder_layer in self.layers:
            output_seq = decoder_layer(output_seq, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, 
                                       memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            
            if self.return_intermediate is not None:
                intermediate.append(self.norm(output_seq))
        # 2. 应用最终的归一化层layer norm
        if self.norm is not None:
            output_seq = self.norm(output_seq)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output_seq)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        # 3. 如果设置了返回中间结果，则将它们堆叠起来返回；否则，返回最终输出
        return output_seq.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)4
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    #Normalizing after, however, can often help maintain more effective gradient flow through the network
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        f = C2048, H/32, W/32
        z0 = d,HW
        src: [494, bs, 256]   [seq_len, batch_size, d_model] .  backbone输入下采样32倍后 再压缩维度到256的特征图
        src_mask: None，在Transformer中用来“防作弊”,即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        src_key_padding_mask: [bs, 494]  记录backbone生成的特征图中哪些是原始图像pad的部分 这部分是没有意义的
                           计算注意力会被填充为-inf，这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        pos: [494, bs, 256]  位置编码
        """

        #query,key embedding + position embedding  #[1064,2,256]
        query = key = self.with_pos_embed(src, pos)#[2, 256, 28,38]
        value = src


        # TODO: 实现 Transformer 编码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）

        
        # Ensure the attention mask has the correct shape
        

        #The shape of the 3D attn_mask is torch.Size([2, 28, 38]), but should be (16, 1064, 1064).
        attn_output  = self.self_attn(query, key, value, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)[0]        
        add_norm1_output = self.norm1(src + self.dropout1(attn_output))
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(add_norm1_output))))
        add_norm2_output = self.norm2(add_norm1_output + self.dropout2(ffn_output))
        return add_norm2_output
    


    '''
    Normalizing before processing helps control the range of inputs into each sub-layer, 
    potentially improving stability and training efficiency, especially in deeper networks
    '''
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        src: [494, bs, 256]  backbone输入下采样32倍后 再压缩维度到256的特征图
        src_mask: None，在Transformer中用来“防作弊”,即遮住当前预测位置之后的位置，忽略这些位置，不计算与其相关的注意力权重
        src_key_padding_mask: [bs, 494]  记录backbone生成的特征图中哪些是原始图像pad的部分 这部分是没有意义的
                           计算注意力会被填充为-inf，这样最终生成注意力经过softmax时输出就趋向于0，相当于忽略不计
        pos: [494, bs, 256]  位置编码
        """
        norm_src = self.norm1(src)
        query = key = self.with_pos_embed(norm_src, pos)
        value = norm_src
    
        attn_output   = self.self_attn(query, key, value, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)[0]        
        add1_output = src + self.dropout1(attn_output)

        norm2_output = self.norm2(add1_output)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(norm2_output))))
        add2_output = add1_output + self.dropout2(ffn_output)
        return add2_output


    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            # 先对输入进行LN
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        tgt: 需要预测的目标 query embedding，负责预测物体
        memory: [h*w, bs, 256]， Encoder的输出，具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # TODO: 实现 Transformer 解码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）


        query = key = self.with_pos_embed(tgt, query_pos)
        value = tgt
        
        attn_output   = self.self_attn(query, key, value, attn_mask = tgt_mask, 
                                       key_padding_mask = tgt_key_padding_mask)[0]        
        add_norm1_output = self.norm1(tgt + self.dropout(attn_output))

        query2 = self.with_pos_embed(add_norm1_output, query_pos)
        key2   = self.with_pos_embed(memory, pos) #memory is just the output of encoder 
        value2 = memory

        attn_output2   = self.multihead_attn(query2, key2, value2, attn_mask = memory_mask, 
                                             key_padding_mask = memory_key_padding_mask)[0]        
        add_norm2_output = self.norm2(add_norm1_output + self.dropout1(attn_output2))

        ffn_output2 = self.linear2(self.dropout2(self.activation(self.linear1(add_norm2_output))))
        add_norm3_output = self.norm3(add_norm2_output + self.dropout3(ffn_output2))


        return add_norm3_output
    
    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        tgt: 需要预测的目标 query embedding，负责预测物体
        memory: [h*w, bs, 256]， Encoder的输出，具有全局相关性（增强后）的特征表示
        tgt_mask: None
        memory_mask: None
        tgt_key_padding_mask: None
        memory_key_padding_mask: [bs, h*w]  记录Encoder输出特征图的每个位置是否是被pad的（True无效   False有效）
        pos: [h*w, bs, 256]  encoder输出特征图的位置编码
        query_pos: [100, bs, 256]  query embedding/tgt的位置编码  负责建模物体与物体之间的位置关系  随机初始化的
        tgt_mask、memory_mask、tgt_key_padding_mask是防止作弊的 这里都没有使用
        """
        # TODO: 实现 Transformer 解码器层的前向传播逻辑（参考DETR论文中Section A.3 & Fig.10）
        #Normalization
        norm_tgt = self.norm1(tgt)
        #word embedding + postion embedding
        query = key = self.with_pos_embed(norm_tgt, query_pos)
        value = norm_tgt
        
        #Multi-Head Self-attention
        attn_output   = self.self_attn(query, key, value, attn_mask = tgt_mask, 
                                       key_padding_mask = tgt_key_padding_mask)[0]      
        #Add   
        add1_output = tgt + self.dropout(attn_output)
        #Normaliztion
        norm2_output = self.norm2(add1_output)

        #decoder query embedding + decoder query position
        query2 = self.with_pos_embed(norm2_output, query_pos)
        #normalization
        norm_src_from_encoder = self.norm2(memory)

        #encoder embedding + encoder position
        key2   = self.with_pos_embed(memory, pos) #memory is just the output of encoder 
        value2 = norm_src_from_encoder

        #Multi-Head Cross Attention
        attn_output2   = self.multihead_attn(query2, key2, value2, attn_mask = memory_mask, 
                                             key_padding_mask = memory_key_padding_mask)[0]        
        add2_output = add1_output + self.dropout1(attn_output2)
        #nomaliztion
        add_norm2_output = self.norm2(add2_output)
        #feed forward network
        ffn_output2 = self.linear2(self.dropout2(self.activation(self.linear1(add_norm2_output))))

        add3_output = add2_output + self.dropout3(ffn_output2)

        return add3_output
        

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            # 先对输入进行LayerNorm
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
