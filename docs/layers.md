
## Transformer 架构

[推理过程](https://raw.githubusercontent.com/LearningInfiniTensor/handout/844a720d04a372e8dd72c3eb2583691e0e56b909/02-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E7%B3%BB%E7%BB%9F/LLaMA.drawio.svg)  
[更多算子](https://github.com/onnx/onnx/blob/main/docs/Operators.md)  

```
输入层:
    model.layers.0.input_layernorm.weight,F32,[128] # 初始归一化，稳定输入数据的分布
    model.layers.1.input_layernorm.weight,F32,[128]

隐藏层(自注意力 → 后归一化 → MLP):
    model.layers.0.self_attn.k_proj.weight,F32,[64, 128] # 自注意力 Key -> Query -> Value -> Output
    model.layers.0.self_attn.q_proj.weight,F32,[128, 128] 
    model.layers.0.self_attn.v_proj.weight,F32,[64, 128]
    model.layers.0.self_attn.o_proj.weight,F32,[128, 128]
    model.layers.0.post_attention_layernorm.weight,F32,[128]  # 后归一化
    model.layers.0.mlp.gate_proj.weight,F32,[384, 128] # 前馈网络（MLP）
    model.layers.0.mlp.up_proj.weight,F32,[384, 128]
    model.layers.0.mlp.down_proj.weight,F32,[128, 384]
    
    model.layers.1.self_attn.k_proj.weight,F32,[64, 128]
    model.layers.1.self_attn.q_proj.weight,F32,[128, 128] 
    model.layers.1.self_attn.v_proj.weight,F32,[64, 128]
    model.layers.1.self_attn.o_proj.weight,F32,[128, 128]
    model.layers.1.post_attention_layernorm.weight,F32,[128]
    model.layers.1.mlp.gate_proj.weight,F32,[384, 128]
    model.layers.1.mlp.up_proj.weight,F32,[384, 128]
    model.layers.1.mlp.down_proj.weight,F32,[128, 384]

输出层:
    model.norm.weight,F32,[128]  # 最终层归一化，稳定输出特征分布
    lm_head.weight,F32,[2048, 128] # 线性投影头，将128维特征映射到2048维输出空间（如词汇表）
```

-   model.layers.0：第 0 层 Transformer 解码器
-   self_attn：自注意力模块（包含 Q/K/V/O 投影）
-   mlp：前馈网络（包含 gate/up/down 投影）
-   input_layernorm / post_attention_layernorm：RMS 归一化层
-   lm_head：语言模型头（词表投影）
-   model.norm：输出层归一化
