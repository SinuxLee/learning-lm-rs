use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    fn load_tensor(safetensor: &SafeTensors, key: &str)-> Tensor<f32> {
        let view = safetensor
            .tensor(key)
            .unwrap_or_else(|_| panic!("can't find tensor{}", key));

        let f32_data: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().expect("chunk too short");
                f32::from_le_bytes(bytes)
            })
            .collect();

        let shape = view.shape().to_vec();
        Tensor::new(f32_data, &shape)
    }
    // safetensors文件的模型参数加载
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        println!("Available tensors: {:?}", safetensor.names());

        // 当共享 embedding 时，safetensors 中只存储 lm_head.weight
        let embedding_table = if config.tie_word_embeddings {
            Self::load_tensor(safetensor,"lm_head.weight")
        } else {
           Self::load_tensor(safetensor,"model.embed_tokens.weight")
        };

        let n_layers = config.num_hidden_layers;
        let mut attn_norm_weights = Vec::with_capacity(n_layers);
        let mut q_proj_weights = Vec::with_capacity(n_layers);
        let mut k_proj_weights = Vec::with_capacity(n_layers);
        let mut v_proj_weights = Vec::with_capacity(n_layers);
        let mut o_proj_weights = Vec::with_capacity(n_layers);
        let mut ffn_norm_weights = Vec::with_capacity(n_layers);
        let mut up_proj_weights = Vec::with_capacity(n_layers);
        let mut gate_proj_weights = Vec::with_capacity(n_layers);
        let mut down_proj_weights = Vec::with_capacity(n_layers);

        // 加载每一层的参数
        for i in 0..n_layers {
            let base = format!("model.layers.{}", i);
            attn_norm_weights.push(Self::load_tensor(safetensor,&format!("{}.input_layernorm.weight", base)));
            q_proj_weights.push(Self::load_tensor(safetensor,&format!("{}.self_attn.q_proj.weight", base)));
            k_proj_weights.push(Self::load_tensor(safetensor,&format!("{}.self_attn.k_proj.weight", base)));
            v_proj_weights.push(Self::load_tensor(safetensor,&format!("{}.self_attn.v_proj.weight", base)));
            o_proj_weights.push(Self::load_tensor(safetensor,&format!("{}.self_attn.o_proj.weight", base)));

            ffn_norm_weights.push(Self::load_tensor(safetensor,&format!("{}.post_attention_layernorm.weight", base)));
            up_proj_weights.push(Self::load_tensor(safetensor,&format!("{}.mlp.up_proj.weight", base)));
            gate_proj_weights.push(Self::load_tensor(safetensor,&format!("{}.mlp.gate_proj.weight", base)));
            down_proj_weights.push(Self::load_tensor(safetensor,&format!("{}.mlp.down_proj.weight", base)));
        }

        let rms_out_w = Self::load_tensor(safetensor,"model.norm.weight");
        
        // 当共享 embedding 时，lm_head 与 embedding_table 数据相同
        let lm_head = if config.tie_word_embeddings {
            Tensor::new(embedding_table.data().to_vec(), &embedding_table.shape().to_vec())
        } else {
           Self::load_tensor(safetensor,"lm_head.weight")
        };

        LLamaParams {
            embedding_table,
            rms_att_w: attn_norm_weights, // 输入层归一化
            wq: q_proj_weights,
            wk: k_proj_weights,
            wv: v_proj_weights,
            wo: o_proj_weights,
            rms_ffn_w: ffn_norm_weights,
            w_up: up_proj_weights,
            w_gate: gate_proj_weights,
            w_down: down_proj_weights,
            rms_out_w,
            lm_head,
        }
    }
}
