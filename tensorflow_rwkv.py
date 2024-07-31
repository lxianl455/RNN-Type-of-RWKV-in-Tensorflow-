import tensorflow as tf
import numpy as np
import glob
import h5py
import sys
from light import light
from config.Config import Config
from config.VisionConfig import VisionConfig

class RWKVRNNCell(tf.keras.layers.Layer):
    def __init__(self, embed_dim = 0, hidden_dim = 0, reg_lambda = 0.0, **kwargs): 
        #reg_lambda (Optional[float]) – L2 regularization term on weights (xgb’s lambda).
        self.embed_dim = Config.TRUE_LSTM_UNIT_SIZE     
        self.hidden_dim =  Config.TRUE_LSTM_UNIT_SIZE
        self.reg_lambda = reg_lambda
        self.ln_1 = tf.keras.layers.LayerNormalization(name="ln_1")
        self.ln_2 = tf.keras.layers.LayerNormalization(name="ln_2")

        self.state_size = [self.embed_dim, self.embed_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim]
        self.output_size = self.embed_dim
        super().__init__(**kwargs)
    def build(self, input_shape):
        super().build(input_shape)
        self.rwkv_intermidiate_size = Config.RWKV_INTERMIDATE_SIZE
        # Time mixing - Mix parameters
        self.tm_mix_k = self.add_weight(
            shape=(self.embed_dim,),
            initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
            constraint=tf.keras.constraints.NonNeg(),
            name='tm_mix_k')
        self.tm_mix_v = self.add_weight(
            shape=(self.embed_dim,),
            initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
            constraint=tf.keras.constraints.NonNeg(),
            name='tm_mix_v')
        self.tm_mix_r = self.add_weight(
            shape=(self.embed_dim,),
            initializer=tf.keras.initializers.RandomUniform(0.0, (2./self.embed_dim)**0.5),
            constraint=tf.keras.constraints.NonNeg(),
            name='tm_mix_r')
        # Time mixing - KVR layer weights
        self.tm_key_weights = self.add_weight(
            shape=(self.embed_dim, self.hidden_dim),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.reg_lambda),
            name='tm_key_weights')
        self.tm_value_weights = self.add_weight(
            shape=(self.embed_dim, self.hidden_dim),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.reg_lambda),
            name='tm_value_weights')
        self.tm_receptance_weights = self.add_weight(
            shape=(self.embed_dim, self.hidden_dim),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.reg_lambda),
            name='tm_receptance_weights')
        # Time mixing - time_decay and time_first
        self.time_decay = self.add_weight(
            shape=(self.hidden_dim,),
            initializer='glorot_uniform',
            constraint=tf.keras.constraints.NonNeg(),
            name='time_decay')
        self.time_first = self.add_weight(
            shape=(self.hidden_dim,),
            initializer=tf.keras.initializers.RandomUniform(0.0, 0.05),
            name='time_first')
        # Time mixing - Output layer weights
        self.output_weights = self.add_weight(
            shape=(self.hidden_dim, self.embed_dim),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.reg_lambda),
            name='tm_output_weights')
        # Channel mixing - Mix parameters
        self.cm_mix_k = self.add_weight(
            shape=(self.embed_dim,),
            initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
            constraint=tf.keras.constraints.NonNeg(),
            name='cm_mix_k')
        self.cm_mix_r = self.add_weight(
            shape=(self.embed_dim,),
            initializer=tf.keras.initializers.RandomUniform(0.0, 2./self.embed_dim),
            constraint=tf.keras.constraints.NonNeg(),
            name='cm_mix_r')
        # Channel mixing - KR layer weights
        self.cm_key_weights = self.add_weight(
            shape=(self.embed_dim, self.rwkv_intermidiate_size*self.hidden_dim),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.reg_lambda),
            name='cm_key_weights')
        self.cm_value_weights = self.add_weight(
            shape=(self.rwkv_intermidiate_size*self.hidden_dim, self.embed_dim),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.reg_lambda),
            name='cm_value_weights')
        self.cm_receptance_weights = self.add_weight(
            shape=(self.embed_dim, self.embed_dim),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.reg_lambda),
            name='cm_receptance_weights')
        self.built = True

    def get_config(self):
        config = {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "reg_lambda": self.reg_lambda
        }
        base_config = super().get_config()
        config.update(base_config)
        return config
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        if batch_size is None or dtype is None:
            raise ValueError(
            "batch_size and dtype cannot be None while constructing initial "
            f"state. Received: batch_size={batch_size}, dtype={dtype}")
        def create_zeros(unnested_state_size):
            flat_dims = tf.TensorShape(unnested_state_size).as_list()
            init_state_size = [batch_size] + flat_dims
            return tf.zeros(init_state_size, dtype=dtype)
        if tf.nest.is_nested(self.state_size):
            return list(tf.nest.map_structure(create_zeros, self.state_size))
        else:
            return list(create_zeros(self.state_size))
        
    def time_mixing(self, inputs, prev_state):
        """Apply time mixing to inputs in RNN mode.
        Args:
            inputs: Expected shape (B, embed_dim)
            prev_state: 5-tuple (inputs_channel_mixing, inputs_time_mixing, num, den, q)
        Returns:
            rkv: Shape (B, embed_dim)
            new_state: 4-tuple (inputs_time_mixing, num, den, q)
        """
        # Mix x with the previous timestep to produce xk, xv, xr
        # batch_size, hidden_dim  = inputs.shape
        # prev_state = [torch.zeros( [ batch_size, hidden_dim]) for _ in range(5) ]
        prev_state_inputs = tf.expand_dims(prev_state[1], axis=1) # 取下来之后保持维度，方便后面concat

        prev_state_num = prev_state[2] # 这是以tuple的形式来保存的
        prev_state_den = prev_state[3] 
        prev_state_q = prev_state[4]
        # print("inputs.shape",inputs.shape)
        if inputs.shape[1] == 1: #推理时
            inputs_shifted = prev_state_inputs
        else: #训练时
            inputs_shifted = tf.concat([prev_state_inputs, inputs[:, :inputs.shape[1]-1, :]], axis=1)


        # xk = inputs * self.tm_mix_k + prev_state_inputs * (1 - self.tm_mix_k) # (B, embed_dim)
        # xv = inputs * self.tm_mix_v + prev_state_inputs * (1 - self.tm_mix_v) # (B, embed_dim)
        # xr = inputs * self.tm_mix_r + prev_state_inputs * (1 - self.tm_mix_r) # (B, embed_dim)

        #这里要有右移操作

        xk = inputs * self.tm_mix_k + inputs_shifted * (1 - self.tm_mix_k) # (B, embed_dim)
        xv = inputs * self.tm_mix_v + inputs_shifted * (1 - self.tm_mix_v) # (B, embed_dim)
        xr = inputs * self.tm_mix_r + inputs_shifted * (1 - self.tm_mix_r) # (B, embed_dim)



        # Learn key, value, and receptance from xk, xv, xr 
        key = xk @ self.tm_key_weights # (B, hidden_dim)
        value = xv @ self.tm_value_weights # (B, hidden_dim)
        receptance = xr @ self.tm_receptance_weights # (B, hidden_dim)
        # Apply activation function to r
        sigmoid_receptance = tf.keras.activations.sigmoid(receptance) # (B, hidden_dim)

        #############################################################
        # 开始计算RWKV
        output = [] #tf.zeros_like(key) # 先申请好空间，一个个存进来
        debug_mode = False

        time_decay = -tf.exp(self.time_decay)
        for current_index in range(inputs.shape[1] ): # 序列内循环Config.LSTM_TIME_STEPS
            current_key = tf.to_float(key[:, current_index])
            current_value = tf.to_float(value[:,current_index])


            # wkv computation at time t
            max_for_output = tf.math.maximum(prev_state_q, current_key + self.time_first)
            e1 = tf.exp(prev_state_q - max_for_output)
            e2 = tf.exp(current_key + self.time_first - max_for_output)
            numerator = e1 * prev_state_num + e2 * current_value
            denominator = e1 * prev_state_den + e2
            output .append( tf.cast(numerator / denominator, key.dtype) )

            # Update state for next iteration
            max_for_state = tf.math.maximum(prev_state_q + time_decay, current_key)
            e1 = tf.exp(prev_state_q + time_decay - max_for_state)
            e2 = tf.exp(current_key - max_for_state)
            prev_state_num = e1 * prev_state_num + e2 * current_value
            prev_state_den = e1 * prev_state_den + e2
            prev_state_q = max_for_state


        wkv = tf.stack(output, axis=1)

        ###############################################################################
        # Compute output
            #乘上前面的wkv结果
        x = (sigmoid_receptance * wkv) @ self.output_weights # (B, embed_dim)
        # 同理，这里也是用 最后一个token 和最后的 那些 num， den， 和q来组成新的状态，传递到下一次计算中
        return x, [inputs[:,-1], prev_state_num, prev_state_den, prev_state_q]
    
    def channel_mixing(self, inputs, prev_state):
        """Apply channel mixing to inputs in RNN mode.
        Args:
            inputs: Expected shape (B, embed_dim)
            prev_state: 5-tuple (inputs_channel_mixing, inputs_time_mixing, num, den, q)
        Returns:
            rkv: Shape (B, embed_dim)
            new_state: 1-tuple (inputs_channel_mixing)
        """
        # Mix x with the previous timestep to produce xk, xr

        # prev_state_inputs, _, _, _, _ = prev_state
        # prev_state_inputs = prev_state[:,[0],:] # [0]的写法能保持维度
        prev_state_inputs = tf.expand_dims(prev_state[0], axis=1) # 取下来之后保持维度，方便后面concat

        
        if  inputs.shape[1] == 1: #推理的时候，如果seq_length=1，那么就直接用之前的token，不用移动了
            inputs_shifted = prev_state_inputs
        else:
            #如果不是推理时，就要右移
            inputs_shifted = tf.concat([prev_state_inputs, inputs[:, :inputs.shape[1]-1, :]], axis=1)


        print("inputs_1",inputs_shifted.shape)
        xk = inputs * self.cm_mix_k + inputs_shifted * (1 - self.cm_mix_k) # (B, embed_dim)
        xr = inputs * self.cm_mix_r + inputs_shifted * (1 - self.cm_mix_r) # (B, embed_dim)
        # Compute k and r
        k = xk @ self.cm_key_weights # (B, 4*embed_dim)
        r = xr @ self.cm_receptance_weights # (B, embed_dim)

        # Compute rkv
        kv = tf.math.square(tf.nn.relu(k)) @ self.cm_value_weights # (B, embed_dim)
        # Compute rkv
        rkv = tf.math.sigmoid(r) * kv # (B, embed_dim)


        return rkv, [inputs[:,-1],] # 这是最后一个token的信息，拿来传递给下一次编码数据的时候，作为上一次的状态
    
    def call(self, inputs, prev_state=None):
        """Apply this layer to inputs in RNN mode.
        Args:
            inputs: Expected shape (B, embed_dim)
            prev_state: 5-tuple (inputs_channel_mixing, inputs_time_mixing, num, den, q)
        Returns:
            x: Shape (B, embed_dim)
            new_state: 5-tuple (inputs_channel_mixing, inputs_time_mixing, num, den, q)
        """
        x = inputs # (B, embed_dim)
        batch_size, sequence_len, hidden_dim = inputs.get_shape().as_list() # 这个要转化为 list，tf的变量是不能出来的


        x_tm, new_state_tm = self.time_mixing(self.ln_1(x), prev_state=prev_state)
        x = x + x_tm # (B, embed_dim)
        x_cm, new_state_cm = self.channel_mixing(self.ln_2(x), prev_state=prev_state)
        x = x + x_cm # (B, embed_dim)
        return x, new_state_cm + new_state_tm
    def compute_output_shape(self, input_shape):
        return input_shape

