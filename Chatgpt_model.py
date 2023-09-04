import tensorflow as tf



def initial_convolution(input_tensor):
    # Initial convolution and max pooling
    conv_layer = tf.keras.layers.Conv1D(filters=12, kernel_size=7, strides=2, padding='same')(input_tensor)
    maxpool_layer = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(conv_layer)
    
    return maxpool_layer

def composite_layer(input_tensor):
    # Composite layer within clique block
    expansion_ratio = 6
    if input_tensor.shape[-1]==None:
        intermediate_channels = expansion_ratio
    else:
        intermediate_channels = input_tensor.shape[-1] * expansion_ratio
    
    # Point-wise convolution for channel expansion
    bn = tf.keras.layers.BatchNormalization()(input_tensor)
    expanded_features = tf.keras.layers.Conv1D(filters=intermediate_channels, kernel_size=1, padding='same')(bn)
    
    # Depth-wise convolution
    expanded_features = tf.keras.layers.BatchNormalization()(expanded_features)
    expanded_features = tf.keras.layers.ReLU()(expanded_features)
    depthwise_conv = tf.keras.layers.DepthwiseConv1D(kernel_size=3, strides=1, padding='same')(expanded_features)

    
    # Point-wise convolution for channel reduction
    pconv = tf.keras.layers.BatchNormalization()(depthwise_conv)
    pconv = tf.keras.layers.ReLU()(pconv)
    bottleneck_features = tf.keras.layers.Conv1D(filters=12, kernel_size=1, padding='same')(pconv)
    #bottleneck_features = tf.keras.layers.LSTM(12,return_sequences=True)(bottleneck_features)
    
    return bottleneck_features

def clique_block(input_tensor):
    
    # Stage 1
    feat_0 = input_tensor
    feat_1 = composite_layer(feat_0)
    feat_0_1 = tf.keras.layers.concatenate([feat_0, feat_1], axis=-1)
    
    feat_2 = composite_layer(feat_0_1)
    feat_0_1_2 = tf.keras.layers.concatenate([feat_0, feat_1, feat_2], axis=-1)
    feat_3 = composite_layer(feat_0_1_2)
    feat_0_1_2_3 = tf.keras.layers.concatenate([feat_0, feat_1, feat_2, feat_3], axis=-1)
    feat_4 = composite_layer(feat_0_1_2_3)
    feat_0_1_2_3_4 = tf.keras.layers.concatenate([feat_0, feat_1, feat_2, feat_3, feat_4], axis=-1)
    feat_5 = composite_layer(feat_0_1_2_3_4)
    
    
    # Stage 2
    feat_3_4_5 = tf.keras.layers.concatenate([feat_3, feat_4, feat_5], axis=-1)
    feat_6 = composite_layer(feat_3_4_5)
    feat_4_5_6 = tf.keras.layers.concatenate([feat_4, feat_5, feat_6], axis=-1)
    feat_7 = composite_layer(feat_4_5_6)
    feat_5_6_7 = tf.keras.layers.concatenate([feat_5, feat_6, feat_7], axis=-1)
    feat_8 = composite_layer(feat_5_6_7)
    feat_6_7_8 = tf.keras.layers.concatenate([feat_6, feat_7, feat_8], axis=-1)
    feat_9 = composite_layer(feat_6_7_8)
    feat_7_8_9 = tf.keras.layers.concatenate([feat_7, feat_8, feat_9], axis=-1)
    feat_10 = composite_layer(feat_7_8_9)
    return feat_10

def scaled_dot_product(signal, weights, scale_factor):
    weights = tf.transpose(weights)
    dot_product = tf.tensordot(signal, weights, axes=1)  # Compute the dot product
    #scaled_dot_product = scale_factor * dot_product  # Scale the result
    return dot_product


def transition_block(input_tensor):
    # Transition block without dimension reduction
    bn = tf.keras.layers.BatchNormalization()(input_tensor)
    relu = tf.keras.layers.ReLU()(bn)
    if input_tensor.shape[-1]==None:
        compress_ratio = 6
    else:
        compress_ratio = input_tensor.shape[-1]//2
  
    compress_ratio = 12
    pconv = tf.keras.layers.Conv1D(compress_ratio, kernel_size=1, padding='same')(relu)
    
    
    glob_pool = tf.keras.layers.GlobalAveragePooling1D()(pconv)
    dense = tf.keras.layers.Dense(compress_ratio)(glob_pool)
    relu = tf.keras.layers.ReLU()(dense)
    dense = tf.keras.layers.Dense(compress_ratio)(relu)
    sigmoid = tf.keras.activations.sigmoid(dense)
    
    dot_product = tf.linalg.matvec(pconv, sigmoid)
    scale = tf.reshape(dot_product, (-1, pconv.shape[1], 1))
    avg_pool = tf.keras.layers.AveragePooling1D(pool_size=2)(scale)        
    return avg_pool


def transition_block_6(input_tensor):
    # Transition block without dimension reduction
    bn = tf.keras.layers.BatchNormalization()(input_tensor)
    relu = tf.keras.layers.ReLU()(bn)
    if input_tensor.shape[-1]==None:
        compress_ratio = 6
    else:
        compress_ratio = input_tensor.shape[-1]//2
  
    compress_ratio = 6
    pconv = tf.keras.layers.Conv1D(compress_ratio, kernel_size=1, padding='same')(relu)
    
    
    glob_pool = tf.keras.layers.GlobalAveragePooling1D()(pconv)
    dense = tf.keras.layers.Dense(compress_ratio)(glob_pool)
    relu = tf.keras.layers.ReLU()(dense)
    dense = tf.keras.layers.Dense(compress_ratio)(relu)
    sigmoid = tf.keras.activations.sigmoid(dense)
    
    dot_product = tf.linalg.matvec(pconv, sigmoid)
    scale = tf.reshape(dot_product, (-1, pconv.shape[1], 1))
    avg_pool = tf.keras.layers.AveragePooling1D(pool_size=2)(scale)        
    return avg_pool

def create_clique_LSTM_model(input_shape):
    inputs = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    # Initial convolution and max pooling
    maxpool_layer = initial_convolution(inputs)
    
    # Clique blocks
    clique_block1 = clique_block(maxpool_layer)
    merge_1 = tf.keras.layers.concatenate([maxpool_layer, clique_block1], axis=-1)
    bn = tf.keras.layers.BatchNormalization()(merge_1)
    relu = tf.keras.layers.ReLU()(bn)
    x = tf.keras.layers.MaxPool1D(2,padding="same")(relu)
    
    x = tf.keras.layers.LSTM(128,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.LSTM(64,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    
    # Squeezed multi-scale representation
    pooled_1 = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    
    # Classification layers
    x = tf.keras.layers.Dense(32)(pooled_1)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
 
    #x = tf.keras.layers.Dense(8)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.1)(x)
    
    
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_small_model(input_shape):
    inputs = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    # Initial convolution and max pooling
    maxpool_layer = initial_convolution(inputs)
    
    # Clique blocks
    clique_block1 = clique_block(maxpool_layer)
    transition_block1 = transition_block(clique_block1)
    clique_block2 = clique_block(transition_block1)
    transition_block2 = transition_block(clique_block2)
    clique_block3 = clique_block(transition_block2)
  

    merge_1 = tf.keras.layers.concatenate([maxpool_layer, clique_block1], axis=-1)
    #merge_1 = tf.keras.layers.LSTM(30,return_sequences=True)(merge_1)
    merge_2 = tf.keras.layers.concatenate([transition_block1, clique_block2], axis=-1)
    #merge_2 = tf.keras.layers.LSTM(30,return_sequences=True)(merge_2)
    merge_3 = tf.keras.layers.concatenate([transition_block2, clique_block3], axis=-1)
    #merge_3 = tf.keras.layers.LSTM(30,return_sequences=True)(merge_3)
    
    # Squeezed multi-scale representation
    pooled_1 = tf.keras.layers.GlobalAveragePooling1D()(merge_1)
    pooled_2 = tf.keras.layers.GlobalAveragePooling1D()(merge_2)
    pooled_3 = tf.keras.layers.GlobalAveragePooling1D()(merge_3)
    
    merge_pool_2_3 = tf.keras.layers.concatenate([pooled_2, pooled_3], axis=-1)
    merge_pool_1_23 = tf.keras.layers.concatenate([pooled_1, merge_pool_2_3], axis=-1)
    

    
    # Classification layers
    x = tf.keras.layers.Dense(30)(merge_pool_1_23)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Dense(15)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
 
    #x = tf.keras.layers.Dense(8)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.1)(x)
    
    
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_small_model_6(input_shape):
    inputs = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    # Initial convolution and max pooling
    maxpool_layer = initial_convolution(inputs)
    
    # Clique blocks
    clique_block1 = clique_block(maxpool_layer)
    transition_block1 = transition_block_6(clique_block1)
    clique_block2 = clique_block(transition_block1)
    transition_block2 = transition_block_6(clique_block2)
    clique_block3 = clique_block(transition_block2)
  

    merge_1 = tf.keras.layers.concatenate([maxpool_layer, clique_block1], axis=-1)
    #merge_1 = tf.keras.layers.LSTM(30,return_sequences=True)(merge_1)
    merge_2 = tf.keras.layers.concatenate([transition_block1, clique_block2], axis=-1)
    #merge_2 = tf.keras.layers.LSTM(30,return_sequences=True)(merge_2)
    merge_3 = tf.keras.layers.concatenate([transition_block2, clique_block3], axis=-1)
    #merge_3 = tf.keras.layers.LSTM(30,return_sequences=True)(merge_3)
    
    # Squeezed multi-scale representation
    pooled_1 = tf.keras.layers.GlobalAveragePooling1D()(merge_1)
    pooled_2 = tf.keras.layers.GlobalAveragePooling1D()(merge_2)
    pooled_3 = tf.keras.layers.GlobalAveragePooling1D()(merge_3)
    
    merge_pool_2_3 = tf.keras.layers.concatenate([pooled_2, pooled_3], axis=-1)
    merge_pool_1_23 = tf.keras.layers.concatenate([pooled_1, merge_pool_2_3], axis=-1)
    

    
    # Classification layers
    x = tf.keras.layers.Dense(30)(merge_pool_1_23)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Dense(15)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
 
    #x = tf.keras.layers.Dense(8)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.1)(x)
    
    
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
