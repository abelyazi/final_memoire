import tensorflow as tf


def conv_block(x, num_filters, kernel_size, MaxPool=True):
    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    if MaxPool:
        x = tf.keras.layers.MaxPool1D(2, padding="same")(x)
    
    return x

def bnrelu_model(input_shape):
    
    input_layer = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    x = conv_block(input_layer, 128, 3, MaxPool=False)
    x = conv_block(x, 128, 3, MaxPool=True)
    x = conv_block(x, 256, 3, MaxPool=True)
    x = conv_block(x, 256, 3, MaxPool=True)
    x = conv_block(x, 512, 3, MaxPool=True)
 
    u=64
    x = tf.keras.layers.LSTM(u,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.LSTM(u//2,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(u//4, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4/2)(x)

    x = tf.keras.layers.Dense(u//8, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3/3)(x)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

def bnrelu_model_156u(input_shape):
    
    input_layer = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    x = conv_block(input_layer, 128, 3, MaxPool=False)
    x = conv_block(x, 128, 3, MaxPool=True)
    x = conv_block(x, 256, 3, MaxPool=True)
    x = conv_block(x, 256, 3, MaxPool=True)
    x = conv_block(x, 512, 3, MaxPool=True)
 
    u=156
    x = tf.keras.layers.LSTM(u,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.LSTM(u//2,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(u//4, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4/2)(x)

    x = tf.keras.layers.Dense(u//8, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3/3)(x)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

def bnrelu_model_7CB(input_shape):
    
    input_layer = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    x = conv_block(input_layer, 128, 5, MaxPool=False)
    x = conv_block(x, 128, 5, MaxPool=True)
    x = conv_block(x, 256, 5, MaxPool=True)
    x = conv_block(x, 256, 5, MaxPool=True)
    x = conv_block(x, 256, 5, MaxPool=True)
    x = conv_block(x, 512, 5, MaxPool=True)
    x = conv_block(x, 512, 5, MaxPool=True)
 
    
    x = tf.keras.layers.LSTM(128,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.LSTM(64,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(64/2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4/2)(x)

    x = tf.keras.layers.Dense(32/2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3/3)(x)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

def bnrelu_model_3KS(input_shape):
    
    input_layer = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    x = conv_block(input_layer, 128, 3, MaxPool=False)
    x = conv_block(x, 128, 3, MaxPool=True)
    x = conv_block(x, 256, 3, MaxPool=True)
    x = conv_block(x, 256, 3, MaxPool=True)
    x = conv_block(x, 512, 3, MaxPool=True)
 
    
    x = tf.keras.layers.LSTM(128,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.LSTM(64,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(64/2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4/2)(x)

    x = tf.keras.layers.Dense(32/2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3/3)(x)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
def bnrelu_model_10KS(input_shape):
    
    input_layer = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    x = conv_block(input_layer, 128, 10, MaxPool=False)
    x = conv_block(x, 128, 10, MaxPool=True)
    x = conv_block(x, 256, 10, MaxPool=True)
    x = conv_block(x, 256, 10, MaxPool=True)
    x = conv_block(x, 512, 10, MaxPool=True)
 
    
    x = tf.keras.layers.LSTM(128,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.LSTM(64,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(64/2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4/2)(x)

    x = tf.keras.layers.Dense(32/2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3/3)(x)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
def bnrelu_model_3(input_shape):
    
    input_layer = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    
    x = conv_block(input_layer, 128, 3, MaxPool=False)
    x = conv_block(x, 128, 3, MaxPool=True)
    x = conv_block(x, 256, 3, MaxPool=True)
    x = conv_block(x, 256, 3, MaxPool=True)
    x = conv_block(x, 512, 3, MaxPool=True)
 
    
    x = tf.keras.layers.LSTM(128,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.LSTM(64,return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(64/2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4/2)(x)

    x = tf.keras.layers.Dense(32/2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3/3)(x)

    output_layer = tf.keras.layers.Dense(3, activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


def CNN1D_model(input_shape):
    inputs = tf.keras.layers.Input(shape=(input_shape[0],input_shape[1]))
    # ConvLayer 1
    conv1 = tf.keras.layers.Conv1D(filters=20, kernel_size=50,activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.MaxPool1D(2,padding="same")(conv1)
    
    # ConvLayer 2
    conv2 = tf.keras.layers.Conv1D(filters=20, kernel_size=50,activation='relu', padding='same')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.MaxPool1D(2,padding="same")(conv2)  
    
    # ConvLayer 3
    conv3 = tf.keras.layers.Conv1D(filters=20, kernel_size=50,activation='relu', padding='same')(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
     
    # ConvLayer 4
    conv4 = tf.keras.layers.Conv1D(filters=20, kernel_size=50,activation='relu', padding='same')(conv3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    
    
    x = tf.keras.layers.LSTM(64,return_sequences=True)(conv4)
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)    


    
    # Classification layers
    x = tf.keras.layers.Dense(30)(x)
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



