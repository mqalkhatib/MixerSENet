from tensorflow import keras
from keras import layers


def SE(xin, se_ratio=8):
    """
   Squeeze-and-Excitation (SE).
   Args:
       xin: Input tensor with shape (batch_size, height, width, channels).
       se_ratio: Reduction ratio for the squeeze operation (default is 8).
       
   Returns:
       output: Tensor with channel attention applied. The output shape is the same as the input shape.
   """
    # Global Average Pooling along spatial dimensions
    xin_gap = layers.GlobalAveragePooling2D()(xin)
    
    # Squeeze Path
    sqz = layers.Dense(xin.shape[-1] // se_ratio, activation='relu')(xin_gap)
    
    # Excitation Path
    excite = layers.Dense(xin.shape[-1], activation='sigmoid')(sqz)
    
    # Multiply the input by the excitation weights
    out = layers.multiply([xin, layers.Reshape((1, 1, xin.shape[-1]))(excite)])

    return out


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    residuals = x
    
    pos_emb1 = layers.DepthwiseConv2D(kernel_size=3, padding="same")(x)
    pos_emb2 = layers.DepthwiseConv2D(kernel_size=5, padding="same")(x)
    pos_emb3 = layers.DepthwiseConv2D(kernel_size=7, padding="same")(x)
    
    x = keras.layers.Add()([residuals, pos_emb1, pos_emb2, pos_emb3])

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)
    x = layers.Add()([x, residuals])

    return x


def MixerSENet(img_list, NumClasses, depth, filters):
    inputs = keras.Input(img_list.shape[1:])
    
    x = layers.Conv2D(filters, kernel_size=1, strides=1)(inputs)
    x = activation_block(x)  

    # Apply the mixer blocks
    for ii in range(depth):
        x = mixer_block(x, filters, 3)

    
    x = SE(x)   
    
    # Global Pooling and Final Classification Block
    x = layers.GlobalAvgPool2D()(x)  # Global pooling in 2D
    logits = layers.Dense(NumClasses , activation="softmax")(x)

    model = keras.Model(inputs=[inputs], outputs=logits)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
