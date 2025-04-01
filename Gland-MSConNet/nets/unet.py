from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import (Activation, Add, Conv1D, Conv2D, 
                          GlobalAveragePooling2D,
                          Reshape, multiply)
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, concatenate
from tensorflow.keras.layers import (AveragePooling2D,DepthwiseConv2D)
import math

from tensorflow.keras.layers import DepthwiseConv2D, Conv2DTranspose
from tensorflow.keras import layers, models

def subpixel_conv2d(inputs, filters, upscale_factor):
    """
    Implements subpixel convolution for upsampling.
    Args:
        inputs: Input tensor.
        filters: Number of filters in the Conv2D layer.
        upscale_factor: Factor to upscale the spatial resolution.
    Returns:
        Upsampled tensor.
    """
    x = layers.Conv2D(filters * (upscale_factor ** 2), (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal')(inputs)
    x = tf.nn.depth_to_space(x, upscale_factor)
    return x


class ResidualConnectionModule(layers.Layer):
    def __init__(self, module, module_factor=1.0, **kwargs):
        super(ResidualConnectionModule, self).__init__(**kwargs)
        self.module = module
        self.module_factor = module_factor

        # 将Conv2D 层移到 __init__ 方法，确保它只在初始化时创建一次
        self.conv_layer = None

    def build(self, input_shape):
        # 在 build 方法中创建卷积层，确保只创建一次
        if self.conv_layer is None:
            self.conv_layer = layers.Conv2D(input_shape[-1], (1, 1))  # 根据输入通道数创建 1x1 卷积层
    @tf.function
    def call(self, inputs):
        module_output = self.module(inputs)
        # 确保卷积层在每次调用时都使用输入的通道数
        module_output = self.conv_layer(module_output)
        return inputs + module_output * self.module_factor


class FeedForwardModule(layers.Layer):
    def __init__(self, encoder_dim, expansion_factor, dropout_p=0.1, **kwargs):
        super(FeedForwardModule, self).__init__(**kwargs)
        self.fc1 = layers.Dense(encoder_dim * expansion_factor)
        self.dropout = layers.Dropout(dropout_p)
        self.fc2 = layers.Dense(encoder_dim)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = tf.nn.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ConformerConvModule(layers.Layer):
    def __init__(self, in_channels, kernel_size, expansion_factor, dropout_p=0.1, **kwargs):
        super(ConformerConvModule, self).__init__(**kwargs)
        self.conv = layers.Conv2D(in_channels * expansion_factor, kernel_size, padding="same")
        self.dropout = layers.Dropout(dropout_p)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.dropout(x)
        return x


class ConbimambaBlock(layers.Layer):
    def __init__(
        self,
        encoder_dim=512,
        num_attention_heads=8,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        feed_forward_dropout_p=0.1,
        attention_dropout_p=0.1,
        conv_dropout_p=0.1,
        conv_kernel_size=31,
        half_step_residual=True,
        **kwargs
    ):
        super(ConbimambaBlock, self).__init__(**kwargs)

        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1.0

        # 定义第一个FeedForward
        self.ResidualConn_A = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )

        # 外部双向Mamba (此处作为占位符，可以根据实际情况替换为对应的模块)
        self.ResidualConn_B = ResidualConnectionModule(
            module=layers.Dense(encoder_dim),  # 示例，替换为实际的 ExBimamba 模块
        )

        # 定义convolution层
        self.ResidualConn_C = ResidualConnectionModule(
            module=ConformerConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
            )
        )

        # 定义第二个FeedForward
        self.ResidualConn_D = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )

        # 正则化
        self.norm = layers.LayerNormalization()

    def call(self, inputs):
        x1 = self.ResidualConn_A(inputs)  # 执行第一个Feed-Forward
        x2 = self.ResidualConn_B(x1)  # 执行ExBimamba (外部双向Mamba)
        x3 = self.ResidualConn_C(x2)  # 执行Conformer的Convolution
        x4 = self.ResidualConn_D(x3)  # 执行第二个Feed-Forward
        out = self.norm(x4)  # 正则化
        return out



# 通道注意力模块 (Channel Attention Module)
class ChannelAttentionModule(tf.keras.layers.Layer):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()
        self.fc1 = layers.Dense(in_channels // 8, activation='relu')
        self.fc2 = layers.Dense(in_channels, activation='sigmoid')

    def call(self, x):
        avg_out = self.global_avg_pool(x)  # (B, C)
        max_out = self.global_max_pool(x)  # (B, C)
        
        avg_out = self.fc1(avg_out)
        max_out = self.fc1(max_out)
        
        avg_out = self.fc2(avg_out)
        max_out = self.fc2(max_out)
        
        out = avg_out + max_out  # element-wise summation
        out = tf.reshape(out, (-1, 1, 1, out.shape[1]))  # Reshape to (B, 1, 1, C)
        return x * out  # Apply the attention

# 空间注意力模块 (Spatial Attention Module)
class SpatialAttentionModule(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B, H, W, 1)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)  # (B, H, W, 1)
        concat_out = tf.concat([avg_out, max_out], axis=-1)  # (B, H, W, 2)
        attention_map = self.conv1(concat_out)  # (B, H, W, 1)
        return x * attention_map  # Apply the spatial attention




class FusionConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = layers.Conv2D(dim, (1, 1), strides=1)
        self.conv_3x3 = layers.Conv2D(dim, (3, 3), padding='same')
        #self.conv_5x5 = layers.Conv2D(dim, (5, 5), padding='same')
        #self.conv_7x7 = layers.Conv2D(dim, (7, 7), padding='same')
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = layers.Conv2D(out_channels, (1, 1), strides=1)
        #self.down_2 = layers.Conv2D(dim, (1, 1), strides=1)

    def call(self, x1, x2, x4):
        x_fused = tf.concat([x1, x2, x4], axis=3)  # (B, C, H, W)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        #x_5x5 = self.conv_5x5(x_fused)
        #x_7x7 = self.conv_7x7(x_fused)
        #x_fused_s = x_3x3 + x_5x5 + x_7x7
        #x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        x_fused_s = x_3x3 * self.spatial_attention(x_3x3)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out





class MSAA(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def call(self, x1, x2, x4, ):
        x_fused = self.fusion_conv(x1, x2 ,x4)
        return x_fused

def eca_block(input_feature, b=1, gamma=2, name=""):
	channel = K.int_shape(input_feature)[-1]
	kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
	kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
	avg_pool = GlobalAveragePooling2D()(input_feature)
	x = Reshape((-1,1))(avg_pool)
	x = Conv1D(1, kernel_size=kernel_size, padding="same", name = "eca_layer_"+str(name), use_bias=False,)(x)
	x = Activation('sigmoid')(x)
	x = Reshape((1, 1, -1))(x)
	output = multiply([input_feature,x])
	return output

def Unet(input_shape=(256,256,3), num_classes=3, backbone = "vgg"):
    inputs = Input(input_shape)
    
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

    feat1 = ConbimambaBlock(encoder_dim=32, num_attention_heads=8,feed_forward_expansion_factor=2,conv_expansion_factor=2,feed_forward_dropout_p=0.1,attention_dropout_p=0.1,conv_dropout_p=0.1,conv_kernel_size=3,half_step_residual=True)(conv1)

    up1 = subpixel_conv2d(feat1, filters=16, upscale_factor=2)

    conv2 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    conv2 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
   
  
    feat2 = eca_block(conv2,name="eac2")

    up2 = subpixel_conv2d(feat2, filters=16, upscale_factor=2)
    
    conv3 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    conv3 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
   
    
    feat3 = eca_block(conv3,name="eac3")
    pool1 = MaxPooling2D(pool_size=(2, 2))(feat3)
    merge7 = concatenate([feat2, pool1], axis=3)


    conv4 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv4 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    
    feat4 = eca_block(conv4,name="eac4")

    pool2 = MaxPooling2D(pool_size=(2, 2))(feat4)
    merge8 = concatenate([feat1, pool2], axis=3)
   
    conv5 = layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv5 = layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    feat5 = eca_block(conv5,name="eac5")
    merge9 = MSAA(in_channels=feat1.shape[-1], out_channels=feat1.shape[-1])(feat1, feat5, feat5)
  

    conv9 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    feat6 = eca_block(conv10,name="eac6")
    conv11 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(feat6)

    if backbone == "vgg":
        # 512, 512, 64 -> 512, 512, num_classes
        P1 = Conv2D(num_classes, 1, activation="softmax")(conv11)

    else:
        raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        
    model = Model(inputs=inputs, outputs=P1)
    return model