# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization

import keras



def get_model(nii_size):

    model = Sequential()

    kernel_num = 3
    cnn_strides_num = 1
    
    pool_num = 3
    pool_strides_num = 3


    
    model.add( Conv3D( 32, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', input_shape = ( nii_size[0], nii_size[1], nii_size[2], 1 ) ) )
    model.add( Conv3D( 32, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform' ) )
    model.add( Conv3D( 32, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform' ) )
    
    model.add( MaxPooling3D( pool_size = (pool_num, pool_num, pool_num), strides = (pool_strides_num, pool_strides_num, pool_strides_num) ) )
    model.add( BatchNormalization() ) 


    
    model.add( Conv3D( 64, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform' ) )
    model.add( Conv3D( 64, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform' ) )
    model.add( Conv3D( 64, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform' ) )
    
    model.add( MaxPooling3D( pool_size = (pool_num, pool_num, pool_num), strides = (pool_strides_num, pool_strides_num, pool_strides_num) ) )
    model.add( BatchNormalization() )



    model.add( Conv3D( 128, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform' ) )
    model.add( Conv3D( 128, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform' ) )
    model.add( Conv3D( 128, kernel_size = (kernel_num, kernel_num, kernel_num), strides = (cnn_strides_num, cnn_strides_num, cnn_strides_num),
                       activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform' ) )
    
    model.add( MaxPooling3D( pool_size = (pool_num, pool_num, pool_num), strides = (pool_strides_num, pool_strides_num, pool_strides_num) ) )
    model.add( BatchNormalization() )


    
    model.add( Flatten() )
    
    model.add( Dense( 2048, activation = 'relu', kernel_initializer = 'he_uniform' ) )
    model.add( Dropout(0.5) )
    
    model.add( Dense( 2048, activation = 'relu', kernel_initializer = 'he_uniform' ) )
    model.add( Dropout(0.5) )

    model.add( Dense( 1000, activation = 'relu', kernel_initializer = 'he_uniform' ) )
    model.add( Dense( 2, activation = 'softmax') )



    return model



##############################
#
# 检修区
#
##############################

if __name__ == '__main__':

    # 放入需要检查的函数并进行检查
    pass
