# -*- coding:utf-8 -*-

def train(model_name, tissue_name):

    # 不能使用垃圾回收系统，这样会变得non-picklable
    # 从而影响 keras 的 generator thread safe
    
    # 垃圾回收系统
    # import gc
    # gc.enable()

    # 清空之前的命令行
    import os
    # os.system('clear')

    # 不显示提示信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import logging
    logging.getLogger('tensorflow').disabled = True

    ####################
    # 双GPU
    ####################

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


    """
    # 避免独占
    
    import tensorflow as tf

    # 设置GPU使用方式
    # 获取GPU列表
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU为增长式占用
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True) 
        except RuntimeError as e:
            # 打印异常
            print(e)
    """


    ####################
    # 读取全部数据
    ####################

    import pandas as pd

    # 这里修改 label 的名字
    # model_name = 'stb_c1'

    metadata = pd.read_csv( tissue_name + '.csv', # 'metadata.csv',
                            usecols = ['doc_name', model_name])

    metadata = metadata.values
    # print(metadata[:3])

    ####################
    # 混洗数据
    ####################

    import numpy as np

    # 设个种子，保持原样
    np.random.seed(42)
    np.random.shuffle(metadata)

    # print(metadata[:3])

    ####################
    # nii参数
    ####################

    nii_size = (91, 109, 91)

    ####################
    # 归一化
    ####################

    import numpy as np

    def normalization(x):

        ####################
        # 移除过大过小值
        ####################

        # 从0开始到第一个计数小于 500 的

        his_data = np.histogram( x.flatten(), bins = 100 )

        # print(his_data)

        cut_index = 0

        for i in range(len(his_data[0])):

            if his_data[0][-i] > 500:
                cut_index = -i
                break

        cut_min = 0
        cut_max = his_data[1][cut_index]

        # print()
        # print(cut_max)
        # print()

        x = x.clip( cut_min, cut_max )

        """
        # 使用均值方差归一化
        mu = np.average(x)
        sigma = np.std(x)
        x = ( x - mu ) / sigma
        """
        # 最大最小值归一化
        data_min = np.min(x)
        data_max = np.max(x)
        x = ( x - data_min ) / ( data_max - data_min )
        
        return x

    ####################
    # 读取nii文件函数
    ####################

    # pip install nibabel 
    import nibabel as nib

    def read_nii(name, counter):

        img = nib.load( 'abcd_data/' + name )
        img_array = img.get_fdata()


        # 进行 1/2 的缩放
        import scipy.ndimage
        
        output = scipy.ndimage.interpolation.zoom(
            input = img_array,
            zoom = 0.5,
            order = 3
            )


        output = normalization( output )


        # print()
        """
        print(counter, end='\t')

        print('name: ', end='')
        print(name, end='\t')

        print('shape: ', end='')
        print(output.shape, end='\t')
        print('max: ', end='')
        print(output.max(), end='\t')
        print('min: ', end='')
        print(output.min())
        """
        # print(counter, end=' ')

        # gc.collect() # 回收三代垃圾

        return output

    ########################
    # 读取nii文件，generator
    ########################

    print('all data len(): ', end='')
    print(len(metadata))
    print()

    batch_size = 10
    # batch_size = 4 # 这里弄小一点，测试显存

    train_size = int( len(metadata)* 4/6 )
    # train_size = 100 # 这里弄小一点，方便测试
    train_steps = train_size // batch_size

    # val_size = int( len(metadata)* 1/6 )
    val_size = 0 # 不设置 val 了
    # val_size = 32 # 这里弄小一点，方便测试
    val_steps = val_size // batch_size



    from keras.utils import Sequence
    from keras.utils.np_utils import to_categorical

    # 本方法线程安全，不会内存泄漏

    class train_gen(Sequence):

        def __init__(self):
            pass

        def __len__(self):
            return train_steps

        def __getitem__(self, idx):

            x_train = np.empty( [ batch_size, nii_size[0], nii_size[1], nii_size[2], 1 ], dtype = float) 
            y_train = []

            for j in range(batch_size):

                x_train[ j, : , : , : , 0 ] = read_nii( metadata[ idx * batch_size + j ][0], idx * batch_size + j )
                y_train.append( metadata[ idx * batch_size + j ][1] )

            # print('x_train.shape: ', end='')
            # print(x_train.shape, end='\t')
                
            y_train = to_categorical(y_train, 2)
                
            # print('y_train.shape: ', end='')
            # print(y_train.shape)

            # gc.collect() # 回收三代垃圾
                    
            return x_train, y_train



    class val_gen(Sequence):

        def __init__(self):
            pass

        def __len__(self):
            return val_steps

        def __getitem__(self, idx):

            idx += train_steps

            x_val = np.empty( [ batch_size, nii_size[0], nii_size[1], nii_size[2], 1 ], dtype = float) 
            y_val = []

            for j in range(batch_size):

                x_val[ j, : , : , : , 0 ] = read_nii( metadata[ idx * batch_size + j ][0], idx * batch_size + j )
                y_val.append( metadata[ idx * batch_size + j ][1] )

            # print('x_val.shape: ', end='')
            # print(x_val.shape, end='\t')
                
            y_val = to_categorical(y_val, 2)
                
            # print('y_val.shape: ', end='')
            # print(y_val.shape)

            # gc.collect() # 回收三代垃圾
                    
            return x_val, y_val



    ####################
    # 绘图
    ####################

    # 因为服务器没有图形界面，所以必须这样弄
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams['savefig.dpi'] = 500 # 图片像素

    import keras

    class PlotProgress(keras.callbacks.Callback):


        def __init__(self, entity = ['loss', 'accuracy', 'auc_roc']):
            
            self.entity = entity


        def on_train_begin(self, logs={}):
            
            self.i = 0
            self.x = []
            
            self.losses = []
            self.val_losses = []

            self.accs = []
            self.val_accs = []

            self.auc_rocs = []
            self.val_auc_rocs = []

            # gc.collect() # 回收三代垃圾


        def on_epoch_end(self, epoch, logs={}):
            
            self.x.append(self.i)
            # 损失函数
            self.losses.append(logs.get('{}'.format(self.entity[0])))
            # self.val_losses.append(logs.get('val_{}'.format(self.entity[0])))
            # 准确率
            self.accs.append(logs.get('{}'.format(self.entity[1])))
            # self.val_accs.append(logs.get('val_{}'.format(self.entity[1])))
            # auc_roc
            # self.auc_rocs.append(logs.get('{}'.format(self.entity[2])))
            # self.val_auc_rocs.append(logs.get('val_{}'.format(self.entity[2])))

            self.i += 1


            plt.figure(0)
            plt.clf() # 清理历史遗迹
            plt.plot(self.x, self.losses, label='{}'.format(self.entity[0]))
            # plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity[0]))
            plt.legend()
            plt.savefig( 'train/' + model_name + '_' + tissue_name + '_loss.png' )

            plt.figure(1)
            plt.clf() # 清理历史遗迹
            plt.plot(self.x, self.accs, label='{}'.format(self.entity[1]))
            # plt.plot(self.x, self.val_accs, label="val_{}".format(self.entity[1]))
            plt.legend()
            plt.savefig( 'train/' + model_name + '_' + tissue_name + '_acc.png' )
            """
            plt.figure(2)
            plt.clf() # 清理历史遗迹
            plt.plot(self.x, self.auc_rocs, label='{}'.format(self.entity[2]))
            plt.plot(self.x, self.val_auc_rocs, label="val_{}".format(self.entity[2]))
            plt.legend()
            plt.savefig( 'train/' + model_name + '_auc_roc.png' )
            """
            # gc.collect() # 回收三代垃圾



    ####################
    # auc早产
    ####################
    """
    from sklearn.metrics import roc_auc_score

    # 不平衡问题真的很多

    def auc_roc(y_true, y_pred):

        try:
            roc_auc_score(y_true, y_pred) # 要是成，就最后返回
        
        except:
            return 0 # 要是不成，就直接结束

        return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
    """
    ####################
    # GPU并行
    ####################

    import tensorflow as tf

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

        ####################
        # 模型
        ####################

        import abcd_model

        parallel_model = abcd_model.get_model(nii_size)

        parallel_model.compile( loss = 'categorical_crossentropy',
                                optimizer = keras.optimizers.Adam( lr = 1e-4 ),
                                metrics = ['accuracy'] ) # , auc_roc] )

        ####################
        # 打印模型
        ####################

        parallel_model.summary()



    # pip install pydot
    from keras.utils import plot_model
    plot_model(parallel_model, to_file = 'abcd_model.png')



    # 早产
    # from keras.callbacks import EarlyStopping
    # early_stopping = EarlyStopping(monitor = 'val_auc_roc', patience = 10, restore_best_weights = True)

    # 绘图函数
    plot_progress = PlotProgress(entity = ['loss', 'accuracy'] ) # , 'auc_roc'])

    # gc.collect() # 回收三代垃圾



    # 在 use_multiprocessing 模式下，很容易内存泄漏
    # workers 的数量必须小于（ 应该也可以 等于 ? ） max_queue_size
    # 最好是 max_queue_size 等于 2倍 的 workers
    # 这样每次内存都能充分释放

    # workers 应小于等于核心数量

    ####################
    # 0 1 权重配比
    ####################

    zeros = 0

    for sid in range(train_size):

        if metadata[sid][1] == 1: # 原先是 metadata[sid][1] 可能弄反了（划掉） 之前的是对的
            zeros += 1

    inbalance = 3 / 4 # 不平衡度，用来避免对于 1 的 overfitting

    cw = { 0: zeros / train_size + ( train_size - zeros ) / train_size * (1 - inbalance), 1: ( train_size - zeros ) / train_size * inbalance }

    print()
    print('class_weight')
    print(cw)
    print()

    ####################
    # 保存每一个模型
    ####################

    from keras.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint( 'model/' + model_name + '_' + tissue_name + '_{epoch:02d}.h5',
                                  verbose = 1, # 使用详细信息模式
                                  save_weights_only = True
                                  )
    print('-> ModelCheckpoint')
    print('save_freq = epoch,')
    print('save_weights_only = True,')
    print('filepath = model/' + model_name + '_' + tissue_name + '_{epoch:02d}.h5')
    print()

    ####################
    # 训练
    ####################

    history = parallel_model.fit_generator( generator = train_gen(),
                                            steps_per_epoch = train_steps,

                                            epochs = 10,
                                            
                                            verbose = 2, # 1 显示进度条 2 只显示结果

                                            # validation_data = val_gen(),
                                            # validation_steps = val_steps,

                                            workers = 80, # 最大进程数，跑满
                                            use_multiprocessing = True, # 多线程

                                            callbacks = [ plot_progress, checkpoint ], # , early_stopping], # 使用 auc_roc 进行评估
                       
                                            max_queue_size = 40, # precache 内存大就可以多弄

                                            shuffle = True, # 再次打乱

                                            class_weight = cw, # 类间权重调整
                                            
                                            )

    ####################
    # 保存
    ####################
     
    with open( 'train/' + model_name + '_' + tissue_name + '_train.log', 'w' ) as f: # 打开文件
        print( history.history, file = f ) # 把历史写入txt文件，方便检查
