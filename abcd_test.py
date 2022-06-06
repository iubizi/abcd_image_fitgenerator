# -*- coding:utf-8 -*-

def test(model_name, tissue_name):

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
    # 单GPU
    ####################

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
                            usecols = [ 'doc_name', model_name ]
                            )

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

    nii_size = ( 91, 109, 91 )

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

    # val 做掉了，暂时不用

    # train_val_size = int( len(metadata)* 5/6 )
    train_val_size = int( len(metadata)* 4/6 )
    # train_val_size = 100 # 这里弄小一点，方便测试
    train_val_steps = train_val_size // batch_size

    # 这里不能用，需要偏置
    # test_size = int( len(metadata)* 1/6 )
    test_size = int( len(metadata)* 2/6 )
    test_steps = test_size // batch_size



    from keras.utils import Sequence
    from keras.utils.np_utils import to_categorical

    # 本方法线程安全，不会内存泄漏
    
    class test_gen(Sequence):

        def __init__(self):
            pass

        def __len__(self):
            return test_steps

        def __getitem__(self, idx):

            idx += train_val_steps

            x_test = np.empty( [ batch_size, nii_size[0], nii_size[1], nii_size[2], 1 ], dtype = float) 
            y_test = []

            for j in range(batch_size):

                x_test[ j, : , : , : , 0 ] = read_nii( metadata[ idx * batch_size + j ][0], idx * batch_size + j )
                y_test.append( metadata[ idx * batch_size + j ][1] )

            # print('x_test.shape: ', end='')
            # print(x_test.shape, end='\t')
                
            y_test = to_categorical(y_test, 2)
                
            # print('y_test.shape: ', end='')
            # print(y_test.shape)

            # gc.collect() # 回收三代垃圾
                    
            return x_test, y_test



    ####################
    # 绘图
    ####################

    # 因为服务器没有图形界面，所以必须这样弄
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams['savefig.dpi'] = 500 # 图片像素



    ####################
    # 同一模型多epoch
    ####################

    def multi_models(model_name, tissue_name):

        ####################
        # GPU单卡运行
        ####################

        ####################
        # 模型
        ####################

        import abcd_model

        clf = abcd_model.get_model(nii_size)

        import keras

        clf.compile( loss = 'categorical_crossentropy',
                     optimizer = keras.optimizers.Adam( lr = 1e-4 ),
                     metrics = ['accuracy'] )

        ####################
        # 加载weights
        ####################

        clf.load_weights( 'model/' + model_name + '_' + tissue_name + '.h5' )

        print( '#' * 40 )
        print( 'load_weights: model/' + model_name + '_' + tissue_name + '.h5' )


        


        ####################
        # 模型输出
        ####################

        # 得到预测结果( 0 & 1 )
        score = clf.evaluate_generator( generator = test_gen(),
                                         steps = test_steps,
                                         
                                         # max_queue_size = 40, # 使用原生的就行了
                                         workers = 80, # 不要逼迫系统，不然很慢

                                         verbose = 2, # 1 显示进度条 2 只显示结果
                                         )

        y_prob = clf.predict_generator( generator = test_gen(),
                                         steps = test_steps,
                                         
                                         # max_queue_size = 40, # 使用原生的就行了
                                         workers = 80, # 不要逼迫系统，不然很慢

                                         verbose = 2, # 1 显示进度条 2 只显示结果
                                        )
        
        print( 'loss & acc: ' + str(score) )

        

        ####################
        # 原始标签
        ####################

        from keras.utils.np_utils import to_categorical

        # print(metadata)
        # print()
        # print(metadata[ train_val_size : ])

        label_test = [ i[1] for i in metadata[ train_val_size + 1 : ] ]
        # print(label_test)
        
        label_test = to_categorical(label_test)
        # print(label_test)



        ###########################
        # confusion matrix 混淆矩阵
        ###########################

        # conda install scikit-learn=0.22
        from sklearn.metrics._plot.confusion_matrix import confusion_matrix

        tn, fp, fn, tp = confusion_matrix( [ np.argmax(each_one_hot) for each_one_hot in label_test ],
                                           [ np.argmax(each_one_hot) for each_one_hot in y_prob ],
                                           ).ravel()

        print()
        print('Confusion Matrix')
        print('=====================')
        print('TN = ' + str(tn), end=' | ')
        print('FP = ' + str(fp))
        print('---------------------')
        print('FN = ' + str(fn), end=' | ')
        print('TP = ' + str(tp))
        print()

        #######################
        # 为0和1分别计算ROC AUC
        #######################
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        n_classes = label_test.shape[1] # n_classes = 2
        
        from sklearn.metrics import roc_curve, auc

        # print(n_classes)
        # print(label_test)
        # print()
        # print(y_prob)

        # 使用实际类别和预测概率计算 ROC 曲线各个点
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        print('roc_auc: ' + str(roc_auc))

        # gc.collect() # 回收三代垃圾



        ####################
        # 绘制ROC AUC曲线
        ####################

        plt.figure()
        
        lw = 2
        
        plt.plot(fpr[0], tpr[0], color = 'darkorange', 
                 lw = lw, label = 'ROC curve (area = %0.6f)' % roc_auc[1])
        
        plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)\nargs.name = ' + model_name)
        plt.legend(loc = 'lower right')

        plt.savefig( 'test/'+ model_name + '_' + tissue_name + '.png' )
        print('savefig: ' + 'test/'+ model_name + '_' + tissue_name + '.png')
        print( '#' * 40 )



    ####################
    # 同一模型多epoch
    ####################

    for i in range(1, 11): # 第二个比最多的加一个
        
        multi_models( model_name + '_' + str(i).zfill(2), tissue_name )
