# -*- coding:utf-8 -*-

def pre(tissue_name):

    ####################
    # 清空屏幕
    ####################

    import os
    # os.system('clear')

    ####################
    # 创建文件夹
    ####################
     
    def mkdir(path):
         
        folder = os.path.exists(path)
         
        if not folder: # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path) # makedirs 创建文件时如果路径不存在会创建这个路径
        else:
            pass # 存在就不做操作
                        
    mkdir('histogram/T1w') # 调用函数
    mkdir('histogram/T2w') # 调用函数
    
    mkdir('visualization/T1w') # 调用函数
    mkdir('visualization/T2w') # 调用函数

    ####################
    # 读取全部数据
    ####################

    import pandas as pd

    metadata = pd.read_csv( tissue_name + '.csv', # 'metadata.csv',
                            usecols = ['doc_name'] )

    metadata = metadata.values
    # print(metadata[:3])

    ####################
    # 绘图
    ####################

    # 因为服务器没有图形界面，所以必须这样弄
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    # plt.rcParams['savefig.dpi'] = 500 # 图片像素

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

    def read_nii(name, procnum, return_dict):

        img = nib.load( 'abcd_data/' + name )
        img_array = img.get_fdata()


        # 进行 1/2 的缩放
        import scipy.ndimage
            
        output = scipy.ndimage.interpolation.zoom(
            input = img_array,
            zoom = 0.5,
            order = 3
            )


        ####################
        # 归一化
        ####################
        
        output = normalization(output)

        ####################
        # 绘制直方图
        ####################

        """
        绘制直方图
        data: 必选参数，绘图数据
        bins: 直方图的长条形数目，可选项，默认为10
        normed: 是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed = 1，表示归一化，显示频率。
        facecolor: 长条形的颜色
        edgecolor: 长条形边框的颜色
        alpha: 透明度
        """
        import numpy as np
        # print(type(output))

        plt.hist( output[ np.logical_and( output > 0.001, output < 0.999 ) ].flatten(), bins = 100, facecolor = 'blue', edgecolor = 'black', alpha = 0.7 ) # normed = 0, 已废弃
        # plt.hist( output.flatten(), bins = 100, facecolor = 'blue', edgecolor = 'black', alpha = 0.7 ) # normed = 0, 已废弃

        plt.xlabel('Interval')
        plt.ylabel('Frequency')

        plt.title('Histogram')

        plt.savefig( 'histogram/' + tissue_name + '/' + str(procnum) + '.png' )
        plt.close()


        # 有很多值很接近于 0 但是不是 0
        # print( max(output.tolist(), key = output.tolist().count) )

        ####################
        # 绘制扫描图
        ####################

        fig, ax = plt.subplots()

        # pip install scikit-image
        from skimage.util import montage

        fig, ax = plt.subplots( 1, 1, figsize = (20, 20) )
        ax.imshow( montage(output), cmap = 'bone')

        fig.savefig( 'visualization/' + tissue_name + '/' + str(procnum) + '.png',
                     bbox_inches = 'tight',
                     # dpi = 500,
                     pad_inches = 0
                     )
        plt.close()


        print(procnum, end='\t')

        print('name: ', end='')
        print(name, end='\t')

        print('shape: ', end='')
        print(output.shape, end='\t')
        print('max: ', end='')
        print(output.max(), end='\t')
        print('min: ', end='')
        print(output.min())

        return_dict[ procnum ] = [ output.max(), output.min() ]

    ####################
    # 多进程异步扫数据
    ####################

    import time

    import multiprocessing

    return_dict = multiprocessing.Manager().dict()


    for i in range( len(metadata) ): # 可以少弄点
        
        p = multiprocessing.Process( target = read_nii, args = ( metadata[i][0], i, return_dict ) )
        p.start()

        time.sleep( 0.07 ) # 避免加入太快


    """
    print()
    print(return_dict)
    print()
    print(return_dict.values())
    """

    time.sleep( 10 ) # 等所有子进程运行完成 10 秒

    abcd_min = 0
    abcd_max = 0

    for i in range( len(metadata) ): # 可以少弄点

        if return_dict[i][0] > abcd_max:
            abcd_max = return_dict[i][0]

        if return_dict[i][1] < abcd_min:
            abcd_min = return_dict[i][1]

    print()
    print( 'abcd_min = ' + str(abcd_min) )
    print( 'abcd_max = ' + str(abcd_max) )
