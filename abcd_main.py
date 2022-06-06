# -*- coding:utf-8 -*-

if __name__ == '__main__':

    ####################
    # 创建文件夹
    ####################
    
    import os
 
    def mkdir(path):
     
        folder = os.path.exists(path)
     
        if not folder: # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path) # makedirs 创建文件时如果路径不存在会创建这个路径
        else:
            pass # 存在就不做操作
                    
    mkdir('train') # 调用函数
    mkdir('test') # 调用函数
    
    mkdir('model') # 调用函数


    import argparse

    # 不能使用垃圾回收系统，这样会变得non-picklable
    # 从而影响 keras 的 generator thread safe

    # 三代垃圾回收机制
    # import gc
    # gc.enable()

    ####################
    # 解析器
    ####################
    
    parser = argparse.ArgumentParser()

    # 引入一个时间
    import time


    print( '*' * 40 )
    print( 'begin time: ' + time.asctime(time.localtime(time.time())) )
    print()


    #############################
    # 参数：选择类型 训练还是测试
    #############################
    
    parser.add_argument('-t', '--type',
                        help = 'Please choose [train] / [test] / [pre] (all use lowercase).',
                        required = False)

    ######################
    # 参数：选取的标签名称
    ######################
    
    parser.add_argument('-n', '--name',
                        help = 'Please enter a label name, like [stb_c1] (case sensitive).',
                        required = False)

    ####################
    # 参数：选取组织名称
    ####################
    
    parser.add_argument('-s', '--select',
                        help = 'Please choose tissue name [T1w] / [T2w] (case sensitive).',
                        required = False)

    args = parser.parse_args()

    print('=== Below is your input ===')
    
    try: print('args.type = '+ args.type)
    except: pass
    try: print('args.name = '+ args.name)
    except: pass
    try: print('args.select = '+ args.select)
    except: pass
    
    print('===========================\n')

    
    if args.type == 'train':
        import abcd_train
        abcd_train.train(args.name, args.select)

    elif args.type == 'test':
        import abcd_test
        abcd_test.test(args.name, args.select)

    elif args.type == 'pre':
        import abcd_data_pre
        abcd_data_pre.pre(args.select)
    

    print( '\nend time: ' + time.asctime(time.localtime(time.time())) )
    print( '*' * 40 + '\n' * 20 )
