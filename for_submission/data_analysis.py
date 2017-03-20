import utils.parse_history as ph
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_data_epoch_loss(one_th):
    return [one_th.epoch, one_th.loss]

def get_data_epoch_train(one_th):
    return [one_th.epoch, one_th.train_accu]

def get_data_epoch_test(one_th):
    return [one_th.epoch, one_th.test_accu]

def get_final_train_accu(one_th):
    return np.amax(one_th.train_accu)

def get_final_test_accu(one_th):
    return np.amax(one_th.test_accu)



def analysis():
    dataset='gutenberg'
    parameter='hs'
    date='0318'
    para_name='hidden_size'
    training_history_txt_filename='results/training_historyword_'+parameter+'_'+dataset+'_'+date+'.txt'
    output_path='results/word'+parameter+'_'+dataset+'_'+date
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f=open(training_history_txt_filename,'r')
    training_history_list=ph.parse_file(f)
    f.close()

    num_models=len(training_history_list)

    # get the learning_rate and test accuracies
    epoch=training_history_list[0].epoch

    paras=[]
    losses=[]
    train_accus=[]
    test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        """
        choose parameter here
        """
        paras.append(one_th.hidden_size)
        losses.append(one_th.loss)
        train_accus.append(one_th.train_accu)
        test_accus.append(one_th.test_accu)


    fig1=plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    handles1=[]

    fig2=plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    handles2=[]

    fig3=plt.figure(3)
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    handles3=[]
    for i in range(num_models):
        plt.figure(1)
        line1, =plt.plot(epoch,losses[i],label=para_name+'='+str(paras[i]))
        handles1.append(line1)

        plt.figure(2)
        line2, =plt.plot(epoch,train_accus[i],label=para_name+'='+str(paras[i]))
        handles2.append(line2)

        plt.figure(3)
        line3, =plt.plot(epoch,test_accus[i],label=para_name+'='+str(paras[i]))
        handles3.append(line3)

    plt.figure(1)
    plt.legend(handles=handles1,loc='upper right')
    plt.figure(2)
    plt.legend(handles=handles2,loc='lower right')
    plt.figure(3)
    plt.legend(handles=handles3,loc='lower right')


    fig1.savefig(output_path+'/epoch_loss.png')
    fig2.savefig(output_path+'/epoch_train_accu.png')
    fig3.savefig(output_path+'/epoch_test_accu.png')

    fig4=plt.figure(4)
    plt.figure(4)

    final_train_accus=[]
    final_test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        final_train_accus.append(get_final_train_accu(one_th))
        final_test_accus.append(get_final_test_accu(one_th))

    line_train, =plt.plot(paras,final_train_accus,label='train')
    line_test, =plt.plot(paras,final_test_accus,label='test')
    plt.xlabel(para_name)
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend(handles=[line_train,line_test])
    fig4.savefig(output_path+'/parameter.png')

def analysis_lstm():
    dataset='c50'
    parameter='hs'
    date='0319'
    para_name='hidden_size'
    training_history_txt_filename='results/lstm_'+parameter+'_'+dataset+'_'+date+'.txt'
    output_path='results/lstm'+parameter+'_'+dataset+'_'+date
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f=open(training_history_txt_filename,'r')
    training_history_list=ph.parse_file(f)
    f.close()

    num_models=len(training_history_list)

    # get the learning_rate and test accuracies
    epoch=training_history_list[0].epoch

    paras=[]
    losses=[]
    train_accus=[]
    test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        """
        choose parameter here
        """
        paras.append(one_th.hidden_size)
        losses.append(one_th.loss)
        train_accus.append(one_th.train_accu)
        test_accus.append(one_th.test_accu)


    fig1=plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    handles1=[]

    fig2=plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    handles2=[]

    fig3=plt.figure(3)
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    handles3=[]
    for i in range(num_models):
        plt.figure(1)
        line1, =plt.plot(epoch,losses[i],label=para_name+'='+str(paras[i]))
        handles1.append(line1)

        plt.figure(2)
        line2, =plt.plot(epoch,train_accus[i],label=para_name+'='+str(paras[i]))
        handles2.append(line2)

        plt.figure(3)
        line3, =plt.plot(epoch,test_accus[i],label=para_name+'='+str(paras[i]))
        handles3.append(line3)

    plt.figure(1)
    plt.legend(handles=handles1,loc='upper right')
    plt.figure(2)
    plt.legend(handles=handles2,loc='lower right')
    plt.figure(3)
    plt.legend(handles=handles3,loc='lower right')


    fig1.savefig(output_path+'/epoch_loss.png')
    fig2.savefig(output_path+'/epoch_train_accu.png')
    fig3.savefig(output_path+'/epoch_test_accu.png')

    fig4=plt.figure(4)
    plt.figure(4)

    final_train_accus=[]
    final_test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        final_train_accus.append(get_final_train_accu(one_th))
        final_test_accus.append(get_final_test_accu(one_th))

    line_train, =plt.plot(paras,final_train_accus,label='train')
    line_test, =plt.plot(paras,final_test_accus,label='test')
    plt.xlabel(para_name)
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend(handles=[line_train,line_test])
    fig4.savefig(output_path+'/parameter.png')

def analysis_lambda():
    dataset='c50'
    parameter='lam'
    date='0319'
    para_name='lambda'
    training_history_txt_filename='results/training_history_siamese_lambda.txt'
    output_path='results/siamese_lambda'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f=open(training_history_txt_filename,'r')
    training_history_list=ph.parse_file(f)
    f.close()

    num_models=len(training_history_list)

    # get the learning_rate and test accuracies
    epoch=training_history_list[0].epoch

    paras=[]
    losses=[]
    train_accus=[]
    test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        """
        choose parameter here
        """
        paras.append(one_th.regularization)
        losses.append(one_th.loss)
        train_accus.append(one_th.train_accu)
        test_accus.append(one_th.test_accu)
    # lambda here
    paras=[0.001,0.005,0.01,0.05,0.1,0.5]

    fig1=plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    handles1=[]

    fig2=plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    handles2=[]

    fig3=plt.figure(3)
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    handles3=[]
    for i in range(num_models):
        plt.figure(1)
        line1, =plt.plot(epoch,losses[i],label=para_name+'='+str(paras[i]))
        handles1.append(line1)

        plt.figure(2)
        line2, =plt.plot(epoch,train_accus[i],label=para_name+'='+str(paras[i]))
        handles2.append(line2)

        plt.figure(3)
        line3, =plt.plot(epoch,test_accus[i],label=para_name+'='+str(paras[i]))
        handles3.append(line3)

    plt.figure(1)
    plt.legend(handles=handles1,loc='upper right')
    plt.figure(2)
    plt.legend(handles=handles2,loc='lower right')
    plt.figure(3)
    plt.legend(handles=handles3,loc='lower right')


    fig1.savefig(output_path+'/epoch_loss.png')
    fig2.savefig(output_path+'/epoch_train_accu.png')
    fig3.savefig(output_path+'/epoch_test_accu.png')

    fig4=plt.figure(4)
    plt.figure(4)

    final_train_accus=[]
    final_test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        final_train_accus.append(get_final_train_accu(one_th))
        final_test_accus.append(get_final_test_accu(one_th))

    line_train, =plt.plot(paras,final_train_accus,label='train')
    line_test, =plt.plot(paras,final_test_accus,label='test')
    plt.xlabel(para_name)
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend(handles=[line_train,line_test])
    fig4.savefig(output_path+'/parameter.png')

def analysis_lambda2():
    dataset='c50'
    parameter='lam'
    date='0319'
    para_name='lambda'
    training_history_txt_filename='results/training_history_siamese_lambda_long.txt'
    output_path='results/siamese_lambda'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    f=open(training_history_txt_filename,'r')
    training_history_list=ph.parse_file(f)
    f.close()

    num_models=len(training_history_list)

    # get the learning_rate and test accuracies
    epoch=training_history_list[0].epoch

    paras=[]
    losses=[]
    train_accus=[]
    test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        """
        choose parameter here
        """
        paras.append(one_th.regularization)
        losses.append(one_th.loss)
        train_accus.append(one_th.train_accu)
        test_accus.append(one_th.test_accu)
    # lambda here
    paras=[0.001,0.005,0.01,0.05,0.1,0.5]

    fig1=plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    handles1=[]

    fig2=plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    handles2=[]

    fig3=plt.figure(3)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    handles3=[]
    for i in range(num_models):
        plt.figure(1)
        line1, =plt.plot(epoch,losses[i],label=para_name+'='+str(paras[i]))
        handles1.append(line1)

        plt.figure(2)
        line2, =plt.plot(epoch,train_accus[i],label=para_name+'='+str(paras[i]))
        handles2.append(line2)

        plt.figure(3)
        line3, =plt.plot(epoch,train_accus[i],label="train accuracy")
        line4, =plt.plot(epoch,test_accus[i],label="test accuracy")
        handles3.append(line3)
        handles3.append(line4)

    plt.figure(1)
    plt.legend(handles=handles1,loc='upper right')
    plt.figure(2)
    plt.legend(handles=handles2,loc='lower right')
    plt.figure(3)
    plt.legend(handles=handles3,loc='lower right')


    fig1.savefig(output_path+'/loss.png')
    fig2.savefig(output_path+'/train_accu.png')
    fig3.savefig(output_path+'/test_accu.png')
    '''
    fig4=plt.figure(4)
    plt.figure(4)

    final_train_accus=[]
    final_test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        final_train_accus.append(get_final_train_accu(one_th))
        final_test_accus.append(get_final_test_accu(one_th))

    line_train, =plt.plot(paras,final_train_accus,label='train')
    line_test, =plt.plot(paras,final_test_accus,label='test')
    plt.xlabel(para_name)
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend(handles=[line_train,line_test])
    fig4.savefig(output_path+'/parameter.png')
    '''

def main():
    analysis()

if __name__ == '__main__':
    main()
