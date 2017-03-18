import utils.parse_history as ph
import numpy as np
import matplotlib.pyplot as plt

def get_data_epoch_loss(one_th):
    return [one_th.epoch, one_th.loss]

def get_data_epoch_train(one_th):
    return [one_th.epoch, one_th.train_accu]

def get_data_epoch_test(one_th):
    return [one_th.epoch, one_th.test_accu]

def get_final_train_accu(one_th):
    return one_th.train_accu[-1]

def get_final_train_accu(one_th):
    return one_th.test_accu[-1]

def analysis_lr_0316():
    training_history_txt_filename='results/training_history_lr_0316.txt'
    output_path='results/lr_0316/'
    f=open(training_history_txt_filename,'r')
    training_history_list=ph.parse_file(f)
    f.close()

    num_models=len(training_history_list)

    # get the learning_rate and test accuracies
    epoch=training_history_list[0].epoch

    lrs=[]
    losses=[]
    train_accus=[]
    test_accus=[]
    for i in range(num_models):
        one_th=training_history_list[i]
        lrs.append(one_th.learning_rate)
        losses.append(one_th.loss)
        train_accus.append(one_th.train_accu)
        test_accus.append(one_th.train_accu)

    plt.ioff()
    fig1=plt.figure(1)
    fig2=plt.figure(2)
    fig3=plt.figure(3)
    for i in range(num_models):
        plt.figure(1)
        plt.plot(epoch,losses[i])
        plt.figure(2)
        plt.plot(epoch,train_accus[i])
        plt.figure(3)
        plt.plot(epoch,test_accus[i])
    fig1.savefig(output_path+'epoch_loss.png')
    fig2.savefig(output_path+'epoch_train_accu.png')
    fig3.savefig(output_path+'epoch_test_accu.png')

def main():
    analysis_lr_0316()

if __name__ == '__main__':
    main()
