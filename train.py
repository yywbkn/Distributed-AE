"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., asynKL) and
different datasets (with option '--dataset_mode': e.g., covid_split, covid_test).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train Distributed Autoencoder Classifier Network  model:
        python train.py --dataroot  /COVID_data/ --name covid_KL_train --model asynKL

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import sys
import time
from options.train_options import TrainOptions
from options.val_options import ValOptions
from data import create_dataset,create_val_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    print('dataset type = ', type(dataset))
    print('dataset length = ', len(dataset))
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    opt2 = ValOptions().parse()
    val_dataset = create_val_dataset(opt2)
    val_dataset_size = len(val_dataset)  # get the number of images in the dataset.
    print('The number of val_dataset_size images = %d' % val_dataset_size)

    flag = False

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations


    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch


        ii_flag = model.inde
        print('ii_flag = ', ii_flag)
        print('ii_flag type = ', type(ii_flag))
        if flag == True:
            model.load_networks_i('latest', ii = ii_flag)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            # print('*'*20)
            print('this is batchsize   i = ', i)
            sys.stdout.flush()

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            # ii_flag = model.val()
            # flag = True
        if epoch % opt.save_epoch_freq == 0:
            for i, val_data in enumerate(val_dataset):
                model.set_val_input(val_data)  # unpack data from dataset and apply preprocessing
                model.val()  # calculate loss functions, get gradients, update network weights
            flag = True
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch
