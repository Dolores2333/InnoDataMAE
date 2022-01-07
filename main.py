# -*- coding: utf-8 _*_
# @Time : 5/1/2022 10:34 am
# @Author: ZHA Mengyue
# @FileName: main.py
# @Blog: https://github.com/Dolores2333


from models import *


def run_average_mae():
    """Run Latent MAE on pretrined Autoencoder
        set ae_epochs to 0 if you wanna no ae pretrain
        set mae_epochs to 0 if you wanna ae only"""
    home = os.getcwd()
    args = load_arguments(home)

    # Data Loading
    ori_data = load_data(args)
    np.save(args.ori_data_dir, ori_data)  # save ori_data before normalization
    ori_data, min_val, max_val = min_max_scalar(ori_data)
    # Write statistics
    args.min_val = min_val
    args.max_val = max_val
    args.data_var = np.var(ori_data)
    print(f'{args.data_name} data variance is {args.data_var}')

    # Initialize the Model
    model = AverageMAEScheme1(args, ori_data)
    if args.training:
        print(f'Start AutoEncoder Training! {args.ae_epochs} Epochs Needed.')
        model.train_ae()
        print('AutoEncoder Training Finished!')
        print(f'Start Embedding Training! {args.mae_epochs} Epochs Needed.')
        model.train_embed()
        print('Embedding Training Finished!')
        print(f'Start Reconstruction Training! {args.mae_epochs} Epochs Needed.')
        model.train_recon()
        print('Reconstruction Training Finished!')
        print(f'Start Joint Training! {args.mae_epochs * (args.embed_epochs + args.recon_epochs)} Epochs Needed.')
        model.train_joint()
        print('Joint Training Finished!')
    else:
        model = load_model(args, model)
        print(f'Successfully loaded the model!')
    if args.mae_epochs == 0:
        model.evaluate_ae()
    else:
        model.evaluate_mae()


if __name__ == '__main__':
    run_average_mae()
