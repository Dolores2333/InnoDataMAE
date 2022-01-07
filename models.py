# -*- coding: utf-8 _*_
# @Time : 5/1/2022 10:34 am
# @Author: ZHA Mengyue
# @FileName: models.py
# @Blog: https://github.com/Dolores2333

from tqdm import tqdm
from generation import *
from visualization import *


class AverageMAEUnit(nn.Module):
    def __init__(self, args):
        super(AverageMAEUnit, self).__init__()
        self.args = args
        self.ts_size = args.ts_size
        self.mask_size = args.mask_size
        self.num_masks = args.num_masks
        self.total_mask_size = args.num_masks * args.mask_size
        args.total_mask_size = self.total_mask_size
        self.z_dim = args.z_dim
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def mae_random_mask(self, x, masks):
        """With masks seen by the encoder by default"""
        x_vis = mask_with_average(self.args, x, masks)  # (bs, seq_len, z_dim)
        x_enc = self.encoder(x_vis)
        x_dec = self.decoder(x_enc)
        return x_enc, x_dec, masks

    def embed_random_mask(self, x, masks):
        x_vis = mask_with_average(self.args, x, masks)  # (bs, seq_len, z_dim)
        x_enc = self.encoder(x_vis)
        return x_enc

    def mae_pseudo_mask(self, x, masks):
        """mae_pseudo_mask is equivalent to the Autoencoder"""
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_enc, x_dec, masks

    def forward(self, x, masks, mode):
        x_encoded = None
        x_decoded = None
        if mode == 'train_ae':
            x_encoded, x_decoded, masks = self.mae_pseudo_mask(x, masks)
        elif mode == 'train_mae':
            x_encoded, x_decoded, masks = self.mae_random_mask(x, masks)
        elif mode == 'random_generation':
            x_encoded, x_decoded, masks = self.mae_random_mask(x, masks)
        elif mode == 'embed_random_mask':
            return self.embed_random_mask(x, masks)
        return x_encoded, x_decoded, masks


class AverageMAE(nn.Module):
    def __init__(self, args, ori_data):
        super(AverageMAE, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.model = AverageMAEUnit(args).to(self.device)
        self.ori_data = ori_data  # ori_data has been normalized in main.py
        self.random_masks = generate_random_masks(args, args.batch_size)  # (bs. ts_size)
        self.pseudo_masks = generate_pseudo_masks(args, args.batch_size)  # (bs, ts_size)
        self.len_ori_data = len(self.ori_data)
        args.len_ori_data = self.len_ori_data
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.results = {'n_updates': 0,
                        'loss': []}
        print(f'Successfully initialized {self.__class__.__name__}!')

    def train_ae(self):
        self.model.train()

        for t in tqdm(range(self.args.ae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            x_enc, x_dec, masks = self.model(x_ori, self.pseudo_masks, 'train_ae')
            loss = self.criterion(x_dec, x_ori)

            self.results['n_updates'] = t
            self.results['loss'].append(loss.clone().detach().cpu().numpy())
            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} total loss')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_metrics_results(self.args, self.results)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_mae(self):
        for t in tqdm(range(self.args.mae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            random_masks = generate_random_masks(self.args, self.args.batch_size)  # (bs, ts_size)

            # get target x_ori_enc by Autoencoder
            self.model.eval()
            x_ori_enc = self.model(x_ori, self.pseudo_masks, 'embed_random_mask').clone().detach()
            b, l, f = x_ori_enc.size()


            start_epochs = t * (self.args.embed_epochs + self.args.recon_epochs)
            num_iteration = 0

            # Embedding Training!
            for k in range(self.args.embed_epochs):
                self.model.train()
                x_enc, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')

                # loss = self.criterion(x_enc, x_ori_enc)  # embed_loss
                # Only calculate loss for those being masked
                x_enc_masked = x_enc[masks, :].reshape(b, -1, f)
                x_ori_enc_masked = x_ori_enc[masks, :].reshape(b, -1, f)
                loss = self.criterion(x_enc_masked, x_ori_enc_masked)
                # By annotate lines above, we take loss on all patches

                current_epochs = start_epochs + num_iteration
                self.results['n_updates'] = current_epochs
                self.results['loss'].append(loss.clone().detach().cpu().numpy())

                if current_epochs % self.args.log_interval == 0:
                    print(f'Epoch {current_epochs} with {loss.item()} loss.')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_iteration += 1

            b, l, f = x_ori.size()

            # Reconstruction Training!
            for k in range(self.args.recon_epochs):
                self.model.train()
                x_enc, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')

                # loss = self.criterion(x_dec, x_ori)
                # Only calculate loss for those being masked
                x_dec_masked = x_dec[masks, :].reshape(b, -1, f)
                x_ori_masked = x_ori[masks, :].reshape(b, -1, f)
                loss = self.criterion(x_dec_masked, x_ori_masked)
                # By annotate lines above, we take loss on all patches

                current_epochs = start_epochs + num_iteration
                self.results['n_updates'] = current_epochs
                self.results['loss'].append(loss.clone().detach().cpu().numpy())

                if current_epochs % self.args.log_interval == 0:
                    print(f'Epoch {current_epochs} with {loss.item()} loss.')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_iteration += 1

    def evaluate_ae(self):
        """Evaluate the model as a simple AntoEncoder"""
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = full_generation(self.args, self.model, ori_data)
        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization
        plot_time_series_no_masks(self.args)
        pca_and_tsne(self.args)

    def evaluate_mae(self):
        '''Evaluate the model as a Masked AutoEncoder'''
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = None
        if self.args.generation_mode == 'random':
            art_data = random_generation(self.args, self.model, ori_data)
        elif self.args.generation_mode == 'cross':
            art_data = cross_generation(self.args, self.model, ori_data)
        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization
        plot_time_series_with_masks(self.args)
        pca_and_tsne(self.args)


class AverageMAEScheme1(nn.Module):
    def __init__(self, args, ori_data):
        super(AverageMAEScheme1, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.model = AverageMAEUnit(args).to(self.device)
        self.ori_data = ori_data  # ori_data has been normalized in main.py
        self.random_masks = generate_random_masks(args, args.batch_size)  # (bs. ts_size)
        self.pseudo_masks = generate_pseudo_masks(args, args.batch_size)  # (bs, ts_size)
        self.len_ori_data = len(self.ori_data)
        args.len_ori_data = self.len_ori_data
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.results = {'n_updates': 0,
                        'loss': []}
        print(f'Successfully initialized {self.__class__.__name__}!')

    def train_ae(self):
        self.model.train()

        for t in tqdm(range(self.args.ae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            x_enc, x_dec, masks = self.model(x_ori, self.pseudo_masks, 'train_ae')
            loss = self.criterion(x_dec, x_ori)

            self.results['n_updates'] = t
            self.results['loss'].append(loss.clone().detach().cpu().numpy())
            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} total loss')
                if bool(self.args.save):
                    save_model(self.args, self.model)
                    save_metrics_results(self.args, self.results)
                    save_args(self.args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_embed(self):
        for t in tqdm(range(self.args.mae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            random_masks = generate_random_masks(self.args, self.args.batch_size)  # (bs, ts_size)

            # get target x_ori_enc by Autoencoder
            self.model.eval()
            x_ori_enc = self.model(x_ori, self.pseudo_masks, 'embed_random_mask').clone().detach()
            b, l, f = x_ori_enc.size()

            self.model.train()
            x_enc, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')

            # loss = self.criterion(x_enc, x_ori_enc)  # embed_loss
            # Only calculate loss for those being masked
            x_enc_masked = x_enc[masks, :].reshape(b, -1, f)
            x_ori_enc_masked = x_ori_enc[masks, :].reshape(b, -1, f)
            loss = self.criterion(x_enc_masked, x_ori_enc_masked)
            # By annotate lines above, we take loss on all patches

            self.results['n_updates'] = t + self.args.ae_epochs
            self.results['loss'].append(loss.clone().detach().cpu().numpy())

            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} loss.')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_recon(self):
        for t in tqdm(range(self.args.mae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            random_masks = generate_random_masks(self.args, self.args.batch_size)  # (bs, ts_size)

            b, l, f = x_ori.size()
            self.model.train()
            x_enc, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')

            # loss = self.criterion(x_dec, x_ori)
            # Only calculate loss for those being masked
            x_dec_masked = x_dec[masks, :].reshape(b, -1, f)
            x_ori_masked = x_ori[masks, :].reshape(b, -1, f)
            loss = self.criterion(x_dec_masked, x_ori_masked)
            # By annotate lines above, we take loss on all patches

            self.results['n_updates'] = t + self.args.ae_epochs + self.args.mae_epochs
            self.results['loss'].append(loss.clone().detach().cpu().numpy())

            if t % self.args.log_interval == 0:
                print(f'Epoch {t} with {loss.item()} loss.')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_joint(self):
        for t in tqdm(range(self.args.mae_epochs)):
            x_ori = get_batch(args=self.args, data=self.ori_data)
            x_ori = torch.tensor(x_ori, dtype=torch.float32).to(self.device)
            random_masks = generate_random_masks(self.args, self.args.batch_size)  # (bs, ts_size)

            # get target x_ori_enc by Autoencoder
            self.model.eval()
            x_ori_enc = self.model(x_ori, self.pseudo_masks, 'embed_random_mask').clone().detach()
            b, l, f = x_ori_enc.size()

            start_epochs = (t * (self.args.embed_epochs + self.args.recon_epochs)
                            + self.args.ae_epochs + self.args.mae_epochs * 2)
            num_iteration = 0

            # Embedding Training!
            for k in range(self.args.embed_epochs):
                self.model.train()
                x_enc, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')

                # loss = self.criterion(x_enc, x_ori_enc)  # embed_loss
                # Only calculate loss for those being masked
                x_enc_masked = x_enc[masks, :].reshape(b, -1, f)
                x_ori_enc_masked = x_ori_enc[masks, :].reshape(b, -1, f)
                loss = self.criterion(x_enc_masked, x_ori_enc_masked)
                # By annotate lines above, we take loss on all patches

                current_epochs = start_epochs + num_iteration
                self.results['n_updates'] = current_epochs
                self.results['loss'].append(loss.clone().detach().cpu().numpy())

                if current_epochs % self.args.log_interval == 0:
                    print(f'Epoch {current_epochs} with {loss.item()} loss.')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_iteration += 1

            b, l, f = x_ori.size()

            # Reconstruction Training!
            for k in range(self.args.recon_epochs):
                self.model.train()
                x_enc, x_dec, masks = self.model(x_ori, random_masks, 'train_mae')

                # loss = self.criterion(x_dec, x_ori)
                # Only calculate loss for those being masked
                x_dec_masked = x_dec[masks, :].reshape(b, -1, f)
                x_ori_masked = x_ori[masks, :].reshape(b, -1, f)
                loss = self.criterion(x_dec_masked, x_ori_masked)
                # By annotate lines above, we take loss on all patches

                current_epochs = start_epochs + num_iteration
                self.results['n_updates'] = current_epochs
                self.results['loss'].append(loss.clone().detach().cpu().numpy())

                if current_epochs % self.args.log_interval == 0:
                    print(f'Epoch {current_epochs} with {loss.item()} loss.')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_iteration += 1

    def evaluate_ae(self):
        """Evaluate the model as a simple AntoEncoder"""
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = full_generation(self.args, self.model, ori_data)
        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization
        plot_time_series_no_masks(self.args)
        pca_and_tsne(self.args)

    def evaluate_mae(self):
        '''Evaluate the model as a Masked AutoEncoder'''
        self.model.eval()
        ori_data = torch.tensor(self.ori_data, dtype=torch.float32).to(self.device)
        art_data = None
        if self.args.generation_mode == 'random':
            art_data = random_generation(self.args, self.model, ori_data)
        elif self.args.generation_mode == 'cross':
            art_data = cross_generation(self.args, self.model, ori_data)
        art_data = art_data.clone().detach().cpu().numpy()
        art_data *= self.args.max_val
        art_data += self.args.min_val
        np.save(self.args.art_data_dir, art_data)  # save art_data after renormalization
        plot_time_series_with_masks(self.args)
        pca_and_tsne(self.args)
