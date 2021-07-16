import argparse

import data_loader
import tqdm
import sys
from tqdm import tqdm

import torch
import torch.optim as optim

from models.mutils import save_model
import utils
import losses
from tb_json_logger import log_value
import tb_json_logger
from torch.utils.data import DataLoader, Dataset, RandomSampler,SequentialSampler,WeightedRandomSampler
import numpy as np
from models.model import RNN_CVAE
import cfg
from os.path import join as pjoin
import time
import torch.nn as nn

time_str = time.strftime("%m-%d-%H")
def train(cfgv, model, dataset):
    if cfg.train_mode == 'z_gen':
        trainer = optim.Adam([{'params':model.encoder_EGCM_mu.parameters()}, {'params':model.encoder_EGCM_logvar.parameters()}], lr=cfgv.lr*10)
    else:
        trainer = optim.Adam(model.vae_params(), lr=cfgv.lr if cfg.train_mode == 'pretrain' else cfgv.lr * 0.01)
    val_dataset = data_loader.PeptideInfo(cfg.uniprot_processed if cfg.train_mode=='pretrain'
                                          else cfg.pepbdb_processed, process_mode='validate')
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=cfg.batch_size*4)
    val_avg_loss = 5
    for epoch in range(cfg.total_epoch if cfg.train_mode != 'z_gen' else 200):
        pbar = tqdm(dataset, desc="train Iteration")
        total_loss = []
        for step, batch in enumerate(pbar):
            it = epoch*len(pbar)+step
            if it % cfgv.cheaplog_every == 0 or it % cfgv.expsvlog_every == 0:
                def tblog(k, v):
                    log_value('train_' + k, v, it)
            else:
                tblog = lambda k, v: None
            sequence, eigen_CM = tuple(t.to('cuda') for t in batch)
            beta = utils.anneal(cfgv.beta, it)
            (z_mu, z_logvar), (z_mu2, z_logvar2), z, dec_logits = model(sequence, eigen_CM, q_c='prior', sample_z=1)
            mse_loss = nn.MSELoss()
            mse_mu = mse_loss(z_mu, z_mu2)
            mse_logvar = mse_loss(z_logvar, z_logvar2)
            recon_loss = losses.recon_dec(sequence, dec_logits)
            kl_loss = losses.kl_gaussianprior(z_mu, z_logvar)
            wae_mmdrf_loss = losses.wae_mmd_gaussianprior(z, method='rf')
            z_regu_loss = wae_mmdrf_loss
            z_logvar_L1 = z_logvar.abs().sum(1).mean(0)  # L1 in z-dim, mean over mb.
            z_logvar_KL_penalty = losses.kl_gaussian_sharedmu(z_mu, z_logvar)
            loss = recon_loss + beta * z_regu_loss \
                   + cfgv.lambda_logvar_L1 * z_logvar_L1 \
                   + cfgv.lambda_logvar_KL * z_logvar_KL_penalty \
                   + beta * mse_mu + (beta-1) * mse_logvar if cfg.train_mode=='z_gen' else 0

            trainer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.vae_params(), cfgv.clip_grad)
            trainer.step()
            tblog('z_mu_L1', z_mu.data.abs().mean().item())
            tblog('z_logvar', z_logvar.data.mean().item())
            tblog('z_logvar_L1', z_logvar_L1.item())
            tblog('z_logvar_KL_penalty', z_logvar_KL_penalty.item())
            tblog('L_vae', loss.item())
            tblog('L_vae_recon', recon_loss.item())
            tblog('L_vae_kl', kl_loss.item())
            tblog('L_wae_mmdrf', wae_mmdrf_loss.item())
            tblog('beta', beta)
            tblog('val_avg_loss', val_avg_loss)
            total_loss.append(recon_loss.item())
            avg_loss = np.sum(total_loss) / len(total_loss)
            pbar.set_postfix({f'training...epoch:{epoch},avg loss': avg_loss})
            pbar.update(1)
            if it % cfgv.cheaplog_every == 0 or it % cfgv.expsvlog_every == 0:
                tqdm.write(
                    'ITER {} TRAINING (phase 1). loss_vae: {:.4f}; loss_recon: {:.4f}; loss_kl: {:.4f}; loss_mmdrf: {:.4f}; '
                    'Grad_norm: {:.4e}; mse_mu: {:.4e}; mse_logvar: {:.4e};'
                        .format(it, loss.item(), recon_loss.item(), kl_loss.item(), wae_mmdrf_loss.item(),
                                grad_norm, mse_mu, mse_logvar))
                log_sent, _ = model.generate_sentences(1, eigen_CM[:1], sample_mode='categorical')
                tqdm.write('Sample (cat T=1.0): "{}"'.format(utils.idx2sentence(log_sent.squeeze())))
                sys.stdout.flush()
        model.eval()
        val_avg_loss = validate(cfgv, model, val_dataloader)
        save_model(model, cfgv.chkpt_path.format(time_str,cfg.train_mode, epoch))
        model.train()


def validate(cfgv, model, dataset):
    pbar = tqdm(dataset, desc="validate Iteration")
    total_loss = []
    total_loss_mu = []
    total_loss_logvar = []
    avg_loss = 0
    for step, batch in enumerate(pbar):
        sequence, eigen_CM = tuple(t.to('cuda') for t in batch)
        (z_mu, z_logvar), (z_mu2, z_logvar2), z, dec_logits = model(sequence, eigen_CM, q_c='prior', sample_z=1)
        if cfg.train_mode=='z_gen':
            mse_loss = nn.MSELoss()
            mse_mu = mse_loss(z_mu, z_mu2)
            mse_logvar = mse_loss(z_logvar, z_logvar2)
            total_loss_mu.append(mse_mu.item())
            total_loss_logvar.append(mse_logvar.item())
            avg_loss_mu = np.sum(total_loss_mu) / len(total_loss_mu)
            avg_loss_logvar = np.sum(total_loss_logvar) / len(total_loss_logvar)
            pbar.set_postfix({'validating... avg loss mu': avg_loss_mu, 'avg loss logvar': avg_loss_logvar})
            pbar.update(1)
        else:
            recon_loss = losses.recon_dec(sequence, dec_logits)
            total_loss.append(recon_loss.item())
            avg_loss = np.sum(total_loss) / len(total_loss)
            pbar.set_postfix({'validating... avg loss': avg_loss})
            pbar.update(1)

    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                     description='Override config float & string values')
    cfg._cfg_import_export(parser, cfg, mode='fill_parser')
    cfg._override_config(parser.parse_args(), cfg)
    result_json = pjoin(cfg.savepath, 'result.json') if cfg.resume_result_json else None
    tb_json_logger.configure(cfg.tbpath, result_json)


    model = RNN_CVAE(n_vocab=24, max_seq_len=cfg.pep_max_length_uniprot if
        cfg.train_mode == 'pretrain' else cfg.pep_max_length_pepbdb, **cfg.model).to('cuda')
    cfg.log.info(model)
    if cfg.train_mode == 'pretrain':
        train_dataset = data_loader.PeptideInfo(cfg.uniprot_processed)
    else:
        train_dataset = data_loader.PeptideInfo(cfg.pepbdb_processed)
        pretrain_weight = torch.load(cfg.fintuned_model if cfg.train_mode == 'z_gen' else cfg.pretrained_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrain_weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    train_sampler = RandomSampler(train_dataset)
    dataset = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg.batch_size)
    model.train()
    train(cfg.vae, model, dataset)
    tb_json_logger.export_to_json(pjoin(cfg.savepath.format(time_str), 'result.json'))
    tb_json_logger.export_to_json(pjoin(cfg.savepath.format(time_str), 'vae_result.json'),
                                  it_filter=lambda k, v: k <= cfg.vae.n_iter)