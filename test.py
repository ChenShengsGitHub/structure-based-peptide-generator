import argparse
import pdb

import numpy as np

import data_loader
import tqdm
from tqdm import tqdm
import torch
import utils
from torch.utils.data import DataLoader, Dataset, RandomSampler,SequentialSampler,WeightedRandomSampler
from models.model import RNN_CVAE
import cfg
import json

def test_recon(model, dataset):
    results = []
    for kwargs in [{'sample_mode': 'greedy'},
                   {'sample_mode': 'categorical', 'temp': 1.0},
                   {'sample_mode': 'categorical', 'temp': 0.3},
                   {'sample_mode': 'beam', 'beam_size': 5, 'n_best': 3}]:
        pbar = tqdm(dataset, desc="test Iteration")
        result = {}
        result['source_recon'] = []
        result['score_recon'] = []
        result['score_gen'] = []
        result['z'] = []
        result['mu_var'] = []
        for step, batch in enumerate(pbar):
            sequence, eigen_CM = tuple(t.to('cuda') for t in batch)
            (z_mu, z_logvar),(z_mu2, z_logvar2), z, dec_logits = model(sequence, eigen_CM, q_c='prior', sample_z=1)
            source_seq = utils.idx2sentence(sequence.squeeze())

            log_sent, _ = model.generate_sentences(1, eigen_CM[:1], z, kwargs)
            z_fix = model.sample_z(z_mu2, z_logvar2)
            log_sent2, _ = model.generate_sentences(1, eigen_CM, z=z_fix, **kwargs)
            recon_seq = utils.idx2sentence(log_sent.squeeze())
            generate_seq = utils.idx2sentence(log_sent2.squeeze())
            mu_z = torch.mean(z)
            var_z = torch.var(z)
            mu_z_fix = torch.mean(z_fix)
            var_z_fix = torch.var(z_fix)
            mse_loss = torch.nn.MSELoss()
            mse_mu = mse_loss(z_mu, z_mu2)
            mse_logvar = mse_loss(z_logvar, z_logvar2)
            score = utils.SMalignment(recon_seq, source_seq)
            score2 = utils.SMalignment(generate_seq, source_seq)
            tqdm.write('Source: "{}"'.format(source_seq))
            tqdm.write('Sample: "{}"'.format(recon_seq))
            tqdm.write('Generate:"{}"'.format(generate_seq))
            tqdm.write('Score : {}'.format(score))
            tqdm.write('Score : {}'.format(score2))
            tqdm.write('mse mu_var: ({}, {})'.format(mse_mu.item(), mse_logvar.item()))
            tqdm.write(f'mu_var : ({mu_z},{var_z})({mu_z_fix},{var_z_fix})')
            result['source_recon'].append([source_seq, recon_seq])
            result['z'].append(z.cpu().detach().numpy())
            result['mu_var'].append('mu_var: ({}, {})'.format(torch.mean(z_mu.squeeze()),
                                                              torch.mean(torch.exp(z_logvar.squeeze()))))
            result['score_recon'].append(score)
            result['score_gen'].append(score2)
        avg_score = float(np.mean(result['score_recon']))
        print(kwargs)
        print(f'avg score_recon: {avg_score}')
        avg_score2 = float(np.mean(result['score_gen']))
        print(f'avg score_gen: {avg_score2}')
        mu = np.mean(result['z'])
        var = np.var(result['z'])
        print(f'mu: {mu}, var: {var}')
        result['mu_var'] = 'mu_var: ({}, {})'.format(mu, var)
        result.pop('z')
        results.append(result)
        pdb.set_trace()

    json_result = json.dumps(results)
    return json_result


def test_generate(model, dataset, train_dataset):

    results=[]
    train_pbar = tqdm(train_dataset, desc="test Iteration")
    train_AA_count = {}
    train_AA_count['total_source'] = {}
    train_AA_count['total_recon'] = {}
    train_AA_count['total_source_num'] = 0
    train_AA_count['total_recon_num'] = 0
    for step, batch in enumerate(train_pbar):
        sequence, eigen_CM = tuple(t.to('cuda') for t in batch)
        source_seq = utils.idx2sentence(sequence.squeeze())
        key = source_seq
        train_AA_count[key] = {}
        train_AA_count[key]['source'] = {}
        train_AA_count[key]['source']['total'] = 0
        for ch in source_seq:
            if ch not in cfg.AA_abb_dict:
                continue
            if ch not in train_AA_count[key]['source']:
                train_AA_count[key]['source'][ch] = 1
            if ch not in train_AA_count['total_source']:
                train_AA_count['total_source'][ch] = 1
            train_AA_count[key]['source'][ch] += 1
            train_AA_count['total_source'][ch] += 1
            train_AA_count[key]['source']['total'] += 1
            train_AA_count['total_source_num'] += 1

        train_AA_count[key]['recon'] = {}
        train_AA_count[key]['recon']['total'] = 0

    pbar = tqdm(dataset, desc="test Iteration")
    AA_count = {}
    AA_count['total_source'] = {}
    AA_count['total_recon'] = {}
    AA_count['total_source_num'] = 0
    AA_count['total_recon_num'] = 0
    for step, batch in enumerate(pbar):
        sequence, eigen_CM = tuple(t.to('cuda') for t in batch)
        source_seq = utils.idx2sentence(sequence.squeeze())
        key = source_seq
        AA_count[key] = {}
        AA_count[key]['source'] = {}
        AA_count[key]['source']['total'] = 0
        for ch in source_seq:
            if ch not in cfg.AA_abb_dict:
                continue
            if ch not in AA_count[key]['source']:
                AA_count[key]['source'][ch] = 1
            if ch not in AA_count['total_source']:
                AA_count['total_source'][ch] = 1
            AA_count[key]['source'][ch] += 1
            AA_count['total_source'][ch] += 1
            AA_count[key]['source']['total'] += 1
            AA_count['total_source_num'] += 1

        AA_count[key]['recon'] = {}
        AA_count[key]['recon']['total'] = 0

    for kwargs in [{'sample_mode': 'greedy'},
                   {'sample_mode': 'categorical', 'temp': 1.0},
                   {'sample_mode': 'categorical', 'temp': 0.3},
                   {'sample_mode': 'beam', 'beam_size': 5, 'n_best': 3}]:
        result = {}
        result['source_gen'] = []
        result['score'] = []
        pbar = tqdm(dataset, desc="test Iteration")
        for step, batch in enumerate(pbar):
            sequence, eigen_CM = tuple(t.to('cuda') for t in batch)
            source_seq = utils.idx2sentence(sequence.squeeze())
            tqdm.write('Source: "{}"'.format(source_seq))
            z_prior = model.sample_z_prior(cfg.sample_num)
            eigen_CM = eigen_CM.expand(cfg.sample_num, -1)
            feature = model.linear(eigen_CM)
            z_mu, z_logvar = model.forward_encoder_EGCM(z_prior, feature)
            z_fix = model.sample_z(z_mu, z_logvar)

            log_sent, _ = model.generate_sentences(cfg.sample_num, eigen_CM, z=z_fix, **kwargs)
            max_score = 0
            max_seq = ''
            for i in range(len(log_sent)):
                recon_seq = utils.idx2sentence(log_sent[i].squeeze())
                for ch in recon_seq:
                    if ch not in cfg.AA_abb_dict:
                        continue
                    if ch not in AA_count[source_seq]['recon']:
                        AA_count[source_seq]['recon'][ch] = 1
                    if ch not in AA_count['total_recon']:
                        AA_count['total_recon'][ch] = 1
                    AA_count[source_seq]['recon'][ch] += 1
                    AA_count['total_recon'][ch] += 1
                    AA_count['total_recon_num'] += 1
                    AA_count[source_seq]['recon']['total'] += 1
                score = utils.SMalignment(recon_seq, source_seq)
                if score > max_score:
                    max_seq = recon_seq
                    max_score = score

            tqdm.write('Sample: "{}"'.format(max_seq))
            tqdm.write('Score : {}'.format(max_score))
            result['source_gen'].append([source_seq, max_seq])
            result['score'].append(max_score)

        for ch in cfg.AA_abb_dict:
            if ch not in AA_count['total_recon']:
                continue
            num = train_AA_count['total_source'][ch]
            total = train_AA_count['total_source_num']
            print(f'train- {ch}: {num / total}')

            num = AA_count['total_source'][ch]
            total = AA_count['total_source_num']
            print(f'source-{ch}: {num/total}')

            num = AA_count['total_recon'][ch]
            total = AA_count['total_recon_num']
            print(f'recon- {ch}: {num/total}')
        avg_score = float(np.mean(result['score']))
        print(kwargs)
        print(f'avg score: {avg_score}')
        results.append(result)
        pdb.set_trace()
    json_result = json.dumps(results)
    return json_result


def test_interpolate(model, dataset):
    results=[]
    for kwargs in [{'sample_mode': 'greedy'},
                   {'sample_mode': 'categorical', 'temp': 1.0},
                   {'sample_mode': 'categorical', 'temp': 0.3},
                   {'sample_mode': 'beam', 'beam_size': 5, 'n_best': 3}]:
        pbar = tqdm(dataset, desc="test Iteration")
        result = {}
        result['source_old_recon'] = []
        result['score'] = []
        result['z'] = []
        result['score_old'] = []
        for step, batch in enumerate(pbar):
            if step != 0:
                sequence_old = sequence
                eigen_CM_old = eigen_CM
                source_seq_old = utils.idx2sentence(sequence_old.squeeze())
            sequence, eigen_CM = tuple(t.to('cuda') for t in batch)
            if step == 0:
                continue
            (z_mu, z_logvar), z, dec_logits = model(sequence, eigen_CM, q_c='prior', sample_z=1)
            source_seq = utils.idx2sentence(sequence.squeeze())
            log_sent, _ = model.generate_sentences(1, eigen_CM_old, z, kwargs)
            recon_seq = utils.idx2sentence(log_sent.squeeze())

            tqdm.write('mu_var: ({}, {})'.format(torch.mean(z_mu.squeeze()), torch.mean(torch.exp(z_logvar.squeeze()))))
            tqdm.write('Sample: "{}"'.format(recon_seq))
            tqdm.write('Source: "{}"'.format(source_seq))
            score = utils.SMalignment(recon_seq, source_seq)
            score_old = utils.SMalignment(recon_seq, source_seq_old)
            tqdm.write('Score : {}'.format(score))
            tqdm.write('Source_old: "{}"'.format(source_seq_old))
            tqdm.write('Score_old : {}'.format(score_old))
            result['source_old_recon'].append([source_seq, source_seq_old, recon_seq])
            result['z'].append(z.cpu().detach().numpy())
            result['score'].append(score)
            result['score_old'].append(score_old)
        avg_score = float(np.mean(result['score']))
        print(kwargs)
        print(f'avg score: {avg_score}')
        avg_score_old = float(np.mean(result['score_old']))
        print(f'avg score old: {avg_score_old}')
        mu = np.mean(result['z'])
        var = np.var(result['z'])
        print(f'mu: {mu}, var: {var}')
        result['mu_var'] = 'mu_var: ({}, {})'.format(mu, var)
        result.pop('z')
        results.append(result)
        pdb.set_trace()
    json_result = json.dumps(results)
    return json_result


def test_save(model, dataset):
    seqs = {}
    seqs['pepbdb_all'] = []
    seqs['pepbdb_test'] = []
    seqs['gen_norm'] = []
    seqs['gen_regen'] = []
    seqs['gen_regen_nsample'] = []
    seqs['uniprot'] = []
    uniprot_data = json.load(open(cfg.uniprot_processed))
    for peptide in uniprot_data:
        pep_str = ''
        if 0 in peptide:
            continue
        for i in range(len(peptide)):
            if peptide[i] not in [1, 2, 3]:
                pep_str += cfg.AA_abb_dict_T[int(peptide[i])]
        seqs['uniprot'].append(pep_str)

    pepbdb_data = json.load(open(cfg.pepbdb_processed))
    for pep in pepbdb_data:
        for peptide in pepbdb_data[pep]['peptide']:
            if 0 in peptide:
                continue
            pep_str = ''
            for i in range(len(peptide)):
                if peptide[i] not in [1, 2, 3]:
                    pep_str += cfg.AA_abb_dict_T[int(peptide[i])]
            seqs['pepbdb_all'].append(pep_str)

    pbar = tqdm(dataset, desc="test Iteration")
    for step, batch in enumerate(pbar):
        sequence, eigen_CM = tuple(t.to('cuda') for t in batch)
        source_seq, has_unk = utils.idx2sentence(sequence.squeeze())
        if not has_unk and source_seq != '':
            seqs['pepbdb_test'].append(source_seq)

        z_norm = model.sample_z_prior(1)
        log_sent, _ = model.generate_sentences(1, eigen_CM, z=z_norm, sample_mode='categorical')
        gen_seq_norm, has_unk = utils.idx2sentence(log_sent.squeeze())
        if not has_unk and gen_seq_norm != '':
            seqs['gen_norm'].append(gen_seq_norm)

        z_prior = model.sample_z_prior(cfg.sample_num)
        eigen_CM = eigen_CM.expand(cfg.sample_num, -1)
        feature = model.linear(eigen_CM)
        z_mu, z_logvar = model.forward_encoder_EGCM(z_prior, feature)
        z_fix = model.sample_z(z_mu, z_logvar)

        log_sent, _ = model.generate_sentences(cfg.sample_num, eigen_CM, z=z_fix, sample_mode='categorical')
        gen_seq, has_unk = utils.idx2sentence(log_sent[0].squeeze())
        if not has_unk and gen_seq != '':
            seqs['gen_regen'].append(gen_seq)

        for i in range(len(log_sent)):
            gen_seq_n, has_unk = utils.idx2sentence(log_sent[i].squeeze())
            if not has_unk and gen_seq_n != '':
                seqs['gen_regen_nsample'].append(gen_seq_n)
    file = open(cfg.generated_savepath, 'w')
    json.dump(seqs, file)
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                     description='Override config float & string values')
    cfg._cfg_import_export(parser, cfg, mode='fill_parser')
    cfg._override_config(parser.parse_args(), cfg)


    model = RNN_CVAE(n_vocab=24, max_seq_len=cfg.pep_max_length_uniprot if
        cfg.train_mode == 'pretrain' else cfg.pep_max_length_pepbdb, **cfg.model).to('cuda')
    cfg.log.info(model)
    # model.load_state_dict(torch.load(cfg.finetuned_model))
    model.load_state_dict(torch.load(cfg.final_model, map_location=lambda storage, loc: storage), strict=False)
    test_dataset = data_loader.PeptideInfo(cfg.uniprot_processed if cfg.train_mode == 'pretrain'
                                           else cfg.pepbdb_processed, process_mode='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)
    train_dataset = data_loader.PeptideInfo(cfg.uniprot_processed if cfg.train_mode == 'pretrain'
                                           else cfg.pepbdb_processed, process_mode='train')
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
    model.eval()
    # result = test_recon(model, test_dataloader)
    # result = test_generate(model, test_dataloader, train_dataloader)
    # result = test_interpolate(model, test_dataloader)
    # with open(cfg.test_result, 'w') as file:
    #     file.write(result)
    test_save(model, test_dataloader)
