import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.propagate = False  # do not propagate logs to previously defined root logger (if any).
formatter = logging.Formatter('%(asctime)s - %(levelname)s(%(name)s): %(message)s')
# console
consH = logging.StreamHandler()
consH.setFormatter(formatter)
consH.setLevel(logging.INFO)
logger.addHandler(consH)
# file handler
request_file_handler = True
log = logger
resume_result_json = True


class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self


AA_abb_dict = {"<unk>": 0, "<pad>": 1, "<start>": 2, "<eos>": 3, "A": 4, "C": 5, "D": 6, "E": 7,
               "F": 8, "G": 9, "H": 10, "I": 11, "K": 12, "L": 13, "M": 14, "N": 15, "P": 16,
               "Q": 17, "R": 18, "S": 19, "T": 20, "V": 21, "W": 22, "Y": 23}
AA_abb_dict_T = {v:k for k, v in AA_abb_dict.items()}
AA_dict = {"<unk>": 0, "<pad>": 1, "<start>": 2, "<eos>": 3, "ALA": 4, "CYS": 5, "ASP": 6, "GLU": 7,
           "PHE": 8, "GLY": 9, "HIS": 10, "ILE": 11, "LYS": 12, "LEU": 13, "MET": 14, "ASN": 15,"PRO": 16,
           "GLN": 17, "ARG": 18, "SER": 19, "THR": 20, "VAL": 21, "TRP": 22, "TYR": 23}

pep_max_length_uniprot = 40
pep_max_length_pepbdb = 40
EGCM_max_length = 400
process_pepbdb = True
process_uniprot = False
pepbdb_source = "/home/chens/data/pepbdb/pepbdb"
pepbdb_processed = 'pepbdb_sorted.json'
uniprot_yes_source = '/home/chens/data/uniprot/uniprot-reviewed_yes.fasta'
uniprot_no_source = '/home/chens/data/uniprot/uniprot-reviewed_no.fasta'
uniprot_processed = 'uniprot.json'
test_result = 'test_result.json'

EGCM_max_value = 100
EGCM_embeded_length = 50

train_mode = 'finetune'  #or pretrain
pretrained_model = 'output/07-14/pretrain_model_57.pt'
fintuned_model = 'output/07-14-21/finetune_model_59.pt'
final_model = 'output/07-15-12/z_gen_model_59.pt'
savepath='output/{}'
tbpath = 'tb/default'
generated_savepath = 'generated.json'
batch_size = 16
total_epoch = 60
sample_num = 20

def _cfg_import_export(cfg_interactor, cfg_, prefix='', mode='fill_parser'):
    """ Iterate through cfg_ module/object. For known variables import/export
    from cfg_interactor (dict, argparser, or argparse namespace) """
    for k in dir(cfg_):
        if k[0] == '_': continue  # hidden
        v = getattr(cfg_, k)
        if type(v) in [float, str, int, bool]:
            if mode == 'fill_parser':
                cfg_interactor.add_argument('--{}{}'.format(prefix, k), type=type(v), help='default: {}'.format(v))
            elif mode == 'fill_dict':
                cfg_interactor['{}{}'.format(prefix, k)] = v
            elif mode == 'override':
                prek = '{}{}'.format(prefix, k)
                if prek in cfg_interactor:
                    setattr(cfg_, k, getattr(cfg_interactor, prek))
        elif type(v) == Bunch:  # recurse; descend into Bunch
            _cfg_import_export(cfg_interactor, v, prefix=prefix + k + '.', mode=mode)


def _override_config(args, cfg):
    """ call _cfg_import_export in override mode, update cfg from:
        (1) contents of config_json (taken from (a) loadpath if not auto, or (2) savepath)
        (2) from command line args
    """
    config_json = vars(args).get('config_json', '')
    _cfg_import_export(args, cfg, mode='override')


vae = Bunch(
    batch_size=1,
    lr=1e-3,
    # TODO lrate decay with scheduler
    s_iter=0,
    n_iter=200000,
    beta=Bunch(
        start=Bunch(val=1.0, iter=0),
        end=Bunch(val=2.0, iter=10000)
    ),
    lambda_logvar_L1=0.0,  # default from https://openreview.net/pdf?id=r157GIJvz
    lambda_logvar_KL=1e-3,  # default from https://openreview.net/pdf?id=r157GIJvz
    z_regu_loss='mmdrf',  # kl (vae) | mmd (wae) | mmdrf (wae)
    cheaplog_every=500,  # cheap tensorboard logging eg training metrics
    expsvlog_every=20000,  # expensive logging: model checkpoint, heldout set evals, word emb logging
    chkpt_path='./output/{}/{}_model_{}.pt',
    clip_grad=5.0,
)
vae.beta.start.iter = vae.s_iter
vae.beta.end.iter = vae.s_iter + vae.n_iter // 5

model = Bunch(
    z_dim=100,
    c_dim=2,
    emb_dim=150,
    pretrained_emb=None,  # set True to load from dataset_unl.get_vocab_vectors()
    freeze_embeddings=False,
    flow=0,
    flow_type='',
    E_args=Bunch(
        h_dim=80,  # 20 for amp, 64 for yelp
        biGRU=True,
        layers=1,
        p_dropout=0.0
    ),
    G_args=Bunch(
        G_class='gru',
        GRU_args=Bunch(
            # h_dim = (z_dim + c_dim) for now. TODO parametrize this?
            p_word_dropout=0.3,
            p_out_dropout=0.3,
            skip_connetions=False,
        ),
        deconv_args=Bunch(
            max_seq_len=pep_max_length_pepbdb if train_mode=='finetune' else pep_max_length_uniprot,
            num_filters=100,
            kernel_size=4,
            num_deconv_layers=3,
            useRNN=False,
            temperature=1.0,
            use_batch_norm=True,
            num_conv_layers=2,
            add_final_conv_layer=True,
        ),
    ),
    C_args=Bunch(
        min_filter_width=3,
        max_filter_width=5,
        num_filters=100,
        dropout=0.5
    )
)

# config for the losses, constant during training & phases
losses = Bunch(
    wae_mmd=Bunch(
        sigma=7.0,  # ~ O( sqrt(z_dim) )
        kernel='gaussian',
        # for method = rf
        rf_dim=500,
        rf_resample=False
    ),
)
