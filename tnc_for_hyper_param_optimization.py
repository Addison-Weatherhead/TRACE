import random
import argparse
import torch
from tnc.tnc import main

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    print("STARTED RUNNING", flush=True)
    from datetime import datetime
    print("Started running on ", datetime.now())
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run TNC')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--ID', type=str) 
    parser.add_argument('--plot_embeddings', action='store_true')
    parser.add_argument('--encoder_type', type=str)
    parser.add_argument('--DEBUG', action='store_true')
    
    # Transformer Hyperparameters
    parser.add_argument('--Transformer_fc_dropout', type=float)
    parser.add_argument('--Transformer_act', type=str)
    parser.add_argument('--Transformer_res_dropout', type=float)
    parser.add_argument('--Transformer_d_ff', type=int)
    parser.add_argument('--Transformer_d_v', type=int)
    parser.add_argument('--Transformer_d_qk', type=int)
    parser.add_argument('--Transformer_n_heads', type=int)
    parser.add_argument('--Transformer_hidden_size', type=int)
    parser.add_argument('--Transformer_n_layers', type=int)
    parser.add_argument('--Transformer_encoding_size', type=int)
    parser.add_argument('--Transformer_num_features', type=int)

    # CNN RNN Hyperparameters
    parser.add_argument('--CNN_RNN_latent_size', type=int)
    parser.add_argument('--CNN_RNN_encoding_size', type=int)
    parser.add_argument('--CNN_RNN_in_channel', type=int)

    # GRUD Hyperparameters
    parser.add_argument('--GRUD_num_features', type=int)
    parser.add_argument('--GRUD_hidden_size', type=int)
    parser.add_argument('--GRUD_num_layers', type=int)
    parser.add_argument('--GRUD_encoding_size', type=int)
    parser.add_argument('--GRUD_extra_layer_types', type=str)
    parser.add_argument('--GRUD_dropout', type=float)

    # RNN Hyperparameters
    parser.add_argument('--RNN_hidden_size', type=int)
    parser.add_argument('--RNN_in_channel', type=int)
    parser.add_argument('--RNN_encoding_size', type=int)

    # CNN_Transformer Hyperparameters
    parser.add_argument('--CNN_Transformer_latent_size', type=int)
    parser.add_argument('--CNN_Transformer_encoding_size', type=int)
    parser.add_argument('--CNN_Transformer_in_channel', type=int)
    parser.add_argument('--CNN_Transformer_transformer_n_layers', type=int)
    parser.add_argument('--CNN_Transformer_transformer_hidden_size', type=int)
    parser.add_argument('--CNN_Transformer_transformer_n_heads', type=int)
    parser.add_argument('--CNN_Transformer_transformer_d_ff', type=int)
    parser.add_argument('--CNN_Transformer_transformer_res_dropout', type=float)
    parser.add_argument('--CNN_Transformer_transformer_act', type=str)
    parser.add_argument('--CNN_Transformer_transformer_fc_dropout', type=float)

    # CausalCNNEncoder Hyperparameters
    parser.add_argument('--CausalCNNEncoder_in_channels', type=int)
    parser.add_argument('--CausalCNNEncoder_channels', type=int)
    parser.add_argument('--CausalCNNEncoder_depth', type=int)
    parser.add_argument('--CausalCNNEncoder_reduced_size', type=int)
    parser.add_argument('--CausalCNNEncoder_encoding_size', type=int)
    parser.add_argument('--CausalCNNEncoder_kernel_size', type=int)
    parser.add_argument('--CausalCNNEncoder_window_size', type=int)



    # Learn encoder hyperparams
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--w', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--decay', type=float)
    parser.add_argument('--mc_sample_size', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--n_cross_val_encoder', type=int)
    parser.add_argument('--ETA', type=int)
    parser.add_argument('--ADF', action='store_true')
    parser.add_argument('--ACF', action='store_true')
    parser.add_argument('--ACF_PLUS', action='store_true')
    parser.add_argument('--ACF_nghd_Threshold', type=float)
    parser.add_argument('--ACF_out_nghd_Threshold', type=float)

    # Classifier hyper params
    parser.add_argument('--n_cross_val_classification', type=int)

    args = parser.parse_args()

    if args.encoder_type == 'Transformer':
        encoder_hyper_params = {'verbose': False,
                                'y_range': None,
                                'fc_dropout': args.Transformer_fc_dropout,
                                'act': args.Transformer_act,
                                'res_dropout': args.Transformer_res_dropout,
                                'd_ff': args.Transformer_d_ff,
                                'd_v': args.Transformer_d_v,
                                'd_qk': args.Transformer_d_qk,
                                'n_heads': args.Transformer_n_heads,
                                'hidden_size': args.Transformer_hidden_size,
                                'n_layers': args.Transformer_n_layers,
                                'max_seq_len': None,
                                'seq_len': args.window_size,
                                'encoding_size': args.Transformer_encoding_size,
                                'num_features': args.Transformer_num_features}
    
    elif args.encoder_type == 'CNN_RNN': 
        encoder_hyper_params = {'latent_size': args.CNN_RNN_latent_size,
                                'encoding_size': args.CNN_RNN_encoding_size,
                                'in_channel': 2}

    elif args.encoder_type == 'GRUD':
        encoder_hyper_params = {'num_features': args.GRUD_num_features,
                                'hidden_size': args.GRUD_hidden_size,
                                'num_layers': args.GRUD_num_layers,
                                'encoding_size': args.GRUD_encoding_size,
                                'extra_layer_types': args.GRUD_extra_layer_types,
                                'dropout': args.GRUD_dropout}

    elif args.encoder_type == 'RNN':
        encoder_hyper_params = {'hidden_size': args.RNN_hidden_size,
                                'in_channel': args.RNN_in_channel,
                                'encoding_size': args.RNN_encoding_size}
    
    elif args.encoder_type == 'CNN_Transformer':
        encoder_hyper_params = {'latent_size': args.CNN_Transformer_latent_size,
                                'encoding_size': args.CNN_Transformer_encoding_size,
                                'in_channel': args.CNN_Transformer_in_channel,
                                'transformer_n_layers': args.CNN_Transformer_transformer_n_layers, 
                                'transformer_hidden_size': args.CNN_Transformer_transformer_hidden_size, 
                                'transformer_n_heads': args.CNN_Transformer_transformer_n_heads, 
                                'transformer_d_ff': args.CNN_Transformer_transformer_d_ff, 
                                'transformer_res_dropout': args.CNN_Transformer_transformer_res_dropout, 
                                'transformer_act': args.CNN_Transformer_transformer_act, 
                                'transformer_fc_dropout': args.CNN_Transformer_transformer_fc_dropout}
    
    elif args.encoder_type == 'CausalCNNEncoder':
        encoder_hyper_params = {'in_channels': args.CausalCNNEncoder_in_channels,
                                'channels': args.CausalCNNEncoder_channels, 
                                'depth': args.CausalCNNEncoder_depth, 
                                'reduced_size': args.CausalCNNEncoder_reduced_size,
                                'encoding_size': args.CausalCNNEncoder_encoding_size,
                                'kernel_size': args.CausalCNNEncoder_kernel_size,
                                'window_size': args.CausalCNNEncoder_window_size}

                                
    learn_encoder_hyper_params = {'window_size': args.window_size,
                                    'w': args.w,
                                    'batch_size': args.batch_size,
                                    'lr': args.lr,
                                    'decay': args.decay,
                                    'mc_sample_size': args.mc_sample_size,
                                    'n_epochs': args.n_epochs,
                                    'data_type': args.data_type,
                                    'n_cross_val_encoder': args.n_cross_val_encoder,
                                    'cont': True,
                                    'ETA': args.ETA,
                                    'ADF': args.ADF,
                                    'ACF': args.ACF,
                                    'ACF_PLUS': args.ACF_PLUS,
                                    'ACF_nghd_Threshold': args.ACF_nghd_Threshold,
                                    'ACF_out_nghd_Threshold': args.ACF_out_nghd_Threshold}
    
    classification_hyper_params = {'n_cross_val_classification': args.n_cross_val_classification}
    
    
    UNIQUE_ID = args.ID

    
    UNIQUE_NAME = UNIQUE_ID + '_' + args.encoder_type + '_' + args.data_type
    
    print('UNIQUE_NAME: ', UNIQUE_NAME)

    
    if 'device' not in encoder_hyper_params:
        encoder_hyper_params['device'] = device
    
    if 'device' not in learn_encoder_hyper_params:
        learn_encoder_hyper_params['device'] = device
    
    pretrain_hyper_params = {}

    #################################################################    
    print("ENCODER HYPER PARAMETERS")
    for key in encoder_hyper_params:
        print(key)
        print(encoder_hyper_params[key])
        print()
    print("LEARN ENCODER HYPER PARAMETERS")
    for key in learn_encoder_hyper_params:
        print(key)
        print(learn_encoder_hyper_params[key])
        print()

    main(args.train, args.data_type, args.encoder_type, encoder_hyper_params, learn_encoder_hyper_params, classification_hyper_params, args.cont, pretrain_hyper_params, args.plot_embeddings, UNIQUE_ID, UNIQUE_NAME)
    print("Finished running on ", datetime.now())
    
