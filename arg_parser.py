import argparse


def arg_parse():
    parser = argparse.ArgumentParser()

    hparams = parser.add_argument_group('hyper-parameters')
    hparams.add_argument('-N', '--city_num', type=int, default=50, help="Number of task points in each sample")
    hparams.add_argument('-k', '--clusters_num', type=int, default=3, help="Number of clusters")
    hparams.add_argument('-F', '--feature_dim', type=int, default=2, help="Dimension of task point feature")
    hparams.add_argument('--sample_num', type=int, default=1000000, help="Sample number within the generated dataset")
    hparams.add_argument('-M', '--batch_size', type=int, default=32, help="Batch size to divide the dataset")
    hparams.add_argument('--lamb', type=float, default=0.5,
                         help="Lambda for balancing the distance cost and unsupervised losses")
    hparams.add_argument('--lamb_decay', type=float, default=1.0, help="Decay rate of lambda after each iteration")
    hparams.add_argument('--max_grad_norm', type=float, default=10.0, help="Threshold for gradient clipping")
    hparams.add_argument('--lr', type=float, default=0.01, help="Learning rate for the optimiser")
    hparams.add_argument('--embedding_dim', type=int, default=128,
                         help="Dimension of the embedder of the attention model")
    hparams.add_argument('--hidden_dim', type=int, default=128, help="Dimension of the hidden layer in MLP or MoE MLP")
    hparams.add_argument('--n_component', type=int, default=3, help="Number of experts for MoE")
    hparams.add_argument('--cost_d_op', choices=['sum', 'max'], type=str, default='sum',
                         help="Number of experts for MoE")
    hparams.add_argument('--penalty_score', type=float, default=10.0, help="Penalty score for degeneration")

    options = parser.add_argument_group('options')
    options.add_argument('--model_type', type=str, choices=['mlp', 'moe_mlp', 'attention'], default='moe_mlp',
                         help="Type of the reinforcement agent model")
    options.add_argument('--data_type', type=str, choices=['blob', 'random', 'file'], default='random',
                         help="Type of generated dataset")
    options.add_argument('--data_filename', type=str, default=None, help="directory of the to dataset for importing")
    options.add_argument('--data_normalise', action='store_true',
                         help="Indicate whether the dataset needs normalisation")
    options.add_argument('--log_dir', type=str, default='logs', help="Directory to save the logs")
    options.add_argument('--checkpoint_interval', type=int, default=200,
                         help="Interval to generate showcase and save model")
    options.add_argument('--gradient_check_flag', type=bool, default=False, help="Whether to check the gradient flow")
    options.add_argument('--save_model', type=bool, default=True, help="Whether to save the trained model")

    options.add_argument('--pretrain_dir', type=str, default=None,
                         help="Where to load the pretrained model for training")
    options.add_argument('--eval_dir', type=str, default=None, help="Where to load the model for evaluation")

    train_eval = parser.add_mutually_exclusive_group(required=True)
    train_eval.add_argument('--train', action='store_true', help="Set the system into training mode")
    train_eval.add_argument('--eval', action='store_true', help="Set the system into evaluation mode")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pass
