import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    # Name and directories
    parser.add_argument('-model_name', type=str, default='basic')
    parser.add_argument('-data_dir', type=str, default='/home/rjh347/data/squad/')
    parser.add_argument('-run_id', type=int, default=0)
    parser.add_argument('-out_dir', type=str, default='/home/rjh347/data/squad/')
    parser.add_argument('-forward_name', type=str, default='single')
    parser.add_argument('-answer_path', type=str, default='')
    parser.add_argument('-eval_path', type=str, default='')
    parser.add_argument('-load_path', type=str, default='')
    parser.add_argument('-shared_path', type=str, default='/home/rjh347/data/squad/')

    # Device placement
    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-num_gpus', type=int, default=1)

    # Essential training and test options
    parser.add_argument('-mode', type=str, default="test")
    parser.add_argument('-load', type=bool, default=True)
    parser.add_argument('-single', type=bool, default=False)
    parser.add_argument('-debug', type=bool, default=False)
    parser.add_argument('-load_ema', type=bool, default=True)
    parser.add_argument('-eval', type=bool, default=True)

    # Traning / test parameters
    parser.add_argument('-batch_size', type=int, default=50)
    parser.add_argument('-num_epochs', type=int, default=12)
    parser.add_argument('-num_steps', type=int, default=20000)
    parser.add_argument('-init_lr', type=float, default=0.5)
    parser.add_argument('-input_keep_prob', type=float, default=0.8)
    parser.add_argument('-keep_prob', type=float, default=0.8)
    parser.add_argument('-wd', type=float, default=0.0)
    parser.add_argument('-hidden_size', type=int, default=100)
    parser.add_argument('-char_out_size', type=int, default=100)
    parser.add_argument('-char_emb_size', type=int, default=8)
    parser.add_argument('-out_channel_dims', type=int, default=100)
    parser.add_argument('-filter_heights', type=int, default=5)
    parser.add_argument('-finetune', type=bool, default=False)
    parser.add_argument('-highway', type=bool, default=True)
    parser.add_argument('-highway_num_layers', type=int, default=2)
    parser.add_argument('-share_cnn_weights', type=bool, default=True)
    parser.add_argument('-share_lstm_weights', type=bool, default=True)
    parser.add_argument('-var_decay', type=float, default=0.999)

    # Optimization
    parser.add_argument('-cluster', type=bool, default=False)

    # Logging and saving options
    parser.add_argument('-progress', type=bool, default=True)
    parser.add_argument('-log_period', type=int, default=100)
    parser.add_argument('-eval_period', type=int, default=1000)
    parser.add_argument('-save_period', type=int, default=1000)
    parser.add_argument('-max_to_keep', type=int, default=20)
    parser.add_argument('-dump_eval', type=bool, default=True)
    parser.add_argument('-dump_answer', type=bool, default=True)
    parser.add_argument('-vis', type=bool, default=False)
    parser.add_argument('-dump_pickle', type=bool, default=True)
    parser.add_argument('-decay', type=float, default=0.9)

    # Thresholds for speed and less memory usage
    parser.add_argument('-word_count_th', type=int, default=10)
    parser.add_argument('-char_count_th', type=int, default=50)
    parser.add_argument('-sent_size_th', type=int, default=400)
    parser.add_argument('-num_sents_th', type=int, default=8)
    parser.add_argument('-ques_size_th', type=int, default=30)
    parser.add_argument('-word_size_th', type=int, default=16)
    parser.add_argument('-para_size_th', type=int, default=256)

    # Advanced training options
    parser.add_argument('-lower_word', type=bool, default=True)
    parser.add_argument('-squash', type=bool, default=False)
    parser.add_argument('-swap_memory', type=bool, default=True)
    parser.add_argument('-data_filter', type=str, default="max")
    parser.add_argument('-use_glove_for_unk', type=bool, default=False)
    parser.add_argument('-known_if_glove', type=bool, default=True)
    parser.add_argument('-logit_func', type=str, default="tri_linear")
    parser.add_argument('-answer_func', type=str, default="linear")
    parser.add_argument('-sh_logit_func', type=str, default="tri_linear")

    # Ablation options
    parser.add_argument('-use_char_emb', type=bool, default=True)
    parser.add_argument('-use_word_emb', type=bool, default=True)
    parser.add_argument('-q2c_att', type=bool, default=True)
    parser.add_argument('-c2q_att', type=bool, default=True)
    parser.add_argument('-dynamic_att', type=bool, default=False)

    args = parser.parse_args()

    return args
