import os


# Model 2. bidirectional
os.system('python3 train.py --model bidirectional '
          '--num_layer 2 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name new_bi_l2_u64')
os.system('python3 train.py --model bidirectional '
          '--num_layer 2 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name new_bi_l2_u32')
os.system('python3 train.py --model bidirectional '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name new_bi_l3_u64')
os.system('python3 train.py --model bidirectional '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name new_bi_l3_u32')
os.system('python3 train.py --model bidirectional '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name new_bi_l4_u64')
os.system('python3 train.py --model bidirectional '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name new_bi_l4_u32')

# Model 3. hierarchical
os.system('python3 train.py --model hierarchical '
          '--num_layer 2 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name new_augmented_hi_l2_u64_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 2 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name new_hi_l2_u32_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name new_hi_l3_u64_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name new_hi_l3_u32_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name new_hi_l4_u64_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name new_hi_l4_u32_em16_au')

'''
# Model 1. unidirectional
os.system('python3 train.py --model unidirectional '
          '--num_layer 2 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_un_l2_u64')
os.system('python3 train.py --model unidirectional '
          '--num_layer 2 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_un_l2_u32')
os.system('python3 train.py --model unidirectional '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_un_l3_u64')
os.system('python3 train.py --model unidirectional '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_un_l3_u32')
os.system('python3 train.py --model unidirectional '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_un_l4_u64')
os.system('python3 train.py --model unidirectional '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_un_l4_u32')

# Model 2. bidirectional
os.system('python3 train.py --model bidirectional '
          '--num_layer 2 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_bi_l2_u64')
os.system('python3 train.py --model bidirectional '
          '--num_layer 2 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_bi_l2_u32')
os.system('python3 train.py --model bidirectional '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_bi_l3_u64')
os.system('python3 train.py --model bidirectional '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_bi_l3_u32')
os.system('python3 train.py --model bidirectional '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_bi_l4_u64')
os.system('python3 train.py --model bidirectional '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name augmented_bi_l4_u32')

# Model 3. hierarchical
os.system('python3 train.py --model hierarchical '
          '--num_layer 2 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name test2_augmented_hi_l2_u64_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 2 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --no_aux_sup '
          '--name augmented_hi_l2_u64_em16_noau')
os.system('python3 train.py --model hierarchical '
          '--num_layer 2 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name augmented_hi_l2_u32_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 2 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --no_aux_sup '
          '--name augmented_hi_l2_u32_em16_noau')
os.system('python3 train.py --model hierarchical '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name augmented_hi_l3_u64_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --no_aux_sup '
          '--name augmented_hi_l3_u64_em16_noau')
os.system('python3 train.py --model hierarchical '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name augmented_hi_l3_u32_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --no_aux_sup '
          '--name augmented_hi_l3_u32_em16_noau')
os.system('python3 train.py --model hierarchical '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name augmented_hi_l4_u64_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --no_aux_sup '
          '--name augmented_hi_l4_u64_em16_noau')
os.system('python3 train.py --model hierarchical '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name augmented_hi_l4_u32_em16_au')
os.system('python3 train.py --model hierarchical '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --no_aux_sup '
          '--name augmented_hi_l4_u32_em16_noau')

# Model 4. transformer
os.system('python3 train.py --model transformer '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --num_head 8 '
          '--name augmented_tr_l3_u32_em16_nh8')
os.system('python3 train.py --model transformer '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --num_head 16 '
          '--name augmented_tr_l3_u32_em16_nh16')
os.system('python3 train.py --model transformer '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --num_head 8 '
          '--name augmented_tr_l3_u64_em16_nh8')
os.system('python3 train.py --model transformer '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --num_head 16 '
          '--name augmented_tr_l3_u64_em16_nh16')
os.system('python3 train.py --model transformer '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --num_head 8 '
          '--name augmented_tr_l4_u32_em16_nh8')
os.system('python3 train.py --model transformer '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --num_head 16 '
          '--name augmented_tr_l4_u32_em16_nh16')
os.system('python3 train.py --model transformer '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --num_head 8 '
          '--name augmented_tr_l4_u64_em16_nh8')
os.system('python3 train.py --model transformer '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --num_head 16 '
          '--name augmented_tr_l4_u64_em16_nh16')

# Model 5. seq2seq
os.system('python3 train.py --model seq2seq_attention '
          '--num_layer 2 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 '
          '--name augmented_se_l2_u64_em16')
os.system('python3 train.py --model seq2seq_attention '
          '--num_layer 2 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 '
          '--name augmented_se_l2_u32_em16')
os.system('python3 train.py --model seq2seq_attention '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 '
          '--name augmented_se_l3_u64_em16')
os.system('python3 train.py --model seq2seq_attention '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 '
          '--name augmented_se_l3_u32_em16')
os.system('python3 train.py --model seq2seq_attention '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 '
          '--name augmented_se_l4_u64_em16')
os.system('python3 train.py --model seq2seq_attention '
          '--num_layer 4 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 '
          '--name augmented_se_l4_u32_em16')
'''
