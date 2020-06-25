import os


# Model 2. bidirectional
'''
os.system('python3 test_experiment.py --batch_size 1 --model bidirectional '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--name new_bi_l3_u64')
os.system('python3 test_experiment.py --batch_size 1 --model bidirectional '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 25 '
          '--name augmented_bi_l4_u64')
'''
# Model 3. hierarchical
os.system('python3 test_experiment.py --model hierarchical '
          '--num_layer 3 --num_unit 32 --num_epoch 500 --num_epoch_decay 500 --print_freq 250 '
          '--embedding_size 16 --aux_sup '
          '--name new_hi_l3_u32_em16_au')

'''
os.system('python3 test_experiment.py --batch_size 1 --model hierarchical '
          '--num_layer 2 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 1 '
          '--embedding_size 16 --aux_sup '
          '--name augmented_hi_l2_u64_em16_au')

os.system('python3 test_experiment.py --batch_size 1 --model hierarchical '
          '--num_layer 3 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 25 '
          '--embedding_size 16 --aux_sup '
          '--name augmented_hi_l3_u64_em16_au')
os.system('python3 test_experiment.py --batch_size 1 --model hierarchical '
          '--num_layer 4 --num_unit 64 --num_epoch 500 --num_epoch_decay 500 --print_freq 25 '
          '--embedding_size 16 --aux_sup '
          '--name augmented_hi_l4_u64_em16_au')
'''
