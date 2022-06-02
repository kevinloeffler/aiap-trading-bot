import csv

BLOCK_SIZE = 12
lines = []

# extract lines from console output
with open('test.txt') as file:
    lines = file.readlines()

# open target csv
target_file = open('model_evaluation.csv', 'w')
writer = csv.writer(target_file)

# write title row
writer.writerow(['run', 'rnn_units', 'step', 'batch_size', 'learning_rate',
                 'dropout', 'test_loss', 'train_loss', 'training_loss', 'ratio', 'error_neg', 'error_pos'])


def handle_block(block: [str], block_nr: int):
    row = [block_nr]
    # rnn units
    row.append('-')
    # step
    row.append('-')
    # batch_size
    row.append('-')
    # learning rate
    row.append('-')
    # dropout
    row.append('-')
    # test loss
    test_loss = float(block[8].split()[3])
    row.append(test_loss)
    # train loss
    train_loss = float(block[7].split()[3])
    row.append(train_loss)
    # training loss
    row.append(block[11].split()[3])
    # ratio
    row.append(round(test_loss / train_loss, ndigits=5))
    # error negative
    row.append(block[10].split()[3][1: -1])
    # error positive
    row.append(block[9].split()[3][1: -1])

    writer.writerow(row)



block = []
counter = 0
block_counter = 0

for i in range(len(lines)):
    if counter < BLOCK_SIZE:
        block.append(lines[i])
        counter += 1
    else:
        handle_block(block=block, block_nr=block_counter)
        block = [lines[i]]
        counter = 1
        block_counter += 1

handle_block(block=block, block_nr=block_counter)

target_file.close()
