import random

def main():
    user = {}
    M = 3952
    with open('ratings.dat', 'r') as fin:
        for line in fin.readlines():
            row = [int(x) for x in line.strip().split('::')]
            if row[0] not in user:
                user[row[0]] = [-1] * M
            if row[2] >= 3:
                user[row[0]][row[1] - 1] = 1
            if row[2] < 3:
                user[row[0]][row[1] - 1] = 0

    
    examples = []
    for k, v in user.items():
        examples.append(v)

    random.shuffle(examples)
    n = len(examples)
    train = examples[: (n * 8) // 10]
    valid = examples[(n * 8) // 10: (n * 9) // 10]
    test = examples[(n * 9) // 10:]

    for name, dataset in zip(['train', 'valid', 'test'], [train, valid, test]):
        with open(f'movielens1M_missing.{name}.data', 'w') as fout:
            for example in dataset:
                fout.write(','.join([str(x) for x in example]) + '\n')

if __name__ == '__main__':
    main()