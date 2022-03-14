import random

def main():
    user = {}
    with open('ratings.dat', 'r') as fin:
        for line in fin.readlines():
            row = [int(x) for x in line.strip().split('::')]
            if row[2] >= 3:
                if row[0] not in user:
                    user[row[0]] = []
                user[row[0]].append(row[1])

    M = 3952
    max_basket = 0
    examples = []
    for k, v in user.items():
        example = [0] * M
        max_basket = max(max_basket, len(v))
        for x in v:
            example[x-1] = 1
        examples.append(example)

    random.shuffle(examples)
    n = len(examples)
    train = examples[: (n * 8) // 10]
    valid = examples[(n * 8) // 10: (n * 9) // 10]
    test = examples[(n * 9) // 10:]

    for name, dataset in zip(['train', 'valid', 'test'], [train, valid, test]):
        with open(f'movielens1M.{name}.data', 'w') as fout:
            for example in dataset:
                fout.write(','.join([str(x) for x in example]) + '\n')

    print(max_basket)

if __name__ == '__main__':
    main()