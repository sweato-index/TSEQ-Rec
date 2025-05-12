import argparse
import os
import collections
from tqdm import tqdm


valid_timestamp = 1628643414042
test_timestamp = 1658002729837


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='processed_data/5core/rating_only')
    parser.add_argument('--output_path', type=str, default='processed_data/5core/timestamp')
    parser.add_argument('--seq_path', type=str, default='processed_data/5core/timestamp_w_his')
    parser.add_argument('--zero', action='store_true', help='if true, will process for 0-core, else for 5-core (by default)')
    return parser.parse_args()


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), collections.defaultdict(list)
    for inter in inters:
        user, item, rating, timestamp, review = inter
        user2inters[user].append((user, item, rating, timestamp, review))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        his_items = set()
        for inter in user_inters:
            user, item, rating, timestamp, review = inter
            if item in his_items:
                continue
            his_items.add(item)
            new_inters[user].append(inter)
    return new_inters


def ensure_directory_exists(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


if __name__ == '__main__':
    args = parse_args()

    # 规范化所有路径
    args.input_path = os.path.normpath(args.input_path)
    args.output_path = os.path.normpath(args.output_path)
    args.seq_path = os.path.normpath(args.seq_path)

    if args.zero:
        args.input_path = args.input_path.replace('5core', '0core')
        args.output_path = args.output_path.replace('5core', '0core')
        args.seq_path = args.seq_path.replace('5core', '0core')
        
    # 确保输出目录存在
    ensure_directory_exists(args.output_path)
    ensure_directory_exists(args.seq_path)
    
    print(f"Processing with parameters:")
    print(f"  Input path: {args.input_path}")
    print(f"  Output path: {args.output_path}")
    print(f"  Sequence path: {args.seq_path}")

    all_files = os.listdir(args.input_path)
    for single_file in all_files:
        if not single_file.endswith('.csv'):
            continue
            
        prefix = single_file[:-len('.csv')]
        args.file_path = os.path.join(args.input_path, single_file)
        print(f"Processing file: {args.file_path}")

        inters = []
        with open(args.file_path, 'r') as file:
            for line in tqdm(file, 'Loading'):
                parts = line.strip().split(',')
                user_id, item_id, rating, timestamp = parts[0], parts[1], parts[2], int(parts[3])
                review = parts[4].strip('"') if len(parts) > 4 else ""
                inters.append((user_id, item_id, rating, timestamp, review))

        ordered_inters = make_inters_in_order(inters=inters)

        # 定义输出文件路径
        train_path = os.path.normpath(os.path.join(args.output_path, f'{prefix}.train.csv'))
        valid_path = os.path.normpath(os.path.join(args.output_path, f'{prefix}.valid.csv'))
        test_path = os.path.normpath(os.path.join(args.output_path, f'{prefix}.test.csv'))
        
        print(f"Writing to direct files:")
        print(f"  Train: {train_path}")
        print(f"  Valid: {valid_path}")
        print(f"  Test: {test_path}")

        # For direct recommendation
        train_file = open(train_path, 'w')
        valid_file = open(valid_path, 'w')
        test_file = open(test_path, 'w')

        for user in tqdm(ordered_inters, desc='Write direct files'):
            cur_inter = ordered_inters[user]
            for i in range(len(cur_inter)):
                ts = cur_inter[i][-1]
                out_file = None
                if ts >= test_timestamp:
                    out_file = test_file
                elif ts >= valid_timestamp:
                    out_file = valid_file
                else:
                    out_file = train_file
                out_file.write(f'{cur_inter[i][0]},{cur_inter[i][1]},{cur_inter[i][2]},{cur_inter[i][3]},"{cur_inter[i][4]}"\n')

        for file in [train_file, valid_file, test_file]:
            file.close()

        # 定义序列推荐的输出文件路径
        seq_train_path = os.path.normpath(os.path.join(args.seq_path, f'{prefix}.train.csv'))
        seq_valid_path = os.path.normpath(os.path.join(args.seq_path, f'{prefix}.valid.csv'))
        seq_test_path = os.path.normpath(os.path.join(args.seq_path, f'{prefix}.test.csv'))
        
        print(f"Writing to sequence files:")
        print(f"  Train: {seq_train_path}")
        print(f"  Valid: {seq_valid_path}")
        print(f"  Test: {seq_test_path}")

        # For sequential recommendation
        train_file = open(seq_train_path, 'w')
        valid_file = open(seq_valid_path, 'w')
        test_file = open(seq_test_path, 'w')

        for user in tqdm(ordered_inters, desc='Write seq files'):
            cur_inter = ordered_inters[user]
            for i in range(len(cur_inter)):
                ts = cur_inter[i][-1]
                cur_his = ' '.join([_[1] for _ in cur_inter[:i]])
                out_file = None
                if ts >= test_timestamp:
                    out_file = test_file
                elif ts >= valid_timestamp:
                    out_file = valid_file
                else:
                    out_file = train_file
                out_file.write(f'{cur_inter[i][0]},{cur_inter[i][1]},{cur_inter[i][2]},{cur_inter[i][3]},"{cur_inter[i][4]}",{cur_his}\n')

        for file in [train_file, valid_file, test_file]:
            file.close()
            
        print(f"Finished processing {single_file}")
