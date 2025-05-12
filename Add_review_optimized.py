import pandas as pd
import gzip
import orjson
import os
import argparse
from tqdm import tqdm
import ijson
from multiprocessing import Pool
from collections import defaultdict

def read_jsonl_gz_stream(file_path):
    """
    流式读取gzip压缩的jsonl文件，生成(user_id, parent_asin, review)元组
    """
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            try:
                review = orjson.loads(line)
                user_id = review.get('user_id')
                parent_asin = review.get('parent_asin')
                if user_id and parent_asin:
                    review_text = review.get('text', '')
                    review_title = review.get('title', '')
                    full_review = f"{review_title}: {review_text}" if review_title else review_text
                    yield (user_id, parent_asin, full_review)
            except orjson.JSONDecodeError:
                continue

def create_review_mapping_stream(reviews_file):
    """
    流式创建评论映射，使用更高效的数据结构
    """
    review_map = defaultdict(dict)
    for user_id, parent_asin, review in tqdm(read_jsonl_gz_stream(reviews_file), 
                                           desc="创建评论映射"):
        review_map[user_id][parent_asin] = review
    return review_map

def process_dataset(args):
    """
    处理单个数据集文件的函数，用于并行处理
    """
    input_file, output_file, review_map = args
    df = pd.read_csv(input_file)
    
    # 使用向量化操作替代iterrows
    df['review'] = df.apply(
        lambda row: review_map.get(row['user_id'], {}).get(row['item_id'], ""),
        axis=1
    )
    
    # 计算匹配率
    match_count = df['review'].ne("").sum()
    match_rate = match_count / len(df) * 100 if len(df) > 0 else 0
    
    # 保存结果
    df.to_csv(output_file, index=False)
    
    print(f"{os.path.basename(input_file)}: 找到 {match_count}/{len(df)} 条评论 ({match_rate:.2f}%)")
    return match_count, len(df)

def main(reviews_file, input_dir, output_dir):
    """
    优化后的主函数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"流式处理评论文件 {reviews_file}...")
    review_map = create_review_mapping_stream(reviews_file)
    print(f"创建了 {sum(len(v) for v in review_map.values())} 条评论映射")
    
    # 准备并行处理任务
    input_files = {
        'train': os.path.join(input_dir, 'Books.train.reduced.csv'),
        'valid': os.path.join(input_dir, 'Books.valid.reduced.csv'),
        'test': os.path.join(input_dir, 'Books.test.reduced.csv')
    }
    
    output_files = {
        'train': os.path.join(output_dir, 'Books.train.reduced.with_reviews.csv'),
        'valid': os.path.join(output_dir, 'Books.valid.reduced.with_reviews.csv'),
        'test': os.path.join(output_dir, 'Books.test.reduced.with_reviews.csv')
    }
    
    tasks = [
        (input_files[t], output_files[t], review_map)
        for t in ['train', 'valid', 'test']
        if os.path.exists(input_files[t])
    ]
    
    # 使用多进程并行处理
    with Pool(processes=min(3, os.cpu_count())) as pool:
        results = pool.map(process_dataset, tasks)
    
    # 计算总体统计
    total_missing = sum(r[1] - r[0] for r in results)
    total_records = sum(r[1] for r in results)
    overall_match_rate = (total_records - total_missing) / total_records * 100 if total_records > 0 else 0
    
    print(f"\n总体统计: 找到 {total_records - total_missing}/{total_records} 条评论 ({overall_match_rate:.2f}%)")
    
    if overall_match_rate < 50:
        print("\n警告: 评论匹配率较低，请检查数据集一致性")
    
    print("\n优化处理完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='优化版: 为数据集添加评论数据')
    parser.add_argument('--reviews', type=str, required=True,
                      help='包含评论数据的jsonl.gz文件路径')
    parser.add_argument('--input', type=str, default='processed_data/reduced/',
                      help='包含已缩减数据集的目录')
    parser.add_argument('--output', type=str, default='processed_data/reduced_with_reviews/',
                      help='输出添加了评论的数据集的目录')
    args = parser.parse_args()
    
    main(args.reviews, args.input, args.output)
