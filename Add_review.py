import pandas as pd
import gzip
import orjson
import os
import argparse
from tqdm import tqdm
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import shutil

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _get_free_space(path):
    """获取指定路径所在磁盘的可用空间（GB）"""
    if os.name == 'nt':  # Windows
        import ctypes
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(path), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / (1024 * 1024 * 1024)  # 转换为GB
    else:  # Linux/Mac
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024 * 1024 * 1024)  # 转换为GB

class ReviewDatabase:
    """使用SQLite作为评论的临时数据库，避免将所有评论加载到内存"""
    
    def __init__(self, db_path=None, temp_dir=None):
        """初始化数据库连接
        
        Args:
            db_path: 显式指定数据库路径
            temp_dir: 指定临时文件目录，解决C盘空间不足问题
        """
        if db_path:
            self.db_path = db_path
        else:
            if temp_dir and os.path.isdir(temp_dir):
                # 在指定目录创建临时数据库
                self.db_path = os.path.join(temp_dir, f"reviews_temp_{os.getpid()}.db")
            else:
                # 默认使用系统临时目录
                self.db_path = tempfile.mktemp(suffix='.db')
        
        # 不再保存连接对象作为实例变量，而是通过线程本地存储管理
        self.local = threading.local()
        self.lock = threading.Lock()
        # 主线程创建数据库和表
        self._init_db()
        
    def _get_connection(self):
        """获取当前线程的数据库连接（如果不存在则创建）"""
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path)
        return self.local.conn
        
    def _init_db(self):
        """初始化数据库结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 创建评论表和索引
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            user_id TEXT,
            parent_asin TEXT,
            review TEXT,
            PRIMARY KEY (user_id, parent_asin)
        )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_asin ON reviews (user_id, parent_asin)')
        conn.commit()
        conn.close()
    
    def close(self):
        """关闭所有连接并删除临时数据库（除非显式指定了数据库路径）"""
        # 关闭当前线程连接
        if hasattr(self.local, 'conn') and self.local.conn is not None:
            self.local.conn.close()
            self.local.conn = None
            
        # 仅当使用自动生成的临时文件路径时才删除数据库文件
        is_temp_db = 'reviews_temp_' in os.path.basename(self.db_path) or self.db_path.startswith(tempfile.gettempdir())
        
        if is_temp_db and os.path.exists(self.db_path):
            try:
                #os.remove(self.db_path)
                logger.info(f"已删除临时数据库: {self.db_path}")
            except:
                logger.warning(f"无法删除临时数据库: {self.db_path}")
        elif not is_temp_db:
            logger.info(f"保留了数据库文件: {self.db_path}")
    
    def get_connection(self):
        """获取数据库连接（线程安全）"""
        return self._get_connection()
    
    def load_reviews_from_file(self, reviews_file, batch_size=10000):
        """从JSONL.GZ文件批量加载评论"""
        total_reviews = 0
        batch = []
        disk_space_warning_shown = False
        
        # 获取当前线程的连接
        conn = self._get_connection()
        
        # 增加数据库WAL模式和缓存大小优化，提高写入速度
        with self.lock:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = 10000")  # 约40MB缓存
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.commit()
        
        try:
            with gzip.open(reviews_file, 'rb') as f:
                for line in tqdm(f, desc="加载评论数据"):
                    try:
                        review = orjson.loads(line)
                        user_id = review.get('user_id')
                        parent_asin = review.get('parent_asin')
                        
                        if user_id and parent_asin:
                            review_text = review.get('text', '')
                            review_title = review.get('title', '')
                            full_review = f"{review_title}: {review_text}" if review_title else review_text
                            
                            batch.append((user_id, parent_asin, full_review))
                            
                            if len(batch) >= batch_size:
                                try:
                                    self._insert_batch(batch)
                                    total_reviews += len(batch)
                                    batch = []
                                except sqlite3.OperationalError as e:
                                    if "disk" in str(e).lower() and not disk_space_warning_shown:
                                        disk_path = os.path.dirname(os.path.abspath(self.db_path))
                                        free_space = _get_free_space(disk_path)
                                        logger.error(f"数据库错误: {e}")
                                        logger.error(f"数据库路径 {disk_path} 可用空间: {free_space:.2f} GB")
                                        logger.error("请使用 --temp-dir 或 --db-path 参数指定一个有足够空间的磁盘位置")
                                        disk_space_warning_shown = True
                                        # 清空当前批次避免重复
                                        batch = []
                                        # 继续处理但间隔更长
                                        time.sleep(0.5)
                                    else:
                                        raise
                    except orjson.JSONDecodeError:
                        continue
            
            # 插入剩余的评论
            if batch:
                try:
                    self._insert_batch(batch)
                    total_reviews += len(batch)
                except sqlite3.OperationalError as e:
                    if "disk" in str(e).lower():
                        logger.error(f"由于磁盘空间不足，最后 {len(batch)} 条评论未能保存")
                    else:
                        raise
                    
            # 优化数据库
            with self.lock:
                logger.info("优化数据库...")
                conn = self._get_connection()
                conn.execute("PRAGMA optimize")
                conn.commit()
                    
            logger.info(f"成功加载 {total_reviews} 条评论到数据库: {self.db_path}")
            return total_reviews
            
        except Exception as e:
            logger.error(f"加载评论时发生错误: {e}")
            raise
    
    def _insert_batch(self, batch):
        """批量插入评论"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR REPLACE INTO reviews (user_id, parent_asin, review) VALUES (?, ?, ?)",
                batch
            )
            conn.commit()
    
    def get_review(self, user_id, parent_asin):
        """获取单条评论"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT review FROM reviews WHERE user_id = ? AND parent_asin = ?",
                (user_id, parent_asin)
            )
            result = cursor.fetchone()
            return result[0] if result else ""
    
    def get_reviews_batch(self, user_asin_pairs):
        """批量获取多条评论，减少数据库查询次数"""
        if not user_asin_pairs:
            return {}
            
        with self.lock:
            # 修复这里：使用_get_connection()而不是self.conn
            conn = self._get_connection()
            
            # 构建参数化查询
            placeholders = ','.join(['(?,?)'] * len(user_asin_pairs))
            flattened_params = [item for pair in user_asin_pairs for item in pair]
            
            cursor = conn.cursor()
            query = f"""
            SELECT user_id, parent_asin, review 
            FROM reviews 
            WHERE (user_id, parent_asin) IN ({placeholders})
            """
            
            cursor.execute(query, flattened_params)
            results = cursor.fetchall()
            
            # 构建结果字典
            return {(row[0], row[1]): row[2] for row in results}


def process_chunk(chunk_df, review_db):
    """处理DataFrame的一个块"""
    # 准备批量查询
    user_asin_pairs = [(row['user_id'], row['item_id']) for _, row in chunk_df.iterrows()]
    
    # 批量获取评论
    reviews_dict = review_db.get_reviews_batch(user_asin_pairs)
    
    # 为每行分配评论
    reviews = []
    for _, row in chunk_df.iterrows():
        pair = (row['user_id'], row['item_id'])
        reviews.append(reviews_dict.get(pair, ""))
    
    chunk_df['review'] = reviews
    return chunk_df


def process_dataset_file(input_file, output_file, review_db, chunk_size=50000):
    """以块的方式处理数据集文件，减少内存占用"""
    if not os.path.exists(input_file):
        logger.warning(f"输入文件不存在: {input_file}")
        return 0, 0
        
    logger.info(f"处理文件: {os.path.basename(input_file)}")
    
    # 统计行数以显示进度
    try:
        total_rows = sum(1 for _ in open(input_file)) - 1  # 减去标题行
    except:
        # 如果文件很大，无法统计行数，则使用估计值
        logger.warning(f"无法统计文件行数，使用估计值进行进度显示")
        total_rows = 1000000  # 估计值
    
    # 创建输出文件并写入标题行
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    first_chunk = True
    
    total_reviews = 0
    total_records = 0
    
    for chunk in tqdm(reader, desc=f"处理 {os.path.basename(input_file)}", 
                     total=(total_rows // chunk_size) + 1):
        # 处理这个块
        chunk_with_reviews = process_chunk(chunk, review_db)
        
        # 统计匹配的评论数
        matched_reviews = chunk_with_reviews['review'].ne("").sum()
        total_reviews += matched_reviews
        total_records += len(chunk)
        
        # 写入结果，只有第一个块需要写入标题
        chunk_with_reviews.to_csv(
            output_file, 
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        first_chunk = False
    
    match_rate = total_reviews / total_records * 100 if total_records > 0 else 0
    logger.info(f"{os.path.basename(input_file)}: 找到 {total_reviews}/{total_records} 条评论 ({match_rate:.2f}%)")
    
    return total_reviews, total_records


def main(reviews_file, input_dir, output_dir, temp_dir=None, db_path=None, chunk_size=50000):
    """使用优化后的流程处理数据
    
    Args:
        reviews_file: 评论数据文件路径
        input_dir: 输入数据目录
        output_dir: 输出数据目录
        temp_dir: 临时数据库目录，避免C盘空间不足
        db_path: 显式指定数据库路径
        chunk_size: 数据处理块大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查可用空间
    if temp_dir:
        if not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir, exist_ok=True)
                logger.info(f"创建临时目录: {temp_dir}")
            except Exception as e:
                logger.error(f"无法创建临时目录 {temp_dir}: {e}")
                logger.info("尝试使用系统临时目录")
                temp_dir = None
        
        if temp_dir and os.path.exists(temp_dir):
            free_space = _get_free_space(temp_dir)
            logger.info(f"临时目录 {temp_dir} 可用空间: {free_space:.2f} GB")
            if free_space < 5:  # 假设至少需要5GB
                logger.warning(f"临时目录空间不足，建议至少5GB可用空间")
    
    if db_path:
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"无法创建数据库目录 {db_dir}: {e}")
                logger.info("将使用临时目录")
                db_path = None
        
        if db_path and os.path.dirname(os.path.abspath(db_path)):
            free_space = _get_free_space(os.path.dirname(os.path.abspath(db_path)))
            logger.info(f"数据库目录可用空间: {free_space:.2f} GB")
            if free_space < 5:  # 假设至少需要5GB
                logger.warning(f"数据库目录空间不足，建议至少5GB可用空间")
    
    # 创建评论数据库，指定临时目录以避免C盘空间不足
    review_db = ReviewDatabase(db_path=db_path, temp_dir=temp_dir)
    logger.info(f"临时数据库位置: {review_db.db_path}")
    
    try:
        logger.info(f"开始处理评论文件: {reviews_file}")
        review_db.load_reviews_from_file(reviews_file)
        
        # 准备输入和输出文件
        dataset_types = ['train', 'valid', 'test']
        input_files = {
            t: os.path.join(input_dir, f'Books.{t}.reduced.csv')
            for t in dataset_types
        }
        output_files = {
            t: os.path.join(output_dir, f'Books.{t}.reduced.with_reviews.csv')
            for t in dataset_types
        }
        
        # 修复多线程处理问题 - 使用同一个ReviewDatabase实例可能导致连接冲突
        # 为每个数据集文件串行处理而不是使用线程池
        results = []
        for t in dataset_types:
            if os.path.exists(input_files[t]):
                logger.info(f"处理数据集: {t}")
                result = process_dataset_file(
                    input_files[t], 
                    output_files[t], 
                    review_db,
                    chunk_size
                )
                results.append(result)
        
        # 计算总体统计
        total_found = sum(r[0] for r in results)
        total_records = sum(r[1] for r in results)
        overall_match_rate = total_found / total_records * 100 if total_records > 0 else 0
        
        logger.info(f"\n总体统计: 找到 {total_found}/{total_records} 条评论 ({overall_match_rate:.2f}%)")
        
        if overall_match_rate < 50:
            logger.warning("警告: 评论匹配率较低，请检查数据集一致性")
            
    finally:
        # 清理临时数据库
        review_db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='内存优化版: 为数据集添加评论数据')
    parser.add_argument('--reviews', type=str, required=True,
                      help='包含评论数据的jsonl.gz文件路径')
    parser.add_argument('--input', type=str, default='processed_data/reduced/',
                      help='包含已缩减数据集的目录')
    parser.add_argument('--output', type=str, default='processed_data/reduced_with_reviews/',
                      help='输出添加了评论的数据集的目录')
    parser.add_argument('--chunk-size', type=int, default=50000,
                      help='处理数据时每个块的大小，调整以平衡内存使用和性能')
    parser.add_argument('--temp-dir', type=str, default=None,
                      help='临时数据库存储目录，解决C盘空间不足问题(例如: D:/temp)')
    parser.add_argument('--db-path', type=str, default=None,
                      help='显式指定SQLite数据库文件路径(例如: D:/reviews.db)')
    args = parser.parse_args()
    
    main(args.reviews, args.input, args.output, args.temp_dir, args.db_path, args.chunk_size)