import os
import pandas as pd
import numpy as np
import yaml
import pickle
from tqdm import tqdm

class RecBoleDataProcessor:
    def __init__(self, output_dir='recbole_data/books'):
        """初始化处理器"""
        self.output_dir = output_dir
        self.user_id_map = {}  # 原始user_id到索引的映射
        self.item_id_map = {}  # 原始item_id到索引的映射
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def process_interaction_data(self, train_file, valid_file, test_file):
        """处理交互数据并生成RecBole交互文件"""
        print("处理交互数据...")
        
        # 处理CSV文件
        def read_interaction_csv(file_path):
            print(f"读取文件: {file_path}")
            try:
                # 尝试直接读取，适用于带标题的CSV
                df = pd.read_csv(file_path)
                # 检查是否包含预期的列
                if 'user_id' not in df.columns or 'item_id' not in df.columns:
                    # 如果列名不符合预期，可能是没有标题的CSV
                    column_names = ['user_id', 'item_id', 'timestamp', 'rating', 'history', 'review']
                    df = pd.read_csv(file_path, names=column_names, header=None)
            except Exception as e:
                print(f"读取错误: {e}")
                # 尝试使用指定列名读取
                column_names = ['user_id', 'item_id', 'timestamp', 'rating', 'history', 'review']
                df = pd.read_csv(file_path, names=column_names, header=None)
            
            return df
        
        # 读取所有交互数据
        train_df = read_interaction_csv(train_file)
        valid_df = read_interaction_csv(valid_file)  
        test_df = read_interaction_csv(test_file)
        
        # 打印数据样例，以便验证
        print("训练集数据样例:")
        print(train_df.head(2))
        
        # 合并所有数据以获取完整的用户和物品集合
        all_df = pd.concat([train_df, valid_df, test_df], axis=0)
        
        # 获取唯一的用户和物品ID
        unique_users = all_df['user_id'].unique()
        unique_items = all_df['item_id'].unique()
        
        print(f"找到 {len(unique_users)} 个唯一用户和 {len(unique_items)} 个唯一物品")
        
        # 创建ID映射
        for idx, user_id in enumerate(unique_users):
            self.user_id_map[user_id] = idx + 1  # 从1开始编号
        
        for idx, item_id in enumerate(unique_items):
            self.item_id_map[item_id] = idx + 1  # 从1开始编号
        
        # 保存ID映射
        self._save_id_mapping()
        
        # 转换交互数据
        self._convert_interaction_file(train_df, os.path.join(self.output_dir, 'books.train.inter'))
        self._convert_interaction_file(valid_df, os.path.join(self.output_dir, 'books.valid.inter'))
        self._convert_interaction_file(test_df, os.path.join(self.output_dir, 'books.test.inter'))
    
    def process_item_data(self, item_file):
        """处理物品元数据并生成RecBole物品文件"""
        print("处理物品元数据...")
        
        # 读取物品元数据
        item_df = pd.read_csv(item_file)
        item_df = item_df.rename(columns={'parent_asin': 'item_id'})
        
        # 仅处理在交互数据中出现的物品
        item_df = item_df[item_df['item_id'].isin(self.item_id_map.keys())]
        
        # 转换物品数据
        self._convert_item_file(item_df, os.path.join(self.output_dir, 'books.item'))
    
    def create_config_file(self):
        """创建RecBole配置文件"""
        print("创建配置文件...")
        
        # 确保字段顺序与数据处理时保持一致
        config = {
            # 基本字段
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'RATING_FIELD': 'rating',
            'TIME_FIELD': 'timestamp',
            
            # 自定义字段
            'review_field': 'review',
            'history_field': 'history',
            'title_field': 'title',
            'price_field': 'price',
            'average_rating_field': 'average_rating',
            'rating_number_field': 'rating_number',
            'features_field': 'features',
            
            # 加载配置
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating', 'timestamp', 'history', 'review'],
                'item': ['item_id', 'title', 'average_rating', 'rating_number', 'features', 'price']
            },
            
            # 数据类型
            'numerical_features': ['rating', 'timestamp', 'average_rating', 'rating_number', 'price'],
            'token_features': ['history', 'review', 'features'],
            
            # 其他配置
            'ITEM_LIST_LENGTH_FIELD': 'history',
            'PRICE_FIELD': 'price',
            
            # 数据集信息
            'field_separator': '\t'
        }
        
        # 保存yaml配置文件
        with open(os.path.join(self.output_dir, 'books.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"配置文件已保存至 {os.path.join(self.output_dir, 'books.yaml')}")
    
    def _convert_interaction_file(self, df, output_file):
        """转换交互数据为RecBole格式"""
        # 创建一个新的DataFrame用于输出
        output_df = pd.DataFrame()
        
        # 转换用户ID和物品ID
        output_df['user_id'] = df['user_id'].map(self.user_id_map)
        output_df['item_id'] = df['item_id'].map(self.item_id_map)
        
        # 先检查一下数据类型和值，以确保正确处理
        print("转换前数据类型:")
        print(df.dtypes)
        print("转换前数据样例:")
        print(df[['rating', 'timestamp']].head(2))
        
        # 复制其他字段，确保顺序正确
        # RecBole要求的字段顺序：user_id, item_id, rating, timestamp, ...
        output_df['rating'] = df['rating'].astype('float')  # 确保评分为浮点数
        output_df['timestamp'] = df['timestamp'].astype('int64')  # 确保时间戳为整数
        output_df['history'] = df['history']  # 历史记录
        output_df['review'] = df['review']  # 评论
        
        # 检查转换后的数据
        print("转换后数据样例:")
        print(output_df[['rating', 'timestamp']].head(2))
        
        # 保存为csv文件（RecBole使用的格式）
        output_df.to_csv(output_file, index=False, sep='\t')
        
        print(f"交互文件已保存至 {output_file}")
    
    def _convert_item_file(self, df, output_file):
        """转换物品数据为RecBole格式"""
        # 创建一个新的DataFrame用于输出
        output_df = pd.DataFrame()
        
        # 转换物品ID
        output_df['item_id'] = df['item_id'].map(self.item_id_map)
        
        # 复制其他字段
        output_df['title'] = df['title']
        output_df['average_rating'] = df['average_rating']
        output_df['rating_number'] = df['rating_number']
        output_df['features'] = df['features']
        output_df['price'] = df['price']
        
        # 保存为csv文件（RecBole使用的格式）
        output_df.to_csv(output_file, index=False, sep='\t')
        
        print(f"物品文件已保存至 {output_file}")
    
    def _save_id_mapping(self):
        """保存ID映射关系"""
        # 创建反向映射（从索引到原始ID）
        user_idx_to_id = {v: k for k, v in self.user_id_map.items()}
        item_idx_to_id = {v: k for k, v in self.item_id_map.items()}
        
        # 保存为pickle文件
        with open(os.path.join(self.output_dir, 'user_id_map.pkl'), 'wb') as f:
            pickle.dump({'id2index': self.user_id_map, 'index2id': user_idx_to_id}, f)
        
        with open(os.path.join(self.output_dir, 'item_id_map.pkl'), 'wb') as f:
            pickle.dump({'id2index': self.item_id_map, 'index2id': item_idx_to_id}, f)
        
        # 同时保存为CSV以便查看
        pd.DataFrame({
            'original_user_id': list(self.user_id_map.keys()),
            'recbole_index': list(self.user_id_map.values())
        }).to_csv(os.path.join(self.output_dir, 'user_id_map.csv'), index=False)
        
        pd.DataFrame({
            'original_item_id': list(self.item_id_map.keys()),
            'recbole_index': list(self.item_id_map.values())
        }).to_csv(os.path.join(self.output_dir, 'item_id_map.csv'), index=False)
        
        print(f"ID映射已保存至 {self.output_dir}")

if __name__ == "__main__":
    # 定义输入文件路径
    train_file = "processed_data/reduced_with_reviews/Books.train.reduced.with_reviews.csv"
    valid_file = "processed_data/reduced_with_reviews/Books.valid.reduced.with_reviews.csv"
    test_file = "processed_data/reduced_with_reviews/Books.test.reduced.with_reviews.csv"
    item_file = "processed_data/reduced/books_metadata.reduced.csv"
    
    # 检查文件是否存在
    for file_path in [train_file, valid_file, test_file, item_file]:
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在!")
    
    # 创建处理器
    processor = RecBoleDataProcessor()
    
    # 处理数据
    processor.process_interaction_data(train_file, valid_file, test_file)
    processor.process_item_data(item_file)
    processor.create_config_file()
    
    print("数据处理完成！数据已保存至 recbole_data/books 目录")