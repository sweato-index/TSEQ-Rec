import gzip
import json
import csv
import os

# 输入和输出文件
input_file = 'meta_Books.jsonl.gz'
output_file = 'books_metadata.csv'

# 要提取的字段
fields = ['parent_asin', 'title', 'average_rating', 'rating_number', 'features', 'price']

# 统计信息
total_records = 0
processed_records = 0
missing_values = {field: 0 for field in fields}

def process_features(features):
    """处理features字段，将列表转换为字符串"""
    if not features or not isinstance(features, list):
        return ""
    return "|||".join(features)  # 使用特殊分隔符连接多个特性

# 打开输出CSV文件
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    # 创建CSV写入器
    csv_writer = csv.writer(csvfile)
    
    # 写入标题行
    csv_writer.writerow(fields)
    
    # 打开并处理gzip压缩的JSONL文件
    try:
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            for line in f:
                total_records += 1
                
                try:
                    # 解析JSON行
                    record = json.loads(line.strip())
                    
                    # 准备输出行
                    row = []
                    valid_record = True
                    
                    for field in fields:
                        value = record.get(field)
                        
                        # 处理特殊字段和异常值
                        if field == 'average_rating' and (value is None or not isinstance(value, (int, float))):
                            row.append("")
                            missing_values[field] += 1
                        elif field == 'rating_number' and (value is None or not isinstance(value, (int, float))):
                            row.append("")
                            missing_values[field] += 1
                        elif field == 'price' and (value is None or value == '-' or not isinstance(value, (int, float))):
                            row.append("")
                            missing_values[field] += 1
                        elif field == 'features':
                            processed_value = process_features(value)
                            row.append(processed_value)
                            if not processed_value:
                                missing_values[field] += 1
                        elif value is None:
                            row.append("")
                            missing_values[field] += 1
                        else:
                            row.append(value)
                    
                    # 写入CSV行
                    csv_writer.writerow(row)
                    processed_records += 1
                    
                    # 每处理1000条记录显示一次进度
                    if processed_records % 10000 == 0:
                        print(f"已处理 {processed_records} 条记录...")
                        
                except json.JSONDecodeError as e:
                    print(f"跳过无效的JSON行: {e}")
                except Exception as e:
                    print(f"处理记录时出错: {e}")
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_file}'")
    except Exception as e:
        print(f"处理文件时出错: {e}")

# 打印统计信息
print("\n处理完成!")
print(f"总记录数: {total_records}")
print(f"成功处理的记录数: {processed_records}")
print("\n缺失值统计:")
for field, count in missing_values.items():
    print(f"  {field}: {count} ({count/processed_records*100:.2f}% 为空或异常)")

print(f"\n输出文件保存为: {os.path.abspath(output_file)}")