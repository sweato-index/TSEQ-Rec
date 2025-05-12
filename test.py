import pandas as pd
meta_data_path = "D:\\Amazon_Reviews\\books_metadata.csv"
meta_df = pd.read_csv(meta_data_path, dtype={
        'parent_asin': str,
        'title': str,
        'average_rating': float,
        'rating_number': float,
        'features': str,
        'price': float
    }, low_memory=False)
sample_ratio = 0.0001
meta_df = meta_df.sample(frac=sample_ratio, random_state=42)
meta_df.to_csv('D:\\Amazon_Reviews\\books_metadata_small.csv', index=False)