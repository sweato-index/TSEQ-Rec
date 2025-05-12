import pandas as pd
import numpy as np
import os
from collections import Counter
import random
import json
random.seed(42)  # For reproducibility

# Define paths
input_path = "processed_data/5core/timestamp_w_his/"
output_path = "processed_data/reduced/"
metadata_path = "books_metadata.csv"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# 1. Load and process interaction data
def load_interaction_data():
    train_df = pd.read_csv(input_path + "Books.train.csv", header=None)
    valid_df = pd.read_csv(input_path + "Books.valid.csv", header=None)
    test_df = pd.read_csv(input_path + "Books.test.csv", header=None)
    
    # Add column names
    columns = ['user_id', 'item_id', 'timestamp', 'rating', 'review', 'history']
    train_df.columns = columns
    valid_df.columns = columns
    test_df.columns = columns
    
    print(f"Original data sizes: Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    return train_df, valid_df, test_df

# 2. Load and process metadata
def load_metadata():
    meta_df = pd.read_csv(metadata_path)
    print(f"Original metadata size: {len(meta_df)}")
    
    # Filter metadata to keep only items with valid features and prices
    valid_meta = meta_df[(meta_df['features'].notna() & meta_df['features'] != '') & 
                         (meta_df['price'].notna() & meta_df['price'] > 0)]
    
    print(f"Metadata after filtering invalid features/prices: {len(valid_meta)}")
    return valid_meta

# 3. Identify all unique items in interaction data
def get_all_interaction_items(train_df, valid_df, test_df):
    all_items = set(train_df['item_id'].unique())
    all_items.update(valid_df['item_id'].unique())
    all_items.update(test_df['item_id'].unique())
    
    # Extract items from history lists as well
    for df in [train_df, valid_df, test_df]:
        for history in df['history']:
            if isinstance(history, str):
                all_items.update(history.split())
    
    return all_items

# 4. Sample users while maintaining user distribution
def sample_users(train_df, valid_df, test_df, target_size=0.02):
    # Count interactions per user across all sets
    all_users = pd.concat([train_df['user_id'], valid_df['user_id'], test_df['user_id']])
    user_counts = Counter(all_users)
    
    # Group users by activity level
    activity_groups = {
        'low': [], 'medium': [], 'high': []
    }
    
    for user, count in user_counts.items():
        if count <= 5:
            activity_groups['low'].append(user)
        elif count <= 20:
            activity_groups['medium'].append(user)
        else:
            activity_groups['high'].append(user)
    
    # Sample from each group
    sampled_users = []
    for group, users in activity_groups.items():
        sample_size = int(len(users) * target_size)
        # Ensure we have at least some users from each group for diversity
        if sample_size == 0 and users:
            sample_size = min(10, len(users))
        if sample_size > 0:
            sampled_users.extend(random.sample(users, sample_size))
    
    # If the total number of users is still too large, further reduce
    if len(sampled_users) > 10000:
        sampled_users = random.sample(sampled_users, 10000)
    
    return set(sampled_users)

# 5. Filter datasets based on sampled users and items
def filter_datasets(train_df, valid_df, test_df, selected_users, valid_meta):
    valid_items = set(valid_meta['parent_asin'])
    
    # Filter by users
    train_reduced = train_df[train_df['user_id'].isin(selected_users)]
    valid_reduced = valid_df[valid_df['user_id'].isin(selected_users)]
    test_reduced = test_df[test_df['user_id'].isin(selected_users)]
    
    # Now filter by items (only keep interactions with items in valid metadata)
    train_reduced = train_reduced[train_reduced['item_id'].isin(valid_items)]
    valid_reduced = valid_reduced[valid_reduced['item_id'].isin(valid_items)]
    test_reduced = test_reduced[test_reduced['item_id'].isin(valid_items)]
    
    # Ensure review column is preserved
    train_reduced['review'] = train_reduced['review'].fillna('')
    valid_reduced['review'] = valid_reduced['review'].fillna('')
    test_reduced['review'] = test_reduced['review'].fillna('')
    
    # Filter history to only contain valid items
    def filter_history(history):
        if isinstance(history, str):
            items = history.split()
            filtered_items = [item for item in items if item in valid_items]
            return ' '.join(filtered_items)
        return ''
    
    train_reduced['history'] = train_reduced['history'].apply(filter_history)
    valid_reduced['history'] = valid_reduced['history'].apply(filter_history)
    test_reduced['history'] = test_reduced['history'].apply(filter_history)
    
    return train_reduced, valid_reduced, test_reduced

# 6. Further reduce the metadata
def reduce_metadata(meta_df, all_interaction_items, extra_factor=1.2):
    # Keep all items that appear in interactions
    meta_with_interactions = meta_df[meta_df['parent_asin'].isin(all_interaction_items)]
    
    # Sample from remaining items
    meta_without_interactions = meta_df[~meta_df['parent_asin'].isin(all_interaction_items)]
    
    # Calculate how many additional items to keep
    additional_items_count = int(len(all_interaction_items) * (extra_factor - 1))
    additional_items_count = min(additional_items_count, len(meta_without_interactions))
    
    # Sample additional items
    if additional_items_count > 0:
        sampled_additional = meta_without_interactions.sample(n=additional_items_count, random_state=42)
        reduced_meta = pd.concat([meta_with_interactions, sampled_additional])
    else:
        reduced_meta = meta_with_interactions
    
    return reduced_meta

# 7. Main function to execute the reduction process
def main():
    # Load data
    train_df, valid_df, test_df = load_interaction_data()
    meta_df = load_metadata()
    
    # Print original scale information
    print(f"\nOriginal scale:")
    print(f"Total interactions: {len(train_df) + len(valid_df) + len(test_df)}")
    print(f"Unique users: {len(set(pd.concat([train_df['user_id'], valid_df['user_id'], test_df['user_id']])))}")
    all_items = set(train_df['item_id']) | set(valid_df['item_id']) | set(test_df['item_id'])
    print(f"Unique items in direct interactions: {len(all_items)}")
    print(f"Total items in metadata: {len(meta_df)}")
    
    # Sample users (reduce to 2% of original to target ~10k scale)
    sampled_users = sample_users(train_df, valid_df, test_df, target_size=0.02)
    print(f"Sampled {len(sampled_users)} users from original {len(set(pd.concat([train_df['user_id'], valid_df['user_id'], test_df['user_id']])))} users")
    
    # Filter datasets based on users and valid metadata items
    train_reduced, valid_reduced, test_reduced = filter_datasets(train_df, valid_df, test_df, sampled_users, meta_df)
    
    # Get all items in the reduced interaction datasets
    all_interaction_items = set(train_reduced['item_id'].unique())
    all_interaction_items.update(valid_reduced['item_id'].unique())
    all_interaction_items.update(test_reduced['item_id'].unique())
    
    # Extract items from history lists as well
    for df in [train_reduced, valid_reduced, test_reduced]:
        for history in df['history']:
            if isinstance(history, str):
                all_interaction_items.update(history.split())
    
    print(f"Unique items in reduced interaction data: {len(all_interaction_items)}")
    
    # Reduce metadata with smaller extra factor to keep total items around 10k
    reduced_meta = reduce_metadata(meta_df, all_interaction_items, extra_factor=1.2)
    
    # Save reduced datasets
    train_reduced.to_csv(output_path + "Books.train.reduced.csv", index=False)
    valid_reduced.to_csv(output_path + "Books.valid.reduced.csv", index=False)
    test_reduced.to_csv(output_path + "Books.test.reduced.csv", index=False)
    reduced_meta.to_csv(output_path + "books_metadata.reduced.csv", index=False)
    
    # Print statistics
    print("\nReduction Results:")
    print(f"Train: {len(train_df)} → {len(train_reduced)} ({len(train_reduced)/len(train_df)*100:.2f}%)")
    print(f"Valid: {len(valid_df)} → {len(valid_reduced)} ({len(valid_reduced)/len(valid_df)*100:.2f}%)")
    print(f"Test: {len(test_df)} → {len(test_reduced)} ({len(test_reduced)/len(test_df)*100:.2f}%)")
    print(f"Metadata: {len(meta_df)} → {len(reduced_meta)} ({len(reduced_meta)/len(meta_df)*100:.2f}%)")
    
    # Check distribution of ratings to ensure balanced dataset
    print("\nRating distributions:")
    print("Original train:", train_df['rating'].value_counts(normalize=True))
    print("Reduced train:", train_reduced['rating'].value_counts(normalize=True))
    
    # Output unique users and items counts
    print(f"\nUnique users in reduced dataset: {len(set(train_reduced['user_id']) | set(valid_reduced['user_id']) | set(test_reduced['user_id']))}")
    print(f"Unique items in reduced dataset: {len(all_interaction_items)}")
    print(f"Total items in reduced metadata: {len(reduced_meta)}")
    
    # Check if we have any users or items with too few interactions
    user_counts = Counter(pd.concat([train_reduced['user_id'], valid_reduced['user_id'], test_reduced['user_id']]))
    item_counts = Counter(pd.concat([train_reduced['item_id'], valid_reduced['item_id'], test_reduced['item_id']]))
    
    print(f"\nUsers with only 1 interaction: {sum(1 for u, c in user_counts.items() if c == 1)}")
    print(f"Items with only 1 interaction: {sum(1 for i, c in item_counts.items() if c == 1)}")
    
    # Print detailed user interaction distribution
    user_interaction_counts = sorted(user_counts.values())
    if user_interaction_counts:
        print(f"\nUser interaction distribution:")
        print(f"Min interactions per user: {min(user_interaction_counts)}")
        print(f"Max interactions per user: {max(user_interaction_counts)}")
        print(f"Mean interactions per user: {sum(user_interaction_counts)/len(user_interaction_counts):.2f}")
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            idx = int(len(user_interaction_counts) * p / 100)
            print(f"{p}th percentile: {user_interaction_counts[idx]}")
    
    # Print target scale achievement
    print(f"\nTarget scale achievement:")
    print(f"Target was approximately 10,000 scale")
    print(f"Achieved: {len(user_counts)} users, {len(all_interaction_items)} interaction items, {len(reduced_meta)} total items")

if __name__ == "__main__":
    main()
