import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Set figure size for better visibility
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def analyze_symptom_dataset():
    """Analyze the Symptom2Disease dataset"""
    print("=== SYMPTOM2DISEASE DATASET ANALYSIS ===")
    print("=" * 50)
    
    # Load the dataset
    df_symptoms = pd.read_csv('Symptom2Disease.csv')
    
    print(f"Dataset Shape: {df_symptoms.shape}")
    print(f"Columns: {list(df_symptoms.columns)}")
    
    # Basic Statistics
    disease_counts = df_symptoms['label'].value_counts()
    df_symptoms['text_length'] = df_symptoms['text'].str.len()
    df_symptoms['word_count'] = df_symptoms['text'].str.split().str.len()
    
    print(f"\nTotal number of diseases: {len(disease_counts)}")
    print(f"Total number of samples: {len(df_symptoms)}")
    
    print(f"\nText Length Statistics:")
    print(f"Mean: {df_symptoms['text_length'].mean():.2f} characters")
    print(f"Median: {df_symptoms['text_length'].median():.2f} characters")
    print(f"Min: {df_symptoms['text_length'].min():.2f} characters")
    print(f"Max: {df_symptoms['text_length'].max():.2f} characters")
    print(f"Range: {df_symptoms['text_length'].max() - df_symptoms['text_length'].min():.2f} characters")
    print(f"Standard Deviation: {df_symptoms['text_length'].std():.2f} characters")
    
    # Visualization 1: Disease Distribution (Bar Chart)
    plt.figure(figsize=(14, 8))
    disease_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Diseases in Symptom2Disease Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Disease Type', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('symptom_disease_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 2: Text Length Distribution (Histogram)
    plt.figure(figsize=(12, 6))
    plt.hist(df_symptoms['text_length'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Text Length in Symptom Descriptions', fontsize=16, fontweight='bold')
    plt.xlabel('Text Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Calculate statistics
    mean_val = df_symptoms['text_length'].mean()
    median_val = df_symptoms['text_length'].median()
    std_val = df_symptoms['text_length'].std()
    mode_val = df_symptoms['text_length'].mode().iloc[0] if len(df_symptoms['text_length'].mode()) > 0 else mean_val
    
    # Add vertical lines for statistics
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.0f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {median_val:.0f}')
    plt.axvline(mode_val, color='purple', linestyle='--', linewidth=2, 
                label=f'Mode: {mode_val:.0f}')
    
    # Add standard deviation lines
    plt.axvline(mean_val + std_val, color='blue', linestyle=':', linewidth=1.5, 
                label=f'Mean + Std: {mean_val + std_val:.0f}')
    plt.axvline(mean_val - std_val, color='blue', linestyle=':', linewidth=1.5, 
                label=f'Mean - Std: {mean_val - std_val:.0f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('symptom_text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 3: Disease Distribution (Pie Chart)
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(disease_counts)))
    plt.pie(disease_counts.values, labels=disease_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Proportion of Diseases in Dataset', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.savefig('symptom_disease_pie.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 4: Word Count Distribution (Box Plot)
    plt.figure(figsize=(14, 8))
    df_symptoms.boxplot(column='word_count', by='label', figsize=(14, 8))
    plt.title('Word Count Distribution by Disease Type', fontsize=16, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Disease Type', fontsize=12)
    plt.ylabel('Word Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('symptom_word_count_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 5: Text Length vs Word Count (Scatter Plot)
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_symptoms['label'].unique())))
    
    for i, disease in enumerate(df_symptoms['label'].unique()):
        subset = df_symptoms[df_symptoms['label'] == disease]
        plt.scatter(subset['text_length'], subset['word_count'], 
                    label=disease, alpha=0.6, s=50, color=colors[i])
    
    plt.title('Relationship between Text Length and Word Count by Disease', fontsize=16, fontweight='bold')
    plt.xlabel('Text Length (characters)', fontsize=12)
    plt.ylabel('Word Count', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    correlation = df_symptoms['text_length'].corr(df_symptoms['word_count'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
    
    plt.tight_layout()
    plt.savefig('symptom_correlation_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 6: Statistical Summary (Bar Chart)
    plt.figure(figsize=(14, 8))
    
    # Calculate all statistics
    mean_val = df_symptoms['text_length'].mean()
    median_val = df_symptoms['text_length'].median()
    std_val = df_symptoms['text_length'].std()
    mode_val = df_symptoms['text_length'].mode().iloc[0] if len(df_symptoms['text_length'].mode()) > 0 else mean_val
    
    # Create bar chart
    stats_names = ['Mean', 'Median', 'Mode', 'Std Dev', 'Mean + Std', 'Mean - Std']
    stats_values = [mean_val, median_val, mode_val, std_val, mean_val + std_val, mean_val - std_val]
    colors = ['red', 'orange', 'purple', 'green', 'blue', 'blue']
    
    bars = plt.bar(stats_names, stats_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, stats_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Statistical Summary of Text Length Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Statistical Measures', fontsize=12)
    plt.ylabel('Text Length (characters)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('symptom_statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_symptoms, disease_counts

def analyze_image_dataset():
    """Analyze the Skin Disease Image Dataset"""
    print("\n=== SKIN DISEASE IMAGE DATASET ANALYSIS ===")
    print("=" * 50)
    
    # Get the list of disease categories
    data_dir = "Skin_Diseases"
    disease_categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    disease_categories.sort()
    
    print(f"Found {len(disease_categories)} disease categories:")
    for i, category in enumerate(disease_categories, 1):
        print(f"{i}. {category}")
    
    # Count images in each category
    image_counts = {}
    total_images = 0
    
    for category in disease_categories:
        category_path = os.path.join(data_dir, category)
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_counts[category] = len(image_files)
        total_images += len(image_files)
    
    print(f"\nTotal number of images: {total_images}")
    
    # Basic Statistics
    image_counts_list = list(image_counts.values())
    print(f"\nMean images per category: {np.mean(image_counts_list):.2f}")
    print(f"Median images per category: {np.median(image_counts_list):.2f}")
    print(f"Min images per category: {min(image_counts_list)}")
    print(f"Max images per category: {max(image_counts_list)}")
    print(f"Range: {max(image_counts_list) - min(image_counts_list)}")
    print(f"Standard Deviation: {np.std(image_counts_list):.2f}")
    
    # Create DataFrame for easier analysis
    df_images = pd.DataFrame(list(image_counts.items()), columns=['Disease', 'Image_Count'])
    df_images = df_images.sort_values('Image_Count', ascending=False)
    
    # Visualization 1: Image Distribution by Disease (Bar Chart)
    plt.figure(figsize=(16, 8))
    bars = plt.bar(range(len(df_images)), df_images['Image_Count'], 
                   color='lightcoral', edgecolor='black', alpha=0.8)
    plt.title('Distribution of Images by Skin Disease Category', fontsize=16, fontweight='bold')
    plt.xlabel('Disease Categories', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(range(len(df_images)), df_images['Disease'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('image_disease_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 2: Image Count Distribution (Histogram)
    plt.figure(figsize=(12, 6))
    plt.hist(df_images['Image_Count'], bins=10, color='lightblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Image Counts per Disease Category', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Images per Category', fontsize=12)
    plt.ylabel('Number of Disease Categories', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Calculate statistics
    mean_val = df_images['Image_Count'].mean()
    median_val = df_images['Image_Count'].median()
    std_val = df_images['Image_Count'].std()
    mode_val = df_images['Image_Count'].mode().iloc[0] if len(df_images['Image_Count'].mode()) > 0 else mean_val
    
    # Add vertical lines for statistics
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_val:.0f}')
    plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {median_val:.0f}')
    plt.axvline(mode_val, color='purple', linestyle='--', linewidth=2, 
                label=f'Mode: {mode_val:.0f}')
    
    # Add standard deviation lines
    plt.axvline(mean_val + std_val, color='blue', linestyle=':', linewidth=1.5, 
                label=f'Mean + Std: {mean_val + std_val:.0f}')
    plt.axvline(mean_val - std_val, color='blue', linestyle=':', linewidth=1.5, 
                label=f'Mean - Std: {mean_val - std_val:.0f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('image_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 3: Disease Categories (Pie Chart)
    plt.figure(figsize=(14, 10))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(df_images)))
    plt.pie(df_images['Image_Count'], labels=df_images['Disease'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Proportion of Images by Disease Category', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.savefig('image_disease_pie.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 4: Image Count Distribution (Box Plot)
    plt.figure(figsize=(10, 6))
    plt.boxplot(df_images['Image_Count'], patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    plt.title('Distribution of Image Counts per Disease Category', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('image_count_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 5: Dataset Balance Analysis (Horizontal Bar Chart)
    plt.figure(figsize=(12, 10))
    mean_count = df_images['Image_Count'].mean()
    colors = ['red' if count < mean_count * 0.5 else 
              'orange' if count < mean_count else 
              'green' for count in df_images['Image_Count']]
    
    bars = plt.barh(range(len(df_images)), df_images['Image_Count'], 
                    color=colors, alpha=0.7, edgecolor='black')
    
    plt.title('Dataset Balance Analysis by Disease Category', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Images', fontsize=12)
    plt.ylabel('Disease Categories', fontsize=12)
    plt.yticks(range(len(df_images)), df_images['Disease'])
    plt.grid(axis='x', alpha=0.3)
    plt.axvline(mean_count, color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_count:.0f} images')
    plt.legend()
    plt.tight_layout()
    plt.savefig('image_balance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualization 6: Statistical Summary for Image Dataset (Bar Chart)
    plt.figure(figsize=(14, 8))
    
    # Calculate all statistics
    mean_val = df_images['Image_Count'].mean()
    median_val = df_images['Image_Count'].median()
    std_val = df_images['Image_Count'].std()
    mode_val = df_images['Image_Count'].mode().iloc[0] if len(df_images['Image_Count'].mode()) > 0 else mean_val
    
    # Create bar chart
    stats_names = ['Mean', 'Median', 'Mode', 'Std Dev', 'Mean + Std', 'Mean - Std']
    stats_values = [mean_val, median_val, mode_val, std_val, mean_val + std_val, mean_val - std_val]
    colors = ['red', 'orange', 'purple', 'green', 'blue', 'blue']
    
    bars = plt.bar(stats_names, stats_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, stats_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Statistical Summary of Image Count Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Statistical Measures', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('image_statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_images, total_images

def print_summary(df_symptoms, disease_counts, df_images, total_images):
    """Print summary statistics"""
    print("\n=== SUMMARY STATISTICS ===")
    print("=" * 30)
    
    print("\nSYMPTOM2DISEASE DATASET:")
    print(f"- Total samples: {len(df_symptoms)}")
    print(f"- Number of diseases: {len(df_symptoms['label'].unique())}")
    print(f"- Average text length: {df_symptoms['text_length'].mean():.0f} characters")
    print(f"- Average word count: {df_symptoms['word_count'].mean():.0f} words")
    print(f"- Most common disease: {disease_counts.index[0]} ({disease_counts.iloc[0]} samples)")
    print(f"- Least common disease: {disease_counts.index[-1]} ({disease_counts.iloc[-1]} samples)")
    
    print("\nSKIN DISEASE IMAGE DATASET:")
    print(f"- Total images: {total_images}")
    print(f"- Number of disease categories: {len(df_images)}")
    print(f"- Average images per category: {df_images['Image_Count'].mean():.0f}")
    print(f"- Most images: {df_images.iloc[0]['Disease']} ({df_images.iloc[0]['Image_Count']} images)")
    print(f"- Least images: {df_images.iloc[-1]['Disease']} ({df_images.iloc[-1]['Image_Count']} images)")
    print(f"- Dataset balance ratio: {df_images.iloc[0]['Image_Count'] / df_images.iloc[-1]['Image_Count']:.2f}")
    
    print("\nKEY INSIGHTS:")
    print("1. Text dataset shows good diversity across disease types")
    print("2. Image dataset has significant class imbalance")
    print("3. Both datasets provide complementary information for disease prediction")
    print("4. Text descriptions vary significantly in length and detail")
    print("5. Image categories range from well-represented to under-represented")
    
    # Additional Visualization: Comparative Statistical Analysis
    print("\n=== GENERATING COMPARATIVE STATISTICAL ANALYSIS ===")
    
    # Calculate statistics for both datasets
    symptom_mean = df_symptoms['text_length'].mean()
    symptom_median = df_symptoms['text_length'].median()
    symptom_std = df_symptoms['text_length'].std()
    symptom_mode = df_symptoms['text_length'].mode().iloc[0] if len(df_symptoms['text_length'].mode()) > 0 else symptom_mean
    
    image_mean = df_images['Image_Count'].mean()
    image_median = df_images['Image_Count'].median()
    image_std = df_images['Image_Count'].std()
    image_mode = df_images['Image_Count'].mode().iloc[0] if len(df_images['Image_Count'].mode()) > 0 else image_mean
    
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Symptom dataset statistics
    stats_names = ['Mean', 'Median', 'Mode', 'Std Dev']
    symptom_stats = [symptom_mean, symptom_median, symptom_mode, symptom_std]
    colors = ['red', 'orange', 'purple', 'green']
    
    bars1 = ax1.bar(stats_names, symptom_stats, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Symptom Dataset Statistics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Text Length (characters)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, symptom_stats):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Image dataset statistics
    image_stats = [image_mean, image_median, image_mode, image_std]
    
    bars2 = ax2.bar(stats_names, image_stats, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Image Dataset Statistics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, image_stats):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Comparative Statistical Analysis of Both Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparative_statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run analysis for both datasets
    df_symptoms, disease_counts = analyze_symptom_dataset()
    df_images, total_images = analyze_image_dataset()
    print_summary(df_symptoms, disease_counts, df_images, total_images)
    
    print("\nAnalysis complete! All visualizations have been saved as PNG files.")
