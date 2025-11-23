import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison():
    # Model names
    models = ['ResNet18', 'VGG16', 'DenseNet121', 'GoogLeNet']
    
    # Evaluation metrics data
    accuracy = [95.17, 93.33, 95.33, 95.28]
    precision = [95.17, 93.34, 95.33, 95.28]
    recall = [95.17, 93.33, 95.33, 95.28]
    f1_score = [95.16, 93.33, 95.33, 95.28]
    
    # Set the width of each bar and positions
    bar_width = 0.2
    x = np.arange(len(models))
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('#F8F9FA')
    
    # Create bars for each metric
    bars1 = ax.bar(x - 1.5*bar_width, accuracy, bar_width, 
                   label='Accuracy', color='#3498DB', alpha=0.9)
    bars2 = ax.bar(x - 0.5*bar_width, precision, bar_width, 
                   label='Precision', color='#F39C12', alpha=0.9)
    bars3 = ax.bar(x + 0.5*bar_width, recall, bar_width, 
                   label='Recall', color='#9B59B6', alpha=0.9)
    bars4 = ax.bar(x + 1.5*bar_width, f1_score, bar_width, 
                   label='F1 Score', color='#27AE60', alpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('Convolution Neural Networks', fontsize=13, fontweight='normal', color='#2C3E50')
    ax.set_ylabel('Performance Score (%)', fontsize=13, fontweight='normal', color='#2C3E50')
    ax.set_title('CIFAR-10 Classification Performance', 
                 fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, color='#34495E')
    ax.set_ylim([92, 96])
    
    # Customize legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
             edgecolor='#BDC3C7', fancybox=True, shadow=True)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#95A5A6')
    ax.set_axisbelow(True)
    
    # Add value labels on top of each bar
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color='#2C3E50')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('./inforgraphics/model_comparison_styled.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Display the plot
    plt.show()

def plot_adversarial_accuracy():
    # Model names
    models = ['ResNet18', 'VGG16', 'DenseNet121', 'GoogLeNet']
    
    # Adversarial accuracy data (converted to number of images out of 1000)
    total_samples = 1000
    fgsm_acc = [0.429, 0.559, 0.524, 0.339]  # Original accuracy values
    pgd_acc = [0.005, 0.087, 0.111, 0.0]
    
    # Convert to number of correctly classified images
    fgsm_images = [acc * total_samples for acc in fgsm_acc]
    pgd_images = [acc * total_samples for acc in pgd_acc]
    
    # Set the width of each bar and positions
    bar_width = 0.35
    x = np.arange(len(models))
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('#F8F9FA')
    
    # Create bars for each attack type
    bars1 = ax.bar(x - bar_width/2, fgsm_images, bar_width, 
                   label='FGSM Attack', color='#3498DB', alpha=0.9)
    bars2 = ax.bar(x + bar_width/2, pgd_images, bar_width, 
                   label='PGD Attack', color='#C0392B', alpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('Convolution Neural Networks', fontsize=13, fontweight='normal', color='#2C3E50')
    ax.set_ylabel('Number of Correctly Classified Images (out of 1000)', fontsize=13, fontweight='normal', color='#2C3E50')
    ax.set_title('CIFAR-10 Adversarial Robustness Comparison (ε = 0.031)', 
                 fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, color='#34495E')
    ax.set_ylim([0, 600])
    
    # Customize legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
             edgecolor='#BDC3C7', fancybox=True, shadow=True)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#95A5A6')
    ax.set_axisbelow(True)
    
    # Add value labels on top of each bar
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if value is greater than 0
                ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, 
                       fontweight='bold', color='#2C3E50')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Add reference line for total samples
    ax.axhline(y=1000, color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.7, label='Total Samples')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('./inforgraphics/adversarial_accuracy_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Display the plot
    plt.show()



if __name__ == "__main__":
    plot_model_comparison()
    print("Standard performance visualization generated successfully!")
    
    plot_adversarial_accuracy()
    print("Adversarial robustness visualization generated successfully!")