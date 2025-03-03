import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from torchvision.io import read_video
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import torch.cuda as cuda
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import signal

# Configuration
CONFIG = {
    "kinetics_dir": "/ssd_4TB/divake/vivit_kinetics400/k400",  # Base path to Kinetics dataset
    "val_csv": "/ssd_4TB/divake/vivit_kinetics400/k400/annotations/val.csv",  # Path to validation CSV
    "num_samples": None,                            # Set to a smaller number for testing
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use all available GPUs
    "batch_size": 16,                               # Increased batch size for multi-GPU
    "seed": 42,
    "model_id": "MCG-NJU/videomae-base-finetuned-kinetics",  # A verified model for Kinetics
    "num_workers": 16,                              # Increased worker count
    "model_save_path": "videomae_model",            # Path to save the downloaded model
    "use_multi_gpu": True,                          # Whether to use multiple GPUs
    "plots_dir": "/ssd_4TB/divake/vivit_kinetics400/plots",  # Directory to save plots
    "metrics_file": "VMAE_performance.txt",          # File to save performance metrics
    "timeout_seconds": 1800                         # 30-minute timeout for checkpoint creation
}

# Set random seed for reproducibility
random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

def get_video_files_with_labels_from_csv(csv_path, video_dir, num_samples=None):
    """Get video files and labels from CSV annotation file"""
    print(f"Reading annotations from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file {csv_path} does not exist!")
        return [], {}
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Display first few rows to understand structure
    print("First few rows of CSV:")
    print(df.head())
    
    video_files = []
    
    # Get unique class labels
    unique_labels = sorted(df['label'].unique())
    class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"Found {len(unique_labels)} unique classes in CSV")
    
    # List all video files in the directory for faster lookup
    print(f"Scanning video directory: {video_dir}")
    all_video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    video_filenames = {os.path.basename(f): f for f in all_video_files}
    print(f"Found {len(video_filenames)} video files in directory")
    
    # Sample a few video filenames to understand the format
    if video_filenames:
        print("Sample video filenames:")
        for filename in list(video_filenames.keys())[:5]:
            print(f"  - {filename}")
    
    # Collect video files
    for _, row in df.iterrows():
        # Try different filename formats based on the CSV structure
        possible_filenames = []
        
        # Format 1: Standard Kinetics format
        if 'youtube_id' in df.columns and 'time_start' in df.columns and 'time_end' in df.columns:
            filename = f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}.mp4"
            possible_filenames.append(filename)
        
        # Format 2: YouTube ID with timestamps (your format)
        if 'youtube_id' in df.columns:
            filename = f"{row['youtube_id']}_000010_000020.mp4"  # Adjust timestamps as needed
            possible_filenames.append(filename)
        
        # Format 3: First column as ID
        filename = f"{row.iloc[0]}.mp4"
        possible_filenames.append(filename)
        
        # Try each possible filename
        video_path = None
        for filename in possible_filenames:
            if filename in video_filenames:
                video_path = video_filenames[filename]
                break
        
        # If not found, try direct path
        if video_path is None:
            direct_path = os.path.join(video_dir, f"{row['youtube_id']}_000010_000020.mp4")
            if os.path.exists(direct_path):
                video_path = direct_path
        
        # If found, add to our list
        if video_path and os.path.exists(video_path):
            video_files.append({
                "path": video_path,
                "label": class_to_idx[row['label']],
                "class_name": row['label']
            })
    
    print(f"Found {len(video_files)} videos that match CSV entries")
    
    # Sample a subset if requested
    if num_samples is not None and num_samples < len(video_files) and len(video_files) > 0:
        video_files = random.sample(video_files, num_samples)
        print(f"Sampled {len(video_files)} videos for evaluation")
    
    return video_files, class_to_idx

# Handle timeout to prevent script from getting killed
class TimeoutHandler:
    def __init__(self, seconds=1800):
        self.seconds = seconds
        self.triggered = False
        signal.signal(signal.SIGALRM, self._handle_timeout)
    
    def _handle_timeout(self, signum, frame):
        self.triggered = True
        print(f"\nWARNING: Process running for {self.seconds} seconds, triggering checkpoint save")
    
    def start(self):
        signal.alarm(self.seconds)
    
    def reset(self):
        self.triggered = False
        signal.alarm(self.seconds)
    
    def cancel(self):
        signal.alarm(0)

def process_video(args):
    """Process a video file for the model"""
    video_path, processor = args
    try:
        # Read the video file
        video, _, _ = read_video(video_path, pts_unit="sec")
        
        # Check if video is empty or invalid
        if video.numel() == 0 or video.shape[0] == 0:
            print(f"Warning: Empty video file {video_path}, skipping")
            return None
        
        # MVITv2 typically uses 16 frames with a stride of 4
        num_frames = video.shape[0]
        if num_frames < 16:
            # Duplicate frames if video is too short
            # Fix division by zero by always ensuring we have at least 1 frame
            repeat_factor = max(16 // max(num_frames, 1) + 1, 1)
            video = video.repeat(repeat_factor, 1, 1, 1)[:16]
        else:
            # Sample 16 frames uniformly
            indices = torch.linspace(0, num_frames - 1, 16).long()
            video = video[indices]
        
        # Move channels to correct dimension: [num_frames, channels, height, width]
        video = video.permute(0, 3, 1, 2)
        
        # Apply image processor (handles normalization and resize)
        inputs = processor(list(video), return_tensors="pt")
        
        return inputs.pixel_values
    
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def download_and_save_model(model_id, save_path):
    """Download model and save it locally"""
    if os.path.exists(save_path) and os.path.isdir(save_path) and len(os.listdir(save_path)) > 0:
        print(f"Model already downloaded at {save_path}")
        return save_path
    
    print(f"Downloading model {model_id}...")
    model = AutoModelForVideoClassification.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model and processor
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    return save_path

def monitor_gpu_usage():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

def generate_performance_plots(all_labels, all_preds, all_logits, all_top5_preds, class_accuracy, idx_to_class, 
                              videos_per_second_history, elapsed_time_per_batch, conf_mat, plots_dir):
    """Generate and save various performance plots"""
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Confusion Matrix (for top-N classes)
    plt.figure(figsize=(12, 10))
    top_classes = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)[:20]
    top_class_indices = [idx for idx, _ in top_classes]
    
    # Extract the submatrix for top classes
    class_to_matrix_idx = {class_idx: i for i, class_idx in enumerate(sorted(set(all_labels + all_preds)))}
    matrix_indices = [class_to_matrix_idx[idx] for idx in top_class_indices]
    
    # If there are too many classes, just show the top 20
    if len(matrix_indices) > 20:
        matrix_indices = matrix_indices[:20]
        top_class_indices = top_class_indices[:20]
    
    if matrix_indices:  # Only proceed if we have data
        sub_conf_mat = conf_mat[np.ix_(matrix_indices, matrix_indices)]
        
        # Plot the confusion matrix
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            sub_conf_mat / np.sum(sub_conf_mat, axis=1)[:, np.newaxis],
            annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[idx_to_class[idx] for idx in top_class_indices],
            yticklabels=[idx_to_class[idx] for idx in top_class_indices]
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Top 20 Classes)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix_top20.png'), dpi=300)
        plt.close()
    
    # Plot 2: Top Classes by Accuracy
    plt.figure(figsize=(14, 10))
    top_n = min(20, len(class_accuracy))
    top_classes = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    classes = [idx_to_class[idx][:20] + '...' if len(idx_to_class[idx]) > 20 else idx_to_class[idx] for idx, _ in top_classes]
    accuracies = [acc * 100 for _, acc in top_classes]
    
    plt.barh(range(len(classes)), accuracies, color='green')
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Accuracy (%)')
    plt.title(f'Top {top_n} Classes by Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_classes_accuracy.png'), dpi=300)
    plt.close()
    
    # Plot 3: Bottom Classes by Accuracy
    plt.figure(figsize=(14, 10))
    bottom_n = min(20, len(class_accuracy))
    bottom_classes = sorted(class_accuracy.items(), key=lambda x: x[1])[:bottom_n]
    
    classes = [idx_to_class[idx][:20] + '...' if len(idx_to_class[idx]) > 20 else idx_to_class[idx] for idx, _ in bottom_classes]
    accuracies = [acc * 100 for _, acc in bottom_classes]
    
    plt.barh(range(len(classes)), accuracies, color='red')
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Accuracy (%)')
    plt.title(f'Bottom {bottom_n} Classes by Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bottom_classes_accuracy.png'), dpi=300)
    plt.close()
    
    # Plot 4: Class Distribution in Validation Set
    plt.figure(figsize=(14, 10))
    label_counts = {}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Sort by count
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    classes = [idx_to_class[idx][:20] + '...' if len(idx_to_class[idx]) > 20 else idx_to_class[idx] for idx, _ in sorted_counts]
    counts = [count for _, count in sorted_counts]
    
    plt.barh(range(len(classes)), counts, color='blue')
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Number of Samples')
    plt.title('Class Distribution in Validation Set (Top 20)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 5: Processing Speed Over Time
    if videos_per_second_history:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(videos_per_second_history)), videos_per_second_history, marker='o')
        plt.xlabel('Batch')
        plt.ylabel('Videos per Second')
        plt.title('Processing Speed Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'processing_speed.png'), dpi=300)
        plt.close()
    
    # Plot 6: Cumulative Elapsed Time
    if elapsed_time_per_batch:
        plt.figure(figsize=(12, 6))
        cum_time = np.cumsum(elapsed_time_per_batch)
        plt.plot(range(len(cum_time)), cum_time / 60, marker='o')  # Convert to minutes
        plt.xlabel('Batch')
        plt.ylabel('Cumulative Time (minutes)')
        plt.title('Cumulative Processing Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cumulative_time.png'), dpi=300)
        plt.close()
    
    # Plot 7: Model Confidence Distribution (using logits)
    if all_logits:
        plt.figure(figsize=(10, 6))
        # Get the confidence scores (softmax of logits)
        confidences = [np.max(np.exp(logits) / np.sum(np.exp(logits))) for logits in all_logits]
        plt.hist(confidences, bins=50, alpha=0.75)
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Model Confidence Distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confidence_distribution.png'), dpi=300)
        plt.close()
        
        # Plot 8: Confidence vs Correctness
        plt.figure(figsize=(10, 6))
        correct_confidences = [confidences[i] for i in range(len(confidences)) if all_preds[i] == all_labels[i]]
        wrong_confidences = [confidences[i] for i in range(len(confidences)) if all_preds[i] != all_labels[i]]
        
        plt.hist(correct_confidences, bins=50, alpha=0.5, label='Correct Predictions', color='green')
        plt.hist(wrong_confidences, bins=50, alpha=0.5, label='Wrong Predictions', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence vs. Correctness')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confidence_vs_correctness.png'), dpi=300)
        plt.close()

def save_performance_metrics(metrics_file, top1_accuracy, top5_accuracy, class_accuracy, idx_to_class, 
                           all_labels, all_preds, all_top5_preds, total_execution_time, videos_per_second_avg):
    """Save detailed performance metrics to a text file"""
    with open(metrics_file, 'w') as f:
        # Overall metrics
        f.write("===== VMAE PERFORMANCE METRICS =====\n\n")
        f.write(f"Dataset: Kinetics-400 Validation Set\n")
        f.write(f"Model: {CONFIG['model_id']}\n")
        f.write(f"Number of samples evaluated: {len(all_labels)}\n\n")
        
        # Accuracy metrics
        f.write("===== ACCURACY METRICS =====\n")
        f.write(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%\n\n")
        
        # Execution metrics
        hours, remainder = divmod(total_execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write("===== EXECUTION METRICS =====\n")
        f.write(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        f.write(f"Average processing speed: {videos_per_second_avg:.2f} videos/sec\n\n")
        
        # Classification report
        f.write("===== CLASSIFICATION REPORT =====\n")
        # Create a classification report with proper class names
        class_names = [idx_to_class[idx] for idx in sorted(set(all_labels + all_preds))]
        target_names = [name[:30] + "..." if len(name) > 30 else name for name in class_names]
        
        # Generate detailed classification metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, 
            labels=sorted(set(all_labels + all_preds)),
            zero_division=0  # Set to 0 instead of raising warning
        )
        
        # Write precision, recall, f1 for each class
        f.write(f"{'Class':<35} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, class_idx in enumerate(sorted(set(all_labels + all_preds))):
            class_name = idx_to_class[class_idx]
            if len(class_name) > 30:
                class_name = class_name[:27] + "..."
            
            f.write(f"{class_name:<35} {precision[i]:.4f}      {recall[i]:.4f}      {f1[i]:.4f}        {support[i]}\n")
        
        f.write("\n")
        
        # Top classes by accuracy
        f.write("===== TOP 10 CLASSES BY ACCURACY =====\n")
        top_classes = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (class_idx, acc) in enumerate(top_classes):
            f.write(f"{i+1}. {idx_to_class[class_idx]}: {acc * 100:.2f}%\n")
        f.write("\n")
        
        # Bottom classes by accuracy
        f.write("===== BOTTOM 10 CLASSES BY ACCURACY =====\n")
        bottom_classes = sorted(class_accuracy.items(), key=lambda x: x[1])[:10]
        for i, (class_idx, acc) in enumerate(bottom_classes):
            f.write(f"{i+1}. {idx_to_class[class_idx]}: {acc * 100:.2f}%\n")
        f.write("\n")
        
        # Top-5 accuracy contribution
        f.write("===== TOP-5 ACCURACY ANALYSIS =====\n")
        top1_correct = sum(1 for i in range(len(all_labels)) if all_preds[i] == all_labels[i])
        top5_only_correct = sum(1 for i in range(len(all_labels)) 
                              if all_preds[i] != all_labels[i] and all_labels[i] in all_top5_preds[i])
        
        f.write(f"Correctly classified in Top-1: {top1_correct} ({top1_correct/len(all_labels)*100:.2f}%)\n")
        f.write(f"Correctly classified in Top-5 (but not Top-1): {top5_only_correct} ({top5_only_correct/len(all_labels)*100:.2f}%)\n")
        f.write(f"Total correctly classified in Top-5: {top1_correct + top5_only_correct} ({(top1_correct + top5_only_correct)/len(all_labels)*100:.2f}%)\n")

def save_checkpoint(checkpoint_file, all_preds, all_labels, all_top5_preds, all_logits, batch_idx):
    """Save checkpoint to a file"""
    try:
        print(f"Saving checkpoint at batch {batch_idx}...")
        np.savez(
            checkpoint_file,
            preds=np.array(all_preds),
            labels=np.array(all_labels),
            top5_preds=np.array(all_top5_preds),
            logits=np.array(all_logits),
            batch_idx=batch_idx
        )
        print("Checkpoint saved successfully")
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False

def main():
    # Start timing the total execution
    total_start_time = time.time()
    
    # Check CUDA availability and print GPU info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
        monitor_gpu_usage()
    
    print(f"Using device: {CONFIG['device']}")
    
    # Create plots directory if it doesn't exist
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    
    # Download and save model locally
    model_path = download_and_save_model(CONFIG['model_id'], CONFIG['model_save_path'])
    
    # Load model and processor from local path
    print(f"Loading model from: {model_path}")
    model = AutoModelForVideoClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # Move model to device and use DataParallel if multiple GPUs are available
    if CONFIG['use_multi_gpu'] and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(CONFIG['device'])
    model.eval()
    print("Model loaded successfully")
    monitor_gpu_usage()
    
    # Get video files with labels from CSV
    videos, class_to_idx = get_video_files_with_labels_from_csv(
        CONFIG['val_csv'],
        os.path.join(CONFIG['kinetics_dir'], 'val'),
        num_samples=CONFIG['num_samples']
    )
    
    if not videos:
        print("No videos found. Please check the CSV file and video paths.")
        return
    
    # Create index to class mapping for later reference
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    
    # Prepare for evaluation
    all_preds = []
    all_labels = []
    all_top5_preds = []  # Store top-5 predictions for each sample
    all_logits = []      # Store raw logits for additional analysis
    
    # For tracking processing speed
    videos_per_second_history = []
    elapsed_time_per_batch = []
    
    # Setup timeout handler
    timeout_handler = TimeoutHandler(seconds=CONFIG['timeout_seconds'])
    
    # Save checkpoint information
    checkpoint_file = "evaluation_checkpoint.npz"
    checkpoint_interval = 500  # Save checkpoint every 500 batches
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        try:
            print(f"Found checkpoint file {checkpoint_file}, loading...")
            checkpoint = np.load(checkpoint_file, allow_pickle=True)
            all_preds = checkpoint['preds'].tolist()
            all_labels = checkpoint['labels'].tolist()
            all_top5_preds = checkpoint['top5_preds'].tolist()
            all_logits = checkpoint['logits'].tolist()
            # Get the batch index if it exists in the checkpoint
            start_batch = checkpoint['batch_idx'].item() if 'batch_idx' in checkpoint else len(all_preds) // CONFIG['batch_size']
            print(f"Resuming from batch {start_batch}, {len(all_preds)} videos already processed")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            start_batch = 0
    else:
        start_batch = 0
    
    # Process videos in batches
    start_time = time.time()
    try:
        # Start the timeout timer
        timeout_handler.start()
        
        for i in tqdm(range(start_batch, len(videos), CONFIG['batch_size']), desc="Evaluating"):
            batch_start_time = time.time()
            
            # Check if timeout triggered
            if timeout_handler.triggered:
                save_checkpoint(checkpoint_file, all_preds, all_labels, all_top5_preds, all_logits, i)
                timeout_handler.reset()
            
            # Determine effective batch size
            effective_batch_end = min(i + CONFIG['batch_size'], len(videos))
            batch_videos = videos[i:effective_batch_end]
            
            # Explicit garbage collection to prevent memory leaks
            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Dynamic batch size adjustment if we're near the end or having memory issues
            if i > len(videos) * 0.9 and CONFIG['batch_size'] > 16:
                # Reduce batch size for the last 10% to avoid memory issues
                old_batch_size = CONFIG['batch_size']
                CONFIG['batch_size'] = 16
                print(f"Reducing batch size from {old_batch_size} to {CONFIG['batch_size']} for final batches")
            
            # Process videos in parallel with error handling for the entire batch
            try:
                with ThreadPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
                    batch_results = list(executor.map(
                        process_video, 
                        [(video_info['path'], processor) for video_info in batch_videos]
                    ))
                
                # Filter out None values and get corresponding labels
                batch_processed = []
                batch_labels = []
                for j, result in enumerate(batch_results):
                    if result is not None:
                        batch_processed.append(result)
                        batch_labels.append(batch_videos[j]['label'])
                
                # Skip batch if empty
                if not batch_processed:
                    print(f"Warning: Batch {i} is empty after filtering, skipping")
                    continue
                
                # Stack and evaluate with robust error handling
                try:
                    # Stack processed videos
                    stacked_videos = torch.cat(batch_processed, dim=0)
                    
                    # Move to GPU and run inference
                    stacked_videos = stacked_videos.to(CONFIG['device'])
                    
                    # Print GPU usage after moving data
                    if i % 10 == 0:  # Only print every 10 batches to reduce output
                        print(f"\nBatch {i}, Videos shape: {stacked_videos.shape}")
                        monitor_gpu_usage()
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = model(pixel_values=stacked_videos)
                        # Store raw logits for analysis
                        logits = outputs.logits.cpu().numpy()
                        all_logits.extend(logits)
                        
                        # Top-1 predictions
                        predictions = outputs.logits.argmax(-1).cpu().numpy()
                        
                        # Top-5 predictions
                        topk_values, topk_indices = torch.topk(outputs.logits, k=5, dim=-1)
                        top5_predictions = topk_indices.cpu().numpy()
                    
                    # Free GPU memory immediately
                    del stacked_videos
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Save results
                    all_preds.extend(predictions.tolist())
                    all_labels.extend(batch_labels)
                    all_top5_preds.extend(top5_predictions.tolist())
                    
                    # Track batch processing time
                    batch_elapsed = time.time() - batch_start_time
                    elapsed_time_per_batch.append(batch_elapsed)
                    
                    # Print progress
                    if i % 10 == 0:
                        elapsed = time.time() - start_time
                        videos_processed = min((i + 1) * CONFIG['batch_size'], len(videos))
                        videos_per_second = videos_processed / elapsed
                        videos_per_second_history.append(videos_per_second)
                        print(f"Processed {videos_processed}/{len(videos)} videos "
                              f"({videos_per_second:.2f} videos/sec)")
                    
                    # Save checkpoint periodically
                    if i % checkpoint_interval == 0 and i > 0:
                        save_checkpoint(checkpoint_file, all_preds, all_labels, all_top5_preds, all_logits, i)
                
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA out of memory in batch {i}, reducing batch size and retrying")
                        if CONFIG['batch_size'] > 4:
                            CONFIG['batch_size'] = max(CONFIG['batch_size'] // 2, 4)
                            print(f"Reduced batch size to {CONFIG['batch_size']}")
                        
                        # Free memory and continue
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Save checkpoint before retrying
                        save_checkpoint(checkpoint_file, all_preds, all_labels, all_top5_preds, all_logits, i)
                        
                        # Don't process this batch now, it will be retried with smaller batch size
                        continue
                    else:
                        print(f"Runtime error in batch {i}: {e}")
                
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    # Continue with next batch - we don't want to lose progress
            
            except Exception as e:
                print(f"Error in ThreadPoolExecutor for batch {i}: {e}")
                # Continue with next batch
        
        # Cancel the timeout alarm at the end
        timeout_handler.cancel()
        
    except KeyboardInterrupt:
        print("Evaluation interrupted. Saving current results...")
        save_checkpoint(checkpoint_file, all_preds, all_labels, all_top5_preds, all_logits, i)
    
    finally:
        # Always cancel the timeout alarm in case of any exit
        try:
            timeout_handler.cancel()
        except:
            pass
        
        # Only delete checkpoint if we're at the end
        if all_preds and start_batch >= (len(videos) // CONFIG['batch_size'] - 1):
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                    print(f"Evaluation completed, removed checkpoint file {checkpoint_file}")
                except:
                    print(f"Note: Could not remove checkpoint file {checkpoint_file}")
    
    # Calculate metrics
    if all_preds:
        # Top-1 accuracy
        top1_accuracy = accuracy_score(all_labels, all_preds)
        
        # Top-5 accuracy
        top5_correct = 0
        for i, label in enumerate(all_labels):
            if label in all_top5_preds[i]:
                top5_correct += 1
        top5_accuracy = top5_correct / len(all_labels)
        
        # Get the unique classes that actually appear in the predictions and labels
        unique_classes = sorted(set(all_labels + all_preds))
        
        # Create a mapping from class index to position in confusion matrix
        class_to_matrix_idx = {class_idx: i for i, class_idx in enumerate(unique_classes)}
        
        # Generate confusion matrix with the correct size
        conf_mat = confusion_matrix(all_labels, all_preds, labels=unique_classes)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Total samples evaluated: {len(all_labels)}")
        print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
        
        # Save results
        np.save("confusion_matrix.npy", conf_mat)
        print("Confusion matrix saved to confusion_matrix.npy")
        
        # Calculate per-class accuracy only for classes that appear in the evaluation
        class_accuracy = {}
        for class_idx in unique_classes:
            matrix_idx = class_to_matrix_idx[class_idx]
            if conf_mat.sum(axis=1)[matrix_idx] > 0:
                class_accuracy[class_idx] = conf_mat[matrix_idx, matrix_idx] / conf_mat.sum(axis=1)[matrix_idx]
        
        # Sort classes by accuracy
        sorted_classes = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)
        
        # Print top classes
        print("\nTop 5 classes by accuracy:")
        for class_idx, acc in sorted_classes[:5]:
            print(f"  {idx_to_class[class_idx]}: {acc * 100:.2f}%")
        
        # Print bottom classes
        print("\nBottom 5 classes by accuracy:")
        for class_idx, acc in sorted_classes[-5:]:
            print(f"  {idx_to_class[class_idx]}: {acc * 100:.2f}%")
            
        # Calculate videos per second (average)
        total_processing_time = time.time() - start_time
        videos_per_second_avg = len(all_labels) / total_processing_time
        
        # Calculate total execution time
        total_execution_time = time.time() - total_start_time
        hours, remainder = divmod(total_execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        try:
            # Save detailed performance metrics to file
            save_performance_metrics(
                CONFIG['metrics_file'], 
                top1_accuracy, 
                top5_accuracy, 
                class_accuracy, 
                idx_to_class, 
                all_labels, 
                all_preds, 
                all_top5_preds, 
                total_execution_time, 
                videos_per_second_avg
            )
            print(f"Detailed performance metrics saved to {CONFIG['metrics_file']}")
            
            # Generate and save plots
            generate_performance_plots(
                all_labels, 
                all_preds, 
                all_logits, 
                all_top5_preds, 
                class_accuracy, 
                idx_to_class, 
                videos_per_second_history, 
                elapsed_time_per_batch, 
                conf_mat, 
                CONFIG['plots_dir']
            )
            print(f"Performance plots saved to {CONFIG['plots_dir']}")
        except Exception as e:
            print(f"Error during final result generation: {e}")
            # Still save the raw results
            np.savez(
                "evaluation_results.npz",
                preds=np.array(all_preds),
                labels=np.array(all_labels),
                top5_preds=np.array(all_top5_preds),
                logits=np.array(all_logits),
                accuracy_top1=top1_accuracy,
                accuracy_top5=top5_accuracy
            )
            print("Raw results saved to evaluation_results.npz")
    else:
        print("No predictions were made. Please check if PyAV is installed correctly.")
        
        # Calculate total execution time
        total_execution_time = time.time() - total_start_time
        hours, remainder = divmod(total_execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main() 