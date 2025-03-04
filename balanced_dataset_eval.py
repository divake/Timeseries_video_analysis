#!/usr/bin/env python3
"""
Enhanced VideoMAE evaluation script with balanced dataset sampling.
This script extends the original evaluation script with the ability to use
balanced subsets of the Kinetics-400 dataset.
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VideoMAE evaluation on balanced Kinetics-400 dataset")
    
    # Dataset configuration
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--samples-per-class", type=int, default=25,
                        help="Number of samples per class (for balanced evaluation)")
    parser.add_argument("--base-path", type=str, default="/ssd_4TB/divake/vivit_kinetics400/k400",
                        help="Base path to the Kinetics-400 dataset")
    
    # Model configuration
    parser.add_argument("--model-id", type=str, 
                        default="MCG-NJU/videomae-base-finetuned-kinetics",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (defaults to timestamp-based directory)")
    
    # Execution configuration
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of worker threads for data loading")
    parser.add_argument("--no-multi-gpu", action="store_true",
                        help="Disable multi-GPU processing even if available")
    
    return parser.parse_args()

def get_balanced_video_files(base_path: str, split: str, samples_per_class: int, seed: int = 42) -> Tuple[List[Dict], Dict, Dict]:
    """
    Get a balanced subset of video files with labels from the specified split.
    
    Args:
        base_path (str): Base path to the Kinetics-400 dataset
        split (str): Dataset split ('train', 'val', or 'test')
        samples_per_class (int): Number of samples per class
        seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[List[Dict], Dict, Dict]: Video files with labels, class to index mapping, and video counts per class
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Paths for annotations and videos
    base_path = Path(base_path)
    annotation_path = base_path / 'annotations' / f'{split}.csv'
    video_path = base_path / split
    
    print(f"Loading annotations from {annotation_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    # Load annotations
    df = pd.read_csv(annotation_path)
    print(f"Loaded {len(df)} annotations for {split} split")
    
    # Group by class and sample equally
    subset_data = []
    valid_classes = 0
    skipped_classes = 0
    
    for class_name in tqdm(df['label'].unique(), desc=f"Creating balanced subset for {split}"):
        class_samples = df[df['label'] == class_name]
        
        # Skip if not enough samples
        if len(class_samples) < samples_per_class:
            if len(class_samples) < 1:  # Skip classes with no samples
                print(f"Skipping class {class_name} with no samples")
                skipped_classes += 1
                continue
            
            print(f"Warning: Class {class_name} has only {len(class_samples)} samples (< {samples_per_class})")
            selected = class_samples
        else:
            selected = class_samples.sample(n=samples_per_class, random_state=seed)
        
        subset_data.append(selected)
        valid_classes += 1
    
    if skipped_classes > 0:
        print(f"Skipped {skipped_classes} classes with insufficient samples")
    
    subset_df = pd.concat(subset_data, ignore_index=True)
    print(f"Created subset with {len(subset_df)} samples from {valid_classes} classes")
    
    # Create class to index mapping
    classes = sorted(subset_df['label'].unique())
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    
    # Create video info list
    video_files = []
    existing_videos = 0
    class_video_counts = {}  # Track videos found per class
    
    # Initialize class_video_counts for all classes
    for class_name in classes:
        class_video_counts[class_name] = 0
    
    # Find all video files in the split directory
    print(f"Scanning directory for video files: {video_path}")
    all_video_files = {}
    
    # Use glob to find all mp4 files (faster than scanning the entire directory)
    for video_file in video_path.glob("*.mp4"):
        video_id = video_file.stem.split('_')[0]  # Extract YouTube ID from filename
        if video_id not in all_video_files:
            all_video_files[video_id] = []
        all_video_files[video_id].append(video_file)
    
    print(f"Found {len(all_video_files)} unique video IDs in directory")
    
    # Process each annotation
    for _, row in tqdm(subset_df.iterrows(), desc="Preparing video files", total=len(subset_df)):
        youtube_id = row['youtube_id']
        time_start = row['time_start']
        time_end = row['time_end']
        class_name = row['label']
        
        video_found = False
        
        # First try: exact match with time_start and time_end from annotation
        expected_pattern = f"{youtube_id}_{time_start:06d}_{time_end:06d}.mp4"
        video_file = video_path / expected_pattern
        if video_file.exists():
            video_files.append({
                'path': str(video_file),
                'label': class_to_idx[class_name],
                'class_name': class_name
            })
            existing_videos += 1
            class_video_counts[class_name] += 1
            video_found = True
            continue
        
        # Second try: look for any file with matching YouTube ID
        if youtube_id in all_video_files:
            # Use the first file found for this YouTube ID
            video_file = all_video_files[youtube_id][0]
            video_files.append({
                'path': str(video_file),
                'label': class_to_idx[class_name],
                'class_name': class_name
            })
            existing_videos += 1
            class_video_counts[class_name] += 1
            video_found = True
            continue
        
        # Third try: search for files with similar patterns
        # This is a fallback for videos that might have different timestamp formats
        for file_pattern in [
            f"{youtube_id}_*.mp4",  # Any timestamp
            f"*{youtube_id}*.mp4",  # YouTube ID anywhere in filename
        ]:
            matching_files = list(video_path.glob(file_pattern))
            if matching_files:
                video_file = matching_files[0]  # Use the first match
                video_files.append({
                    'path': str(video_file),
                    'label': class_to_idx[class_name],
                    'class_name': class_name
                })
                existing_videos += 1
                class_video_counts[class_name] += 1
                video_found = True
                break
        
        if not video_found:
            print(f"Warning: Video not found for {youtube_id}")
    
    print(f"Found {existing_videos}/{len(subset_df)} videos ({existing_videos/len(subset_df)*100:.2f}%)")
    
    # Print class distribution
    print("\nClass distribution (videos found per class):")
    print("-" * 50)
    
    # Sort classes by number of videos found
    sorted_classes = sorted(class_video_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print all classes with their video counts
    print(f"{'Class Name':<30} {'Videos Found':<12} {'Requested':<10}")
    print("-" * 50)
    for class_name, count in sorted_classes:
        print(f"{class_name:<30} {count:<12} {samples_per_class:<10}")
    
    # Print summary statistics
    classes_with_videos = sum(1 for count in class_video_counts.values() if count > 0)
    classes_with_full_samples = sum(1 for count in class_video_counts.values() if count >= samples_per_class)
    
    print("\nSummary:")
    print(f"Classes with at least one video: {classes_with_videos}/{len(class_video_counts)} ({classes_with_videos/len(class_video_counts)*100:.2f}%)")
    print(f"Classes with full requested samples: {classes_with_full_samples}/{len(class_video_counts)} ({classes_with_full_samples/len(class_video_counts)*100:.2f}%)")
    
    return video_files, class_to_idx, class_video_counts

def main():
    """Main function to run the enhanced evaluation."""
    args = parse_args()
    
    print(f"Running evaluation on {args.split} split with {args.samples_per_class} samples per class")
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # Redirect stdout to a log file
        log_file = os.path.join(args.output_dir, f"eval_{args.split}_{args.samples_per_class}.log")
        sys.stdout = open(log_file, 'w')
    
    # Get balanced video files
    video_files, class_to_idx, class_video_counts = get_balanced_video_files(
        base_path=args.base_path,
        split=args.split,
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )
    
    # Import and run the original evaluation script with our modifications
    import sys
    import os
    
    # Add the directory containing the original script to the Python path
    original_script_dir = os.path.dirname("/ssd_4TB/divake/vivit_kinetics400/huggingface_video_mae_eval_main.py")
    sys.path.insert(0, original_script_dir)
    
    # Import the necessary functions from the original script
    from huggingface_video_mae_eval_main import process_video, monitor_gpu_usage, download_and_save_model
    
    # Now run the evaluation with our balanced dataset
    import torch
    import numpy as np
    from tqdm import tqdm
    import random
    from transformers import AutoImageProcessor, AutoModelForVideoClassification
    from sklearn.metrics import accuracy_score, confusion_matrix
    import torch.cuda as cuda
    from concurrent.futures import ThreadPoolExecutor
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Configuration based on command line arguments
    CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "model_id": args.model_id,
        "num_workers": args.num_workers,
        "model_save_path": "videomae_model",
        "use_multi_gpu": not args.no_multi_gpu,
        "plots_dir": os.path.join(args.output_dir, "plots") if args.output_dir else "plots",
        "metrics_file": os.path.join(args.output_dir, "VMAE_performance.txt") if args.output_dir else "VMAE_performance.txt",
    }
    
    # Set random seed for reproducibility
    random.seed(CONFIG["seed"])
    
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
    
    # Use our balanced video files
    videos = video_files
    
    if not videos:
        print("No videos found. Please check the dataset path and annotations.")
        return
    
    # Create index to class mapping for later reference
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    
    # Prepare for evaluation
    all_preds = []
    all_labels = []
    all_top5_preds = []  # Store top-5 predictions for each sample
    evaluated_videos_per_class = {}  # Track how many videos were actually evaluated per class
    
    # Initialize evaluated_videos_per_class
    for class_name in class_to_idx:
        evaluated_videos_per_class[class_name] = 0
    
    # Process videos in batches
    start_time = time.time()
    for i in tqdm(range(0, len(videos), CONFIG['batch_size']), desc="Evaluating"):
        batch_videos = videos[i:i + CONFIG['batch_size']]
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
            batch_results = list(executor.map(
                process_video, 
                [(video_info['path'], processor) for video_info in batch_videos]
            ))
        
        # Filter out None values and get corresponding labels
        batch_processed = []
        batch_labels = []
        batch_class_names = []  # Store class names for debugging
        for j, result in enumerate(batch_results):
            if result is not None:
                batch_processed.append(result)
                batch_labels.append(batch_videos[j]['label'])
                batch_class_names.append(batch_videos[j]['class_name'])
                # Increment the count of evaluated videos for this class
                evaluated_videos_per_class[batch_videos[j]['class_name']] += 1
        
        # Skip batch if empty
        if not batch_processed:
            continue
        
        # Stack and evaluate
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
                # Top-1 predictions
                predictions = outputs.logits.argmax(-1).cpu().numpy()
                
                # Top-5 predictions
                topk_values, topk_indices = torch.topk(outputs.logits, k=5, dim=-1)
                top5_predictions = topk_indices.cpu().numpy()
            
            # Save results
            all_preds.extend(predictions.tolist())
            all_labels.extend(batch_labels)
            all_top5_preds.extend(top5_predictions.tolist())
            
            # Print progress
            if i % 10 == 0:
                elapsed = time.time() - start_time
                videos_processed = min((i + 1) * CONFIG['batch_size'], len(videos))
                videos_per_second = videos_processed / elapsed
                print(f"Processed {videos_processed}/{len(videos)} videos "
                      f"({videos_per_second:.2f} videos/sec)")
        
        except Exception as e:
            print(f"Error in batch {i}: {e}")
    
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
        np.save(os.path.join(args.output_dir, "confusion_matrix.npy") if args.output_dir else "confusion_matrix.npy", conf_mat)
        print("Confusion matrix saved")
        
        # Calculate per-class accuracy and sample counts
        class_accuracy = {}
        
        # Calculate per-class accuracy
        for class_idx in unique_classes:
            matrix_idx = class_to_matrix_idx[class_idx]
            if conf_mat.sum(axis=1)[matrix_idx] > 0:
                class_accuracy[class_idx] = conf_mat[matrix_idx, matrix_idx] / conf_mat.sum(axis=1)[matrix_idx]
            else:
                class_accuracy[class_idx] = 0.0
        
        # Sort classes by accuracy
        sorted_classes = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out classes with no evaluated videos for console output
        valid_sorted_classes = [(class_idx, acc) for class_idx, acc in sorted_classes 
                               if evaluated_videos_per_class.get(idx_to_class[class_idx], 0) > 0]
        
        # Print top classes
        print("\nTop 5 classes by accuracy:")
        for class_idx, acc in valid_sorted_classes[:5]:
            class_name = idx_to_class[class_idx]
            eval_count = evaluated_videos_per_class.get(class_name, 0)
            found_count = class_video_counts.get(class_name, 0)
            print(f"  {class_name}: {acc * 100:.2f}% (evaluated: {eval_count}/{found_count})")
        
        # Print bottom classes
        print("\nBottom 5 classes by accuracy:")
        for class_idx, acc in valid_sorted_classes[-5:]:
            class_name = idx_to_class[class_idx]
            eval_count = evaluated_videos_per_class.get(class_name, 0)
            found_count = class_video_counts.get(class_name, 0)
            print(f"  {class_name}: {acc * 100:.2f}% (evaluated: {eval_count}/{found_count})")
        
        # Print classes with no samples
        classes_with_no_samples = [class_name for class_name in class_video_counts 
                                 if class_video_counts[class_name] == 0]
        if classes_with_no_samples:
            print(f"\nClasses with no videos found ({len(classes_with_no_samples)}):")
            for i, class_name in enumerate(sorted(classes_with_no_samples)[:10]):  # Show first 10
                print(f"  {class_name}")
            if len(classes_with_no_samples) > 10:
                print(f"  ... and {len(classes_with_no_samples) - 10} more")
    else:
        print("No predictions were made. Please check if PyAV is installed correctly.")
    
    # Calculate total execution time
    total_execution_time = time.time() - total_start_time
    hours, remainder = divmod(total_execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save detailed metrics to file
    metrics_file = CONFIG['metrics_file']
    with open(metrics_file, 'w') as f:
        f.write("===== VMAE PERFORMANCE METRICS =====\n\n")
        f.write(f"Dataset: Kinetics-400 {args.split.upper()} Split (Balanced)\n")
        f.write(f"Samples per class: {args.samples_per_class}\n")
        f.write(f"Model: {CONFIG['model_id']}\n")
        f.write(f"Number of samples evaluated: {len(all_labels)}\n\n")
        
        f.write("===== ACCURACY METRICS =====\n")
        f.write(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%\n\n")
        
        f.write("===== CLASS DISTRIBUTION =====\n")
        f.write(f"{'Class Name':<40} {'Videos Found':<15} {'Videos Evaluated':<20} {'Accuracy':<10}\n")
        f.write("-" * 85 + "\n")
        
        # Sort by class name for easier reading
        sorted_by_name = sorted(class_video_counts.items())
        for class_name, count in sorted_by_name:
            class_idx = class_to_idx.get(class_name, -1)
            
            # Set accuracy to N/A if no videos were evaluated for this class
            if evaluated_videos_per_class.get(class_name, 0) == 0:
                accuracy = "N/A"
            else:
                accuracy = class_accuracy.get(class_idx, 0.0) * 100 if class_idx in class_accuracy else "N/A"
            
            if isinstance(accuracy, str):
                f.write(f"{class_name:<40} {count:<15} {evaluated_videos_per_class.get(class_name, 0):<20} {accuracy:<10}\n")
            else:
                f.write(f"{class_name:<40} {count:<15} {evaluated_videos_per_class.get(class_name, 0):<20} {accuracy:.2f}%\n")
        
        f.write("\n===== TOP 10 CLASSES BY ACCURACY =====\n")
        # Filter out classes with no evaluated videos
        valid_sorted_classes = [(class_idx, acc) for class_idx, acc in sorted_classes 
                               if evaluated_videos_per_class.get(idx_to_class[class_idx], 0) > 0]
        
        for i, (class_idx, acc) in enumerate(valid_sorted_classes[:10]):
            class_name = idx_to_class[class_idx]
            eval_count = evaluated_videos_per_class.get(class_name, 0)
            found_count = class_video_counts.get(class_name, 0)
            f.write(f"{i+1}. {class_name}: {acc * 100:.2f}% (evaluated: {eval_count}/{found_count})\n")
        f.write("\n")
        
        f.write("===== BOTTOM 10 CLASSES BY ACCURACY =====\n")
        for i, (class_idx, acc) in enumerate(valid_sorted_classes[-10:]):
            class_name = idx_to_class[class_idx]
            eval_count = evaluated_videos_per_class.get(class_name, 0)
            found_count = class_video_counts.get(class_name, 0)
            f.write(f"{i+1}. {class_name}: {acc * 100:.2f}% (evaluated: {eval_count}/{found_count})\n")
        f.write("\n")
        
        # List all classes with no videos found
        if classes_with_no_samples:
            f.write("===== CLASSES WITH NO VIDEOS FOUND =====\n")
            for class_name in sorted(classes_with_no_samples):
                f.write(f"  {class_name}\n")
            f.write("\n")
        
        f.write("===== TOP-5 ACCURACY ANALYSIS =====\n")
        top1_correct = sum(1 for i in range(len(all_labels)) if all_preds[i] == all_labels[i])
        top5_only_correct = sum(1 for i in range(len(all_labels)) 
                              if all_preds[i] != all_labels[i] and all_labels[i] in all_top5_preds[i])
        
        f.write(f"Correctly classified in Top-1: {top1_correct} ({top1_correct/len(all_labels)*100:.2f}%)\n")
        f.write(f"Correctly classified in Top-5 (but not Top-1): {top5_only_correct} ({top5_only_correct/len(all_labels)*100:.2f}%)\n")
        f.write(f"Total correctly classified in Top-5: {top1_correct + top5_only_correct} ({(top1_correct + top5_only_correct)/len(all_labels)*100:.2f}%)\n")
    
    print(f"Detailed metrics saved to {metrics_file}")

if __name__ == "__main__":
    main() 