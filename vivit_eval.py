import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from transformers import AutoModelForVideoClassification, VivitImageProcessor
from torchvision.io import read_video
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.cuda as cuda
from concurrent.futures import ThreadPoolExecutor
import time
import av
from PIL import Image
import traceback
# Configuration
CONFIG = {
    "kinetics_dir": "/ssd_4TB/divake/vivit_kinetics400/k400",  # Base path to Kinetics dataset
    "val_csv": "/ssd_4TB/divake/vivit_kinetics400/k400/annotations/val.csv",  # Path to validation CSV
    "num_samples": 1000,                            # Set to a smaller number for testing
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use all available GPUs
    "batch_size": 32,                               # Batch size for multi-GPU
    "seed": 42,
    "model_id": "google/vivit-b-16x2-kinetics400",  # Google's ViViT model
    "num_workers": 16,                              # Worker count for parallel processing
    "model_save_path": "vivit_model",               # Path to save the downloaded model
    "use_multi_gpu": True                           # Whether to use multiple GPUs
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

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (int): Total number of frames to sample.
        frame_sample_rate (int): Sample every n-th frame.
        seg_len (int): Maximum allowed index of sample's last frame.
    Returns:
        indices (List[int]): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = min(converted_len, seg_len)
    start_idx = 0
    indices = np.linspace(start_idx, end_idx-1, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx-1).astype(np.int64)
    return indices

def read_video_pyav(container, indices):
    """
    Read video frames using PyAV and return them as a list of NumPy arrays.
    """
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
            
            # Exit early once we have all the frames we need
            if len(frames) == len(indices):
                break
    return frames

def load_model(model_id, save_path):
    """Load model with correct configuration"""
    print(f"Loading model from: {model_id}")
    
    # Load with original dimensions to avoid position embedding mismatch
    processor = VivitImageProcessor.from_pretrained(model_id)
    
    # Get the original config to match position embeddings
    model = AutoModelForVideoClassification.from_pretrained(
        model_id,
        ignore_mismatched_sizes=False  # Don't ignore mismatches
    )
    
    return model, processor

def process_video_optimized(video_path, processor, model):
    """Optimized video processing for better performance"""
    try:
        # Read video frames directly with PyAV for better performance
        container = av.open(video_path)
        
        # Get video stream info
        stream = container.streams.video[0]
        total_frames = stream.frames or 300  # Fallback to a reasonable number
        
        # The original ViViT model expects 16 frames at 224x224
        # Calculate indices for 16 evenly spaced frames
        indices = np.linspace(0, total_frames - 1, 16).astype(int)
        
        # Seek and extract exactly 16 frames
        frames = []
        for idx in indices:
            container.seek(int(idx), stream=stream)
            for frame in container.decode(video=0):
                # Don't resize here, let the processor handle it correctly
                img = frame.to_image()
                frames.append(img)
                break  # Just get one frame at each position
        
        # Make sure we have exactly 16 frames
        if len(frames) < 16:
            # Duplicate last frame if needed
            last_frame = frames[-1] if frames else Image.new('RGB', (224, 224))
            while len(frames) < 16:
                frames.append(last_frame)
        
        # Let the processor handle the exact preprocessing needed
        inputs = processor(
            frames[:16], 
            return_tensors="pt",
            do_resize=True,
            size={"height": 224, "width": 224}
        )
        
        # Don't override the position encoding behavior
        # Only add this if absolutely necessary
        # inputs['interpolate_pos_encoding'] = True
        
        return inputs
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        traceback.print_exc()
        return None

def download_and_save_model(model_id, save_path):
    """Download model and save it locally"""
    if os.path.exists(save_path) and os.path.isdir(save_path) and len(os.listdir(save_path)) > 0:
        print(f"Model already downloaded at {save_path}")
        return save_path
    
    print(f"Downloading model {model_id}...")
    model = AutoModelForVideoClassification.from_pretrained(model_id)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    return save_path

def monitor_gpu_usage():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

# Process videos in parallel using a ThreadPoolExecutor
def process_video_batch(video_infos, processor, model, device):
    results = []
    labels = []
    
    for video_info in video_infos:
        try:
            inputs = process_video_optimized(video_info['path'], processor, model)
            if inputs is None:
                continue
                
            # Move to GPU
            input_dict = {k: v.to(device) for k, v in inputs.items() 
                         if isinstance(v, torch.Tensor)}
            
            # Keep non-tensor values
            for k, v in inputs.items():
                if not isinstance(v, torch.Tensor):
                    input_dict[k] = v
            
            # Run inference
            with torch.no_grad():
                outputs = model(**input_dict)
                prediction = outputs.logits.argmax(-1).item()
            
            results.append(prediction)
            labels.append(video_info['label'])
            
        except Exception as e:
            print(f"Error processing video {video_info['path']}: {e}")
            traceback.print_exc()
    
    return results, labels

def main():
    # Check CUDA availability and print GPU info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
        monitor_gpu_usage()
    
    print(f"Using device: {CONFIG['device']}")
    
    # Load the model correctly with original parameters
    model, processor = load_model(CONFIG['model_id'], CONFIG['model_save_path'])
    
    # Use DataParallel for multi-GPU processing
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
    
    # Process videos in batches with parallelization
    all_preds = []
    all_labels = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
        # Submit batches for parallel processing
        futures = []
        for i in range(0, len(videos), CONFIG['batch_size']):
            batch_videos = videos[i:i + CONFIG['batch_size']]
            futures.append(executor.submit(
                process_video_batch, batch_videos, processor, model, CONFIG['device']
            ))
        
        # Collect results from completed tasks
        for i, future in enumerate(tqdm(futures, desc="Evaluating")):
            batch_preds, batch_labels = future.result()
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
            
            # Print progress
            if i % 10 == 0:
                elapsed = time.time() - start_time
                videos_processed = min((i + 1) * CONFIG['batch_size'], len(videos))
                videos_per_second = videos_processed / elapsed
                print(f"Processed {videos_processed}/{len(videos)} videos "
                      f"({videos_per_second:.2f} videos/sec)")
    
    # Calculate metrics
    if all_preds:
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Get the unique classes that actually appear in the predictions and labels
        unique_classes = sorted(set(all_labels + all_preds))
        
        # Create a mapping from class index to position in confusion matrix
        class_to_matrix_idx = {class_idx: i for i, class_idx in enumerate(unique_classes)}
        
        # Generate confusion matrix with the correct size
        conf_mat = confusion_matrix(all_labels, all_preds, labels=unique_classes)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Total samples evaluated: {len(all_labels)}")
        print(f"Top-1 Accuracy: {accuracy * 100:.2f}%")
        
        # Save results
        np.save("vivit_confusion_matrix.npy", conf_mat)
        print("Confusion matrix saved to vivit_confusion_matrix.npy")
        
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
    else:
        print("No predictions were made. Please check if PyAV is installed correctly.")

if __name__ == "__main__":
    main() 