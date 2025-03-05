import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import time
from pathlib import Path
import random
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torchvision.transforms as T
from PIL import Image
import cv2  # Add OpenCV for video processing fallback

# Add the current directory to the path to import from existing scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from existing scripts
from huggingface_video_mae_eval_main import download_and_save_model
from balanced_dataset_eval import get_balanced_video_files

def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from Kinetics-400 videos using VideoMAE')
    parser.add_argument('--output_dir', type=str, default='kinetics400_features',
                        help='Directory to save extracted features')
    parser.add_argument('--samples_per_class', type=int, default=10,
                        help='Number of samples per class for balanced sampling (-1 for all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for extraction')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to sample from each video')
    parser.add_argument('--gpu', type=str, default='all',
                        help="GPU selection: 'all' for all GPUs, '0' for GPU 0 only, '1' for GPU 1 only, '0,1' for both GPUs")
    parser.add_argument('--no_multi_gpu', action='store_true',
                        help='Disable multi-GPU even if multiple are available')
    parser.add_argument('--model_id', type=str, default='MCG-NJU/videomae-base-finetuned-kinetics',
                        help='HuggingFace model ID or local path')
    parser.add_argument('--model_save_path', type=str, default='videomae_model',
                        help='Path to save the downloaded model')
    parser.add_argument('--kinetics_dir', type=str, default='/ssd_4TB/divake/vivit_kinetics400/k400',
                        help='Base directory for Kinetics-400 dataset')
    return parser.parse_args()

def process_video(video_path, model, processor, device, num_frames=16):
    """Process a single video and extract features."""
    try:
        # Check if it's a directory of frames or a video file
        if os.path.isdir(video_path):
            # It's a directory of frames
            frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            # Sample frames evenly
            if len(frame_files) >= num_frames:
                indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
                sampled_frames = [frame_files[i] for i in indices]
            else:
                # If we don't have enough frames, repeat the last one
                sampled_frames = frame_files + [frame_files[-1]] * (num_frames - len(frame_files))
            
            # Load the frames
            video_frames = []
            for frame_file in sampled_frames:
                try:
                    img = Image.open(frame_file).convert('RGB')
                    video_frames.append(img)
                except Exception as e:
                    print(f"Error loading frame {frame_file}: {str(e)}")
                    # Create a blank image as fallback
                    img = Image.new('RGB', (224, 224), color='black')
                    video_frames.append(img)
            
            # Process the frames with the processor
            inputs = processor(video_frames, return_tensors="pt")
        else:
            # It's a video file - use torchvision's read_video with error handling
            try:
                # Read the video file
                from torchvision.io import read_video
                video, _, _ = read_video(video_path, pts_unit="sec")
                
                # Check if video is empty or invalid
                if video.numel() == 0 or video.shape[0] == 0:
                    print(f"Warning: Empty video file {video_path}, using dummy frames")
                    # Create dummy frames
                    dummy_frame = torch.zeros(1, 224, 224, 3, dtype=torch.uint8)
                    video = dummy_frame.repeat(num_frames, 1, 1, 1)
                
                # Handle frame count
                num_video_frames = video.shape[0]
                if num_video_frames < num_frames:
                    # Duplicate frames if video is too short
                    repeat_factor = max(num_frames // max(num_video_frames, 1) + 1, 1)
                    video = video.repeat(repeat_factor, 1, 1, 1)[:num_frames]
                else:
                    # Sample frames uniformly
                    indices = torch.linspace(0, num_video_frames - 1, num_frames).long()
                    video = video[indices]
                
                # Apply image processor (handles normalization and resize)
                # Move channels to correct dimension: [num_frames, channels, height, width]
                video = video.permute(0, 3, 1, 2)
                inputs = processor(list(video), return_tensors="pt")
                
            except Exception as e:
                print(f"Error reading video with torchvision: {str(e)}")
                # Fallback to OpenCV
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    
                    frames = []
                    # Get total frame count
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames > 0:
                        # Sample frames evenly
                        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                        
                        for idx in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                # Convert BGR to RGB
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(frame_rgb)
                            else:
                                # Create a blank frame if reading fails
                                blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                                frames.append(blank_frame)
                    
                    cap.release()
                    
                    # Convert frames to PIL images
                    video_frames = [Image.fromarray(frame) for frame in frames]
                    
                    # If we don't have enough frames, pad with the last frame
                    if len(video_frames) < num_frames:
                        last_frame = video_frames[-1] if video_frames else Image.new('RGB', (224, 224), color='black')
                        video_frames.extend([last_frame] * (num_frames - len(video_frames)))
                    
                    # Process the frames with the processor
                    inputs = processor(video_frames, return_tensors="pt")
                    
                except Exception as cv_error:
                    print(f"Error reading video with OpenCV: {str(cv_error)}")
                    # Create dummy inputs as a last resort
                    dummy_frames = [Image.new('RGB', (224, 224), color='black') for _ in range(num_frames)]
                    inputs = processor(dummy_frames, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            # Get both outputs and hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract features from the hidden states
            # For VideoMAE, we typically use the [CLS] token from the last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Get the last hidden state
                last_hidden_state = outputs.hidden_states[-1]
                # Extract the [CLS] token features (first token)
                features = last_hidden_state[:, 0]  # Shape: [batch_size, hidden_size]
            else:
                # Fallback: use mean pooling over the sequence dimension
                features = outputs.last_hidden_state.mean(dim=1)
            
            # Get the logits for classification
            logits = outputs.logits
        
        return features.cpu().numpy(), logits.cpu().numpy()
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def process_batch(batch_video_paths, model, processor, device, num_frames=16):
    """Process a batch of videos and extract features."""
    batch_features = []
    batch_logits = []
    
    for video_path in batch_video_paths:
        features, logits = process_video(video_path, model, processor, device, num_frames)
        if features is not None and logits is not None:
            batch_features.append(features)
            batch_logits.append(logits)
    
    if batch_features:
        return np.vstack(batch_features), np.vstack(batch_logits)
    else:
        return None, None

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Handle GPU selection
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f"Using GPU(s): {args.gpu}")
        # If we're using specific GPUs, update device
        if ',' in args.gpu:
            args.device = 'cuda'
            use_multi_gpu = not args.no_multi_gpu
        else:
            args.device = 'cuda:0'  # With CUDA_VISIBLE_DEVICES set, this will be the selected GPU
            use_multi_gpu = False
    else:
        use_multi_gpu = not args.no_multi_gpu
    
    # Print CUDA information
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. Using CPU.")
        args.device = 'cpu'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a log file
    log_file = os.path.join(args.output_dir, "extraction_log.txt")
    print(f"Logging to {log_file}")
    
    # Redirect stdout to both console and log file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    
    print(f"\n{'='*50}")
    print(f"Feature extraction started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    print(f"Arguments: {args}")
    
    try:
        # Download and load model
        model_path = download_and_save_model(args.model_id, args.model_save_path)
        
        # Load model and processor
        print(f"Loading model from: {model_path}")
        model = AutoModelForVideoClassification.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Move model to device
        device = args.device
        model = model.to(device)
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        else:
            print(f"Using single device: {device}")
        
        model.eval()
        
        # Process each split
        splits = ['train', 'val', 'test']
        split_mapping = {'val': 'cal'}  # Map 'val' to 'cal' in output files
        
        metadata = {
            "num_classes": 400,
            "model_name": "VideoMAE",
            "dataset": "Kinetics-400",
            "num_frames": args.num_frames,
            "extraction_date": time.strftime('%Y-%m-%d'),
            "extraction_time": time.strftime('%H:%M:%S')
        }
        
        for split in splits:
            output_split = split_mapping.get(split, split)
            print(f"\nProcessing {split} split (saving as {output_split})...")
            
            # Get balanced video files
            try:
                video_files, class_to_idx, _ = get_balanced_video_files(
                    base_path=args.kinetics_dir,
                    split=split,
                    samples_per_class=args.samples_per_class if args.samples_per_class > 0 else None,
                    seed=args.seed
                )
                
                print(f"Found {len(video_files)} videos for {split} split")
                
                # Process videos and extract features
                all_features = []
                all_logits = []
                all_labels = []
                
                # Track statistics
                total_videos = len(video_files)
                successful_videos = 0
                failed_videos = 0
                start_time = time.time()
                
                # Process videos in batches
                for i in tqdm(range(0, len(video_files), args.batch_size), desc=f"Processing {split} videos in batches"):
                    batch_end = min(i + args.batch_size, len(video_files))
                    batch_video_info = video_files[i:batch_end]
                    
                    # Extract video paths and labels
                    batch_paths = [video_info['path'] for video_info in batch_video_info]
                    batch_labels = [video_info['label'] for video_info in batch_video_info]
                    
                    # Process batch
                    batch_features = []
                    batch_logits = []
                    valid_indices = []
                    
                    # Process each video in the batch
                    for j, video_path in enumerate(batch_paths):
                        try:
                            features, logits = process_video(video_path, model, processor, device, args.num_frames)
                            if features is not None and logits is not None:
                                batch_features.append(features)
                                batch_logits.append(logits)
                                valid_indices.append(j)
                                successful_videos += 1
                            else:
                                failed_videos += 1
                                print(f"Failed to process video: {video_path}")
                        except Exception as e:
                            failed_videos += 1
                            print(f"Exception processing video {video_path}: {str(e)}")
                    
                    # Add valid results to the overall lists
                    if batch_features:
                        all_features.extend(batch_features)
                        all_logits.extend(batch_logits)
                        all_labels.extend([batch_labels[j] for j in valid_indices])
                    
                    # Print batch progress
                    if (i // args.batch_size) % 5 == 0 or i + args.batch_size >= len(video_files):  # Print every 5 batches or at the end
                        elapsed = time.time() - start_time
                        videos_processed = min(i + args.batch_size, len(video_files))
                        videos_per_second = videos_processed / max(elapsed, 0.001)
                        
                        print(f"\nProgress: {videos_processed}/{total_videos} videos "
                              f"({videos_processed/total_videos*100:.1f}%) - "
                              f"{videos_per_second:.2f} videos/sec")
                        print(f"Success: {successful_videos}, Failed: {failed_videos}, "
                              f"Success rate: {successful_videos/(successful_videos+failed_videos)*100:.1f}%")
                        
                        if torch.cuda.is_available():
                            for gpu_id in range(torch.cuda.device_count()):
                                print(f"GPU {gpu_id} memory: "
                                      f"{torch.cuda.memory_allocated(gpu_id) / 1024**2:.1f} MB / "
                                      f"{torch.cuda.memory_reserved(gpu_id) / 1024**2:.1f} MB")
                
                # Save results if we have any
                if all_features:
                    # Convert to numpy arrays
                    all_features = np.array(all_features, dtype=np.float32)
                    all_logits = np.array(all_logits, dtype=np.float32)
                    all_labels = np.array(all_labels, dtype=np.int64)
                    
                    # Save features, logits, and labels
                    features_file = os.path.join(args.output_dir, f"kinetics400_{output_split}_features.npy")
                    logits_file = os.path.join(args.output_dir, f"kinetics400_{output_split}_logits.npy")
                    labels_file = os.path.join(args.output_dir, f"kinetics400_{output_split}_labels.npy")
                    
                    np.save(features_file, all_features)
                    np.save(logits_file, all_logits)
                    np.save(labels_file, all_labels)
                    
                    print(f"\nSaved {output_split} data:")
                    print(f"  Features: {all_features.shape} -> {features_file}")
                    print(f"  Logits: {all_logits.shape} -> {logits_file}")
                    print(f"  Labels: {all_labels.shape} -> {labels_file}")
                    
                    # Update metadata
                    metadata[f"{output_split}_samples"] = len(all_labels)
                    metadata[f"{output_split}_success_rate"] = f"{successful_videos/(successful_videos+failed_videos)*100:.1f}%"
                    
                    # Set feature dimension if not already set
                    if "feature_dim" not in metadata:
                        metadata["feature_dim"] = all_features.shape[1]
                else:
                    print(f"No valid features extracted for {split} split")
            except Exception as e:
                print(f"Error processing split {split}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Save metadata
        metadata_file = os.path.join(args.output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nExtraction complete. Metadata saved to {metadata_file}")
        print(f"{'='*50}")
        print(f"Feature extraction completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
    
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore stdout
        if isinstance(sys.stdout, Logger):
            sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc() 