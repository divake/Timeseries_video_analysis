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
import scipy.stats  # For entropy calculation

# Add the current directory to the path to import from existing scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from existing scripts
from huggingface_video_mae_eval_main import download_and_save_model
from balanced_dataset_eval import get_balanced_video_files

def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from Kinetics-400 videos using VideoMAE')
    parser.add_argument('--output_dir', type=str, default='/ssd_4TB/divake/vivit_kinetics400/kinetics400_features_extended',
                        help='Directory to save extracted features')
    parser.add_argument('--samples_per_class', type=int, default=10,
                        help='Number of samples per class for balanced sampling (-1 for all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use for extraction')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to sample from each video')
    parser.add_argument('--gpu', type=str, default='1',
                        help="GPU selection: 'all' for all GPUs, '0' for GPU 0 only, '1' for GPU 1 only, '0,1' for both GPUs")
    parser.add_argument('--no_multi_gpu', action='store_true',
                        help='Disable multi-GPU even if multiple are available')
    parser.add_argument('--model_id', type=str, default='MCG-NJU/videomae-base-finetuned-kinetics',
                        help='HuggingFace model ID or local path')
    parser.add_argument('--model_save_path', type=str, default='videomae_model',
                        help='Path to save the downloaded model')
    parser.add_argument('--kinetics_dir', type=str, default='/ssd_4TB/divake/vivit_kinetics400/k400',
                        help='Base directory for Kinetics-400 dataset')
    parser.add_argument('--annotations_dir', type=str, default=None,
                        help='Directory containing cleaned annotation files (optional)')
    parser.add_argument('--extract_attention', action='store_true',
                        help='Extract attention maps from the model')
    return parser.parse_args()

def calculate_confidence_metrics(logits):
    """Calculate confidence metrics from logits."""
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # Get top probabilities and indices
    sorted_probs = np.sort(probabilities)[::-1]
    top1_prob = sorted_probs[0]
    top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
    
    # Calculate margin between top-1 and top-2
    margin = top1_prob - top2_prob
    
    # Calculate entropy of the probability distribution
    entropy = scipy.stats.entropy(probabilities)
    
    return {
        'max_prob': float(top1_prob),
        'margin': float(margin),
        'entropy': float(entropy)
    }

def get_video_metadata(video_path):
    """Extract metadata from a video file."""
    metadata = {
        'path': video_path,
        'filename': os.path.basename(video_path),
        'duration': 0,
        'frame_count': 0,
        'fps': 0,
        'width': 0,
        'height': 0,
        'resolution': '0x0',
        'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Get video properties
            metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['fps'] = float(cap.get(cv2.CAP_PROP_FPS))
            metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['resolution'] = f"{metadata['width']}x{metadata['height']}"
            
            # Calculate duration
            if metadata['fps'] > 0 and metadata['frame_count'] > 0:
                metadata['duration'] = metadata['frame_count'] / metadata['fps']
            
            cap.release()
    except Exception as e:
        print(f"Error extracting metadata from {video_path}: {str(e)}")
    
    return metadata

def process_video(video_path, model, processor, device, num_frames=16, extract_attention=False):
    """Process a single video and extract features, logits, and attention maps."""
    try:
        # Get video metadata first
        video_metadata = get_video_metadata(video_path)
        
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
            outputs = model(**inputs, output_hidden_states=True, output_attentions=extract_attention)
            
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
            
            # Extract attention maps if requested
            attention_maps = None
            if extract_attention and hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # Get attention maps from all layers
                attention_maps = [attn.cpu().numpy() for attn in outputs.attentions]
        
        # Calculate confidence metrics
        confidence_metrics = calculate_confidence_metrics(logits.cpu().numpy()[0])
        
        return features.cpu().numpy()[0], logits.cpu().numpy()[0], confidence_metrics, video_metadata, attention_maps
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def process_batch(batch_video_paths, model, processor, device, num_frames=16, extract_attention=False):
    """Process a batch of videos and extract features."""
    results = []
    
    for video_path in batch_video_paths:
        features, logits, confidence, metadata, attention = process_video(
            video_path, model, processor, device, num_frames, extract_attention
        )
        if features is not None and logits is not None:
            results.append({
                'features': features,
                'logits': logits,
                'confidence': confidence,
                'metadata': metadata,
                'attention': attention,
                'path': video_path
            })
    
    return results

def calculate_accuracy(predictions, labels):
    """Calculate top-1 and top-5 accuracy."""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Top-1 accuracy
    top1_correct = (predictions.argmax(axis=1) == labels).sum()
    top1_accuracy = top1_correct / len(labels) * 100
    
    # Top-5 accuracy
    top5_correct = 0
    for i, label in enumerate(labels):
        if label in np.argsort(predictions[i])[-5:]:
            top5_correct += 1
    top5_accuracy = top5_correct / len(labels) * 100
    
    return top1_accuracy, top5_accuracy

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
            # Keep the user-specified device instead of overriding it
            # This allows using cuda:1 even when CUDA_VISIBLE_DEVICES is set
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
    
    # Create output directory structure
    base_output_dir = args.output_dir
    for split in ['train', 'val', 'test']:
        for subdir in ['features', 'metadata', 'attention']:
            os.makedirs(os.path.join(base_output_dir, split, subdir), exist_ok=True)
    
    # Create a log file
    log_file = os.path.join(base_output_dir, "extraction_log.txt")
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
        
        global_metadata = {
            "num_classes": 400,
            "model_name": args.model_id,
            "model_type": "VideoMAE",
            "dataset": "Kinetics-400",
            "num_frames": args.num_frames,
            "extraction_date": time.strftime('%Y-%m-%d'),
            "extraction_time": time.strftime('%H:%M:%S'),
            "feature_dim": model.config.hidden_size if hasattr(model, 'config') and hasattr(model.config, 'hidden_size') else None,
            "preprocessing": {
                "num_frames": args.num_frames,
                "image_size": processor.size if hasattr(processor, 'size') else None,
                "mean": processor.image_mean if hasattr(processor, 'image_mean') else None,
                "std": processor.image_std if hasattr(processor, 'image_std') else None
            }
        }
        
        # First print the loader for all splits to check if all videos are there
        print(f"\n{'='*50}")
        print(f"CHECKING DATA AVAILABILITY FOR ALL SPLITS")
        print(f"{'='*50}")
        
        for split in splits:
            print(f"\nChecking {split} split data availability...")
            
            # Get balanced video files
            try:
                video_files, class_to_idx, class_video_counts = get_balanced_video_files(
                    base_path=args.kinetics_dir,
                    split=split,
                    samples_per_class=args.samples_per_class if args.samples_per_class > 0 else None,
                    seed=args.seed,
                    annotations_dir=args.annotations_dir
                )
                
                total_videos = len(video_files)
                total_classes = len(class_to_idx)
                
                # Calculate expected videos based on samples_per_class
                expected_videos = args.samples_per_class * total_classes if args.samples_per_class > 0 else "all available"
                
                print(f"Split: {split}")
                print(f"Total classes: {total_classes}")
                print(f"Expected videos: {expected_videos}")
                print(f"Found videos: {total_videos}")
                
                if args.samples_per_class > 0:
                    percentage = (total_videos / (args.samples_per_class * total_classes)) * 100
                    print(f"Data availability: {percentage:.2f}% of requested videos found")
                
                # Count classes with full samples
                classes_with_full_samples = sum(1 for count in class_video_counts.values() 
                                              if count >= args.samples_per_class)
                
                print(f"Classes with full samples: {classes_with_full_samples}/{total_classes} "
                      f"({classes_with_full_samples/total_classes*100:.2f}%)")
                
            except Exception as e:
                print(f"Error checking {split} split: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*50}")
        print(f"STARTING FEATURE EXTRACTION FOR ALL SPLITS")
        print(f"{'='*50}")
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            
            # Get balanced video files
            try:
                video_files, class_to_idx, _ = get_balanced_video_files(
                    base_path=args.kinetics_dir,
                    split=split,
                    samples_per_class=args.samples_per_class if args.samples_per_class > 0 else None,
                    seed=args.seed,
                    annotations_dir=args.annotations_dir
                )
                
                print(f"Found {len(video_files)} videos for {split} split")
                
                # Track statistics
                total_videos = len(video_files)
                successful_videos = 0
                failed_videos = 0
                start_time = time.time()
                
                # For accuracy calculation
                all_predictions = []
                all_labels = []
                
                # Process videos in batches
                for i in tqdm(range(0, len(video_files), args.batch_size), desc=f"Processing {split} videos in batches"):
                    batch_end = min(i + args.batch_size, len(video_files))
                    batch_video_info = video_files[i:batch_end]
                    
                    # Extract video paths and labels
                    batch_paths = [video_info['path'] for video_info in batch_video_info]
                    batch_labels = [video_info['label'] for video_info in batch_video_info]
                    batch_ids = [video_info['id'] if 'id' in video_info else f"{split}_{idx}" 
                                for idx, video_info in enumerate(batch_video_info, start=i)]
                    
                    # Process batch
                    batch_results = process_batch(
                        batch_paths, model, processor, device, 
                        args.num_frames, args.extract_attention
                    )
                    
                    # Save results for each video
                    for result, video_id, label in zip(batch_results, batch_ids, batch_labels):
                        if result:
                            # Save feature file (features, logits, label, confidence)
                            feature_path = os.path.join(base_output_dir, split, 'features', f"video_{video_id}.npz")
                            np.savez_compressed(
                                feature_path,
                                features=result['features'],
                                logits=result['logits'],
                                label=label,
                                confidence=result['confidence']
                            )
                            
                            # Save metadata file
                            metadata_path = os.path.join(base_output_dir, split, 'metadata', f"video_{video_id}.json")
                            with open(metadata_path, 'w') as f:
                                json.dump(result['metadata'], f, indent=2)
                            
                            # Save attention maps if available
                            if args.extract_attention and result['attention'] is not None:
                                attention_path = os.path.join(base_output_dir, split, 'attention', f"video_{video_id}.npz")
                                np.savez_compressed(attention_path, attention=result['attention'])
                            
                            # Store predictions and labels for accuracy calculation
                            all_predictions.append(result['logits'])
                            all_labels.append(label)
                            
                            successful_videos += 1
                        else:
                            failed_videos += 1
                    
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
                        
                        # Calculate and print accuracy if we have processed videos
                        if all_predictions and all_labels:
                            top1_acc, top5_acc = calculate_accuracy(all_predictions, all_labels)
                            print(f"Current Top-1 Accuracy: {top1_acc:.2f}%")
                            print(f"Current Top-5 Accuracy: {top5_acc:.2f}%")
                        
                        if torch.cuda.is_available():
                            for gpu_id in range(torch.cuda.device_count()):
                                print(f"GPU {gpu_id} memory: "
                                      f"{torch.cuda.memory_allocated(gpu_id) / 1024**2:.1f} MB / "
                                      f"{torch.cuda.memory_reserved(gpu_id) / 1024**2:.1f} MB")
                
                # Calculate final accuracy for the split
                if all_predictions and all_labels:
                    top1_acc, top5_acc = calculate_accuracy(all_predictions, all_labels)
                    print(f"\n{'='*50}")
                    print(f"FINAL ACCURACY FOR {split.upper()} SPLIT")
                    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
                    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
                    print(f"{'='*50}")
                
                # Update global metadata with split statistics
                global_metadata[f"{split}_statistics"] = {
                    "total_videos": total_videos,
                    "successful_videos": successful_videos,
                    "failed_videos": failed_videos,
                    "success_rate": f"{successful_videos/(successful_videos+failed_videos)*100:.1f}%" if (successful_videos+failed_videos) > 0 else "N/A",
                    "top1_accuracy": f"{top1_acc:.2f}%" if all_predictions and all_labels else "N/A",
                    "top5_accuracy": f"{top5_acc:.2f}%" if all_predictions and all_labels else "N/A"
                }
                
            except Exception as e:
                print(f"Error processing split {split}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Save global metadata
        metadata_file = os.path.join(base_output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(global_metadata, f, indent=2)
        
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