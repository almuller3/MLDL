import cv2  # Import OpenCV library for video processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from transformers import CLIPProcessor, CLIPModel  # Import CLIP model and processor from Hugging Face
import torch  # Import PyTorch for deep learning operations
import time  # Import time for measuring execution time

# Function to extract frames from a video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)  # Open the video file specified by the path
    frames = []  # Initialize an empty list to store frames
    while cap.isOpened():  # Loop while the video is open
        ret, frame = cap.read()  # Read the next frame
        if not ret:  # If no frame is returned (end of video), break the loop
            break
        frames.append(frame)  # Add the frame to the list
    cap.release()  # Release the video capture object
    return frames  # Return the list of frames

# Function to calculate deltas between frames
def calculate_deltas(frames):
    deltas = []  # Initialize an empty list to store frame deltas
    previous_frame = None  # Initialize a variable to hold the previous frame
    for frame in frames:  # Iterate through each frame
        if previous_frame is None:  # If it's the first frame, skip the delta calculation
            previous_frame = frame  # Set the current frame as the previous frame
            continue  # Continue to the next frame
        delta = cv2.absdiff(previous_frame, frame)  # Calculate the absolute difference between frames
        deltas.append(delta)  # Add the delta to the list
        previous_frame = frame  # Update the previous frame
    return deltas  # Return the list of deltas

# Function to filter significant changes based on a threshold
def filter_significant_changes(deltas, threshold=30):
    significant_frames = []  # Initialize a list to store indices of significant frames
    for i, delta in enumerate(deltas):  # Iterate through deltas with their index
        if np.mean(delta) > threshold:  # Check if the mean of the delta is greater than the threshold
            significant_frames.append(i + 1)  # Add the index (adjusted for 1-based index) to the list
    return significant_frames  # Return the list of significant frame indices

# Function to process relevant frames with the CLIP model
def process_relevant_frames(frames, significant_indices, model_name='openai/clip-vit-base-patch32'):
    model = CLIPModel.from_pretrained(model_name)  # Load the CLIP model
    processor = CLIPProcessor.from_pretrained(model_name)  # Load the CLIP processor

    relevant_frames = [frames[i] for i in significant_indices]  # Select only the significant frames
    inputs = processor(images=relevant_frames, return_tensors="pt")  # Preprocess the frames for the model
    
    with torch.no_grad():  # Disable gradient computation
        outputs = model.get_image_features(**inputs)  # Get image features from the model
    
    return outputs  # Return the model's output features

# Main function to process the video and measure efficiency
def process_video_for_llm(video_path, threshold=20, model_name='openai/clip-vit-base-patch32'):
    start_time = time.time()  # Record the start time
    
    # Extract frames
    start_extract = time.time()  # Record the start time for frame extraction
    frames = extract_frames(video_path)  # Extract frames from the video
    end_extract = time.time()  # Record the end time for frame extraction
    
    # Calculate deltas
    start_deltas = time.time()  # Record the start time for delta calculation
    deltas = calculate_deltas(frames)  # Calculate deltas between consecutive frames
    end_deltas = time.time()  # Record the end time for delta calculation
    
    # Filter significant changes
    start_filter = time.time()  # Record the start time for filtering
    significant_indices = filter_significant_changes(deltas, threshold)  # Filter frames based on significant changes
    end_filter = time.time()  # Record the end time for filtering
    
    # Process relevant frames
    start_process = time.time()  # Record the start time for frame processing
    features = process_relevant_frames(frames, significant_indices, model_name)  # Process the significant frames with CLIP
    end_process = time.time()  # Record the end time for frame processing
    
    end_time = time.time()  # Record the total end time
    
    # Print time taken for each step
    print(f"Total time taken: {end_time - start_time:.2f} seconds")  # Print the total time taken
    print(f"Time to extract frames: {end_extract - start_extract:.2f} seconds")  # Print the time taken to extract frames
    print(f"Time to calculate deltas: {end_deltas - start_deltas:.2f} seconds")  # Print the time taken to calculate deltas
    print(f"Time to filter significant changes: {end_filter - start_filter:.2f} seconds")  # Print the time taken to filter changes
    print(f"Time to process relevant frames: {end_process - start_process:.2f} seconds")  # Print the time taken to process frames
    
    # Print intermediate results
    print(f"Total frames: {len(frames)}")  # Print the total number of frames
    print(f"Significant frames: {len(significant_indices)}")  # Print the number of significant frames
    print(f"Features shape: {features.shape}")  # Print the shape of the output features
    
    # Visualize some frames
    visualize_frames(frames, deltas, significant_indices)  # Call the visualization function
    
    return features  # Return the features extracted by the model

# Function to visualize frames, deltas, and significant frames
def visualize_frames(frames, deltas, significant_indices):
    num_frames = min(len(frames), 5)  # Set the number of frames to visualize (up to 5)
    
    plt.figure(figsize=(15, 10))  # Create a figure for the plots
    
    for i in range(num_frames):  # Loop to visualize frames
        plt.subplot(3, num_frames, i + 1)  # Create a subplot for each frame
        plt.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB and display the frame
        plt.title(f'Frame {i}')  # Set the title as the frame index
        plt.axis('off')  # Turn off the axis
    
    for i in range(num_frames - 1):  # Loop to visualize deltas
        plt.subplot(3, num_frames, num_frames + i + 1)  # Create a subplot for each delta
        plt.imshow(deltas[i], cmap='gray')  # Display the delta as a grayscale image
        plt.title(f'Delta {i+1}')  # Set the title as the delta index
        plt.axis('off')  # Turn off the axis
    
    for i in range(min(len(significant_indices), 5)):  # Loop to visualize significant frames
        plt.subplot(3, num_frames, 2 * num_frames + i + 1)  # Create a subplot for each significant frame
        plt.imshow(cv2.cvtColor(frames[significant_indices[i]], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB and display the frame
        plt.title(f'Significant {significant_indices[i]}')  # Set the title as the frame index
        plt.axis('off')  # Turn off the axis
    
    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the figure

# Example usage
video_path = '/mnt/c/Users/al/Als Stuff/video algorithm/Em and Z dancing.MOV'  # Path to the video file -- needs to be changed
features = process_video_for_llm(video_path)  # Process the video and extract features using the LLM
