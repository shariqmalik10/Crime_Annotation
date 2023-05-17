import os
import subprocess
import numpy as np
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description='Human Action Recognition')
    parser.add_argument('input_file', help='input video file')
    args = parser.parse_args()

    # Call the action recognition function with the input video file
    # Only takes one video as input
    print(human_action_recognition(args.input_file))

def human_action_recognition(input_video):
    # Analysing one vid at a time
    # Path to the label map file
    label_map_file = 'tools/data/kinetics/label_map_k600.txt'

    # Set up the command for running action recognition
    # uniformerv2 human action recognition model
    #the_cmd = 'python demo/demo_inferencer.py {} --print-result --rec configs/recognition/uniformerv2/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb.py --rec-weights checkpoints/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb_20221219-f984f5d2.pth --label-file {}'
    # for cpu users, use the code below and comment the code above
    the_cmd = 'python demo/demo_inferencer.py {} --print-result --device cpu --rec configs/recognition/uniformerv2/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb.py --rec-weights checkpoints/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb_20221219-f984f5d2.pth --label-file {}'

    if input_video is None:
        raise ValueError("Input video file is required")
    
    # Get the input folder path
    input_folder = os.path.dirname(input_video)

    # Set up the command to run action recognition on the input video
    command = the_cmd.format(input_video, label_map_file)

    # Run the command
    output = subprocess.run(command, shell=True, capture_output=True)
    # Convert the standard output of the command into a readable string
    output_str = output.stdout.decode('utf-8').replace(command, '')

    # Regular expression pattern to match the dict
    pattern = r"\{.*\}"

    # extract dictionary using regular expression
    match = re.search(pattern, output_str)
    result_dict = None
    if match:
        result_dict = eval(match.group())
    
    # get top5 labels
    # converting the dict to string so that it can be split in the get_top5_labels function
    if result_dict is None:
        raise RuntimeError("Model is not working properly")
    else:
        return get_top5_labels(str(result_dict), label_map_file)
    
def get_top5_labels (the_dict, label_map_file):
    # Takes the output pred dict and label file as input
    # Extract the scores from the output string
    scores_str = the_dict.split('[')[-1].split(']')[0]
    scores = np.fromiter(scores_str.split(','), dtype=np.float32)

    # Load the label map file
    with open(label_map_file, 'r') as f:
        labels = f.read().splitlines()

    # Get the indices that would sort the scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    # Get the top 5 actions and their scores
    top5_indices = sorted_indices[:5]

    # Get the labels for the top 5 actions
    top5_labels = [labels[idx] for idx in top5_indices]

    return top5_labels


if __name__ == '__main__':
    main()
