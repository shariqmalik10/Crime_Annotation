import argparse
import os
import subprocess
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Human Action Recognition')
    parser.add_argument('input_file', help='input video file')
    args = parser.parse_args()

    # Call the action recognition function with the input video file
    # Only takes one video as input
    print(human_action_recognition(args.input_file))

def get_both_vid_and_top5labels(input_video):
    # Analysing one vid at a time
    # Path to the label map file
    label_map_file = 'tools/data/kinetics/label_map_k600.txt'

    # Set up the command for running action recognition
    #the_cmd = 'python demo/demo_inferencer.py {} --vid-out-dir {} --pred-out-file {} --rec configs/recognition/uniformerv2/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb.py --rec-weights checkpoints/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb_20221219-f984f5d2.pth --label-file {}'
    # for cpu users, use the code below and comment the code above
    the_cmd = 'python demo/demo_inferencer.py {} --vid-out-dir {} --pred-out-file {} --device cpu --rec configs/recognition/uniformerv2/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb.py --rec-weights checkpoints/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb_20221219-f984f5d2.pth --label-file {}'
    
    # Set up the input and output file paths
    video_path = input_video
    output_folder = os.path.join(os.path.dirname(video_path), 'results')
    pred_out_file = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + '.json')

    # Run action recognition on the input video
    command = the_cmd.format(video_path, output_folder, pred_out_file, label_map_file)

    # Run the command
    subprocess.run(command, shell=True)

    # Load the recognition results for this video
    with open(pred_out_file) as f:
        data = json.load(f)

    scores = np.array(data['predictions'][0]['rec_scores'][0])
    sorted_indices = np.argsort(scores)[::-1]

    # Get the top 5 actions
    top5_indices = sorted_indices[:5]
    top5_actions = [str(idx) for idx in top5_indices]

    # Load the label map
    with open(label_map_file, 'r') as f:
        labels = f.read().splitlines()

    # Get the labels for the top 5 actions
    top5_labels = [labels[idx] for idx in top5_indices]

    # Write the list to the output file
    with open(pred_out_file, 'w') as f:
        json.dump(top5_labels, f)
    
    return(top5_labels)

if __name__ == '__main__':
    main()
