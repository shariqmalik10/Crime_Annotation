# CCTV Crime Annotation
## Our fyp topic: Incorporate Interactive and transfer learning for automated video annotation

In this project, we have developed an application that integrates three models: object detection, human action recognition, and scene detection. 
The application is designed to analyze videos and provide insights based on these three models.

## Setup

```shell
git clone https://github.com/atou0004/Crime_Annotation
cd Crime_Annotation
conda create -f fyp_environment.yaml
conda activate lastfyp
```

### Files and Folders Setup 
- [Files & Folders to download from google drive](https://drive.google.com/drive/folders/1m47PDxF1QzKEtduDrLCAYN2tRLLBjwV1?usp=share_link)

  Crime_Annotation/yolov7/best.pt
  
  Crime_Annotation/yolov7/traced_model.pt
  
  Crime_Annotation/traced_model.pt
  
  Crime_Annotation/yolov7/yolov7.pt
  
  Crime_Annotation/yolov7/runs
  (this whole folder (proof of tranfer learning done) is 28.3 GB)

  =========================================================================================

  Crime_Annotation/mmaction2/checkpoints/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb_20221219-f984f5d2.pth

  =========================================================================================
  
  Crime_Annotation/Scene_Detection/RealESRGAN_x4plus.pth
  
  Crime_Annotation/Scene_Detection/wideresnet18_places365.pth.tar
  
  Crime_Annotation/Scene_Detection/image_super_resolution/wideresnet18_places365.pth.tar

  =========================================================================================

### Notes in order for the gui to run successfully

  There are absolute paths here and there that you need to change, in order for the app to run successfully

  our working directory: "..../Crime_Annotation"

  so just change the working directory (for both files) to: path/to/Crime_Annotation


  - ### User Interface file - gui.py
  
    working_dir line 17
    
    C:/Users/Tee/Desktop/FYP/GitFYP/Crime_Annotation/mmaction2 line 54 & 55
  
    There isn't a need to change the name of the env, if the user is following the instructions of creating the env using the yaml file
    
    the_cmd = 'conda run -n lastfyp python analyse_vid.py {}'.format(vid_file) line 60 (the name of the env this command is running on)



  - ### run_placesCNN_unified.py
  
    file_name_category line 69
    
    file_name_IO line 80
    
    file_name_attribute line 93
    
    file_name_W line 100
    
    model_file line 124

## Discussion on the models used 

### Action Detection Model 
[Notes about human_action_recognition model (mmaction2's UniFormerV2):
](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/README.md)

We ran our code on cpu, because our device on has 4GB of dedicated GPU memory, our har requires more than that. So if your GPU on your device has a at least 8GB of dedicated GPU memory, you can just comment line 16 and uncomment line 14 (for both analyse_vid and analyse_vid2), so that it uses that gpu, instead of cpu for a faster runtime.

'analyse_vid.py' is for listing out top5 actions in the video

'analyse_vid2.py' is for saving the top5 actions in a json file and saving the analysed videos

Here, these requirements, e.g., code standards, are not that strict as in the core package. Thus, developers from the community can implement their algorithms much more easily and efficiently in MMAction2. We appreciate all contributions from community to make MMAction2 greater.

#### Preview of the action detection model in action 
Will be added in the next section 

### Scene Detection Model 
A Pre-trained model also known as a ‘Places365CNN’ model was taken and on that model we added preprocessing steps to make it adapt to a new task which is scene annotation in CCTV footage. The pre-processing steps included slicing off a frame of the video from the halfway point of the video(as a major assumption we are asked to take in this project by our supervisor is to assume that all videos that are going to be used as input by the models contain only one scene) and then the frame was used as input for super resolution models to enhance the quality of the taken snapshot. 

#### Preview of the scene detection model in action 
<img src="https://github.com/shariqmalik10/Crime_Annotation/blob/dd706e36daeb1e741cf821c893c1f86c9f12c9df/scene_detection_demo.png" width="50%" alt="Demo Of Scene Detection Model">

### Object detection model 
A combination of object detection models has been implemented with the participation of transfer learning techniques. State-to-the-art model in computer vision called 'You Only Look Once (version 7)'  was chosen as the base model responsible for object detection in the crime video. The combination consists of two Yolov7 models, one of the models (pre-trained) is tasked to detect common objects (80 classes in COCO dataset) while another customised model is primarily trained to detect hand-sized objects and light weapons in a video. The fusion of these models is done efficiently by the concatenation of the output in a dictionary.  Also, it is worth mentioning that the ultimate model accepts both video and photo as input.

#### Preview of the model in action 
<img src="https://github.com/shariqmalik10/Crime_Annotation/blob/dd706e36daeb1e741cf821c893c1f86c9f12c9df/object_detection_demo.png" width="50%" alt="Demo Of Object Detection Model">

## Discussion on the model as a whole 

### The final UI 
The UI was created using the Python framework Streamlit. 
We will be going through the steps that are involved in obtaining a prediction 

1. The base UI.
   
   <img src="https://github.com/shariqmalik10/Crime_Annotation/blob/b00dae2273bfcd95141dcc997057e2b9b8fc7689/Step1.png" width="50%" alt="Demo">   
   
   Here we can see a simple UI with the option of uploading a video from your personal device. (Note: Upload .mp4 files only in order to ensure there are no issues at later stage)

3. Video preview
   
   <img src="https://github.com/shariqmalik10/Crime_Annotation/blob/b00dae2273bfcd95141dcc997057e2b9b8fc7689/Step2.png" width="50%" alt="Demo">   
   
   Here you can preview the video selected for model prediction

5. Scene Prediction
   
   <img src="https://github.com/shariqmalik10/Crime_Annotation/blob/b00dae2273bfcd95141dcc997057e2b9b8fc7689/Step3.png" width="50%" alt="Demo">

7. Action Detection
   
   <img src="https://github.com/shariqmalik10/Crime_Annotation/blob/b00dae2273bfcd95141dcc997057e2b9b8fc7689/Step4.png" width="50%" alt="Demo">

9. Object Detection
    
   <img src="https://github.com/shariqmalik10/Crime_Annotation/blob/b00dae2273bfcd95141dcc997057e2b9b8fc7689/Step5.png" width="50%" alt="Demo">

## Issues/Limitations
- A major reason for not hosting the website on a server was due to limitation of time.  
- Another issue we ran into was being unable to run the application on other devices due to the fact that the object detection model , YoloV7, had a CUDA implementation which meant that users using a Mac or Linux device would not be able to run it at all while users using a Windows device would need a high end Nvidia graphics card in order to run the models within a reasonable tikme frame. The models were being run on a Windows device which had a Nvidia RTX 3060 graphics card and took around 10 mins to run the object detection model.
- The setup process is quite complicated. This is due to the fact that some of the models used had conflicting package requirements and so we had to spend a lot of time on working out which models had common package version requirements as well. This is a big issue with going with the transfer learning approach. 
