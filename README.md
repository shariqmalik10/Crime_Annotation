# CCTV Crime Annotation
## Our fyp topic: Incorporate Interactive and transfer learning for automated video annotation

In this project, we have developed an application that integrates three models: object detection, human action recognition, and scene detection. The application is designed to analyze videos and provide insights based on these three models.

- [Files & Folders to download from google drive]()

  Crime_Annotation/yolov7/best.pt
  Crime_Annotation/yolov7/traced_model.pt
  Crime_Annotation/traced_model.pt
  Crime_Annotation/yolov7/yolov7.pt
  Crime_Annotation/yolov7/runs
  ^this whole folder (proof of tranfer learning done) is 28.3 GB

  =========================================================================================

  Crime_Annotation/mmaction2/checkpoints/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb_20221219-f984f5d2.pth

  =========================================================================================
  
  Crime_Annotation/Scene_Detection/RealESRGAN_x4plus.pth
  Crime_Annotation/Scene_Detection/wideresnet18_places365.pth.tar
  Crime_Annotation/Scene_Detection/image_super_resolution/wideresnet18_places365.pth.tar

  =========================================================================================

- [Notes in order for the gui to run successfully]

  There are absolute paths here and there that you need to change, in order for the app to run successfully

  our working directory: "C:/Users/Tee/Desktop/FYP/GitFYP/Crime_Annotation"

  our fyp folder:
  Crime_Annotation

  so just change the working directory (for both files) to:
  path/to/Crime_Annotation


  gui.py
  working_dir line 17 
  C:/Users/Tee/Desktop/FYP/GitFYP/Crime_Annotation/mmaction2 line 54 & 55

  #there isn't a need to change the name of the env, if the user is following the instructions of creating the env using the yaml file
  the_cmd = 'conda run -n lastfyp python analyse_vid.py {}'.format(vid_file) line 60 (the name of the env this command is running on)


  run_placesCNN_unified.py
  file_name_category line 69
  file_name_IO line 80
  file_name_attribute line 93
  file_name_W line 100
  model_file line 124

  =========================================================================================

- [Notes about human_action_recognition model (mmaction2's UniFormerV2):
](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/README.md)

  We ran our code on cpu, because our device on has 4GB of dedicated GPU memory, our har requires more than that. So if your GPU on your device has a at least 8GB of dedicated GPU memory, you can just comment line 16 and uncomment line 14 (for both analyse_vid and analyse_vid2), so that it uses that gpu, instead of cpu for a faster runtime.

  analyse_vid.py is for listing out top5 actions in the video

  analyse_vid2.py is for saving the top5 actions in a json file and saving the analysed videos



  Here, these requirements, e.g., code standards, are not that strict as in the core package. Thus, developers from the community can implement their algorithms much more easily and efficiently in MMAction2. We appreciate all contributions from community to make MMAction2 greater.
