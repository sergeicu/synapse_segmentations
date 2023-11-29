# synapse_segmentations

Loads exported video .avi file and extracts segmentation from Synapse based on a reference nifti file. 
Allows for radiologists to mark segmentations in Synapse, and for researchers to export these as nifti files. 

```
    Usage: python synapse_segmentation.py $video_avi_file $nifti_file [--green_threshold <200>] [--erode <2>] [--dilate <1>] [--large_object_threshold <100>] [--dilate_erode_it <5>]  [--dwi] [--skip_existing]
```

#How to use: 
    1. [Synapse] Open image series with a segmentation. NB Currently the algorithm will only work on segmentations that are closed loop (i.e. connected lines)
    2. [Synapse] Right click and select Export/Print 
    3. [Synapse] Export as 'AVI' file. Make sure that 'Annotations' box is marked. Export 'All Images in a Series'. This will export .avi video. 
    4. [DCM/NIFTI] Provide a path to .nii.gz file that comes from the SAME image where the segmentation was derived. This could be T2 HASTE, DWI image, or any other image series. NB current algorithm tested for T2 and DWI. If using any other contrast - please check result and adjust parameters as necessary. 
    5. [Python] Run `python synapse_segmentation.py $video_avi_file $nifti_file`

# Important considerations: 
    1. Verify that TEXT is not occluding the segmentation lines when exporting the video from Synapse. 
    2. Verify that segmentations are closed loops (lines are connected in a loop)

# Warnings/Errors: 
    1. Ensure that you export .avi file from actual scan and NOT from Bookmarks (last image series on Synapse). 
    2. The system will try to warn you if open loops are detected and give you slice numbers to verify. Check those. 
    3. If segmentations are larger than expected - consider changing 'green_threshold' parameter
    4. If open loops are detected - fix them in Synapse OR consider changing 'dilate_erode_it' parameter
    5. If two or more segmentations are CONNECTED into one segmentation - consider reducing 'erode' parameter
    6. There is a chance that `scipy` library is not matching. Try `pip install scipy==1.11.3`

# Installation 
    1. Create virtual environment (to not break current python libraries) and check path
        `python -m venv venv`
        `source venv/bin/activate`
        `which python`
    2. Run `pip install -r requirements.txt` 

# Additional input options

    ```
    --green_threshold - RGB threshold above which all pixels are kept (200 is considered default for green values - which is the colour of synapse segmentations). 
    
    # 
    --erode - erosion - dilation will be performed to remove written text from each frame. Default 2. 
    --dilate - erosion - dilation will be performed to remove written text from each frame. Default 1
    
    # detecting unconnected objects (i.e. bad segmentation )
    --large_object_threshold - Threshold above which we will perform additional checks on the image. Default 100. 
    --dilate_erode_it - if unconnected segmentation line is detected - we will erode-dilate over much larger region. Default 5. 
    

    --dwi - if used, performs extra check on dwi images. e.g. b0 is not exported in some cases. Is not necessary by default.
    --skip_existing - skip existing png files. Only use if code was alreary run before - skips time consuming steps. 

    ```