import sys 
import os 
import glob
import argparse

import numpy as np
import nibabel as nb
from PIL import Image
import cv2

from scipy.ndimage import binary_dilation,binary_erosion
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import label, sum as ndi_sum


def avi_to_png(video_path, verbose=False,skip_existing=False):
    
    """Turns .avi file into a directory of .png files"""
    
    assert os.path.exists(video_path)
    
    # create dir     
    savedir = video_path.replace(".avi", "/")
    os.makedirs(savedir, exist_ok=True)
    print(f"PNGs will be saved to: {savedir}")
    print("Processing...")
    
    # check if files exist 
    if skip_existing and os.path.exists(savedir): 
        # skip if not empty (assumes that all pngs were exported successfully)
        if glob.glob(savedir + "*.png"):
            
            return savedir  
        

    # read 
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()

    # save 
    c = 0
    while success:
                        
        # get right name
        if c < 10: 
            i="000"+str(c)
        elif c < 100:
            i="00"+str(c)
        elif c < 1000:
            i="0"+str(c)
        else:
            i=str(c)
        
        
        # maxes out at 9,999 frames
        if c>9999: 
            break 

        # write
        if verbose:
            print(i)
        _ = cv2.imwrite(savedir+"IM"+str(i)+".png", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        c += 1    
        
    return savedir

def load_pngs(pngdir):
    """ loads pngs into a 4D numpy array"""
    
    # get filepaths
    files = sorted(glob.glob(pngdir+"/*png"))

    # get image size 
    x,y,ch = np.array(Image.open(files[0])).shape

    # get num of slices 
    L = len(files)

    # get slices
    slices=np.zeros((x,y,ch,L))
    for i,file in enumerate(files):
        arr = np.array(Image.open(file))
        slices[:,:,:,i] = arr[:,:,:]
        
    return slices 


def check_if_component_changes_significantly(a,slicenum,large_object_threshold=100,dilate_erode_it=5, debug=False):
    """Experimental function
    
    Checks the number of large components in a slice. 
    If the number of components changes after basic erosion - a warning is raised. 
    Assuming that this is caused by an 'unconnected component' (i.e. not closed loop), we try to fix this. 
    We perform dilation and erosion with multiple iterations, hoping that this will connect components. 
    This will only work only if 3-4 voxels are missing (at max). 
    """
    
    # label array with separate components 
    labeled_array, num_features = label(a)

    # find size of each component 
    sizes = ndi_sum(a, labeled_array, range(num_features + 1))
    
    extra_line='If slice # is too high - check by png folder\n'
    erode_dilate=True # default behaviour - unless warning encountered
    
    # check if erosion changes largest component significantly - then try to connect components 
    if np.any(a) and np.max(sizes)>large_object_threshold:
        
        
        a2 = binary_erosion(a,iterations=2)
        
        # label array with separate components 
        labeled_array2, num_features2 = label(a2)

        # find size of each component 
        sizes2 = ndi_sum(a2, labeled_array2, range(num_features2 + 1))
        
        # percentage_diff_in_largest_component_after_erosion
        diff = np.max(sizes2) /np.max(sizes) * 100 
        
        # find large objects - e.g. more than 200 voxels 
        count_large_before=np.sum(sizes>large_object_threshold)
        count_large_after=np.sum(sizes2>large_object_threshold)
        
        # if large object change by more than 5 times - consider this a problem? 
        if diff<20:
            print(f"WARNING: Slice {slicenum} may be incorrectly segmented! Possibly an unconnected segmentation line. Check output! \n{extra_line}")
            erode_dilate=False

            #pick largest 
            # index=np.argmax(sizes)
            # a3=labeled_array==index        
            
            # # pick largest two 
            largest=np.sort(sizes)[-2:]   
            # # only keep objects that fit 'large object' criteria
            largest = largest[largest > large_object_threshold] 
            
            # find index of largest components in size list 
            index=np.isin(sizes,largest)
            # select largest 2 elements indices in original index 
            index2=np.where(index)
            # create array 
            a3 = np.isin(labeled_array, index2) 
            
            # try to connect the components via multiple erosions dilations 
            
            # Adjust the structure and iterations as per the requirement
            structure = np.ones((3, 3))  # This defines the neighborhood over which dilation/erosion is applied
            a3 = binary_dilation(a3, structure=structure, iterations=dilate_erode_it)
            a = binary_erosion(a3, structure=structure, iterations=dilate_erode_it)
            
            # fill holes once again 
            a = binary_fill_holes(a)
                    
            if debug:
                plt.figure(); plt.imshow(a); plt.title('# 5: fixing possibly unconnected component');
            
        # check if there were very large components (e.g. more than 200 voxels) which disappeared 
        
        elif count_large_after!=count_large_before: 
            print(f"WARNING: Slice {slicenum} may be incorrectly segmented! Possibly text over segmentation caused an error. Check output!\n{extra_line}")
            
            # currently this does not work very well. For now - prompt the user to just avoid having numbers within the labels 
            
            # # select only the large components 
            # index=np.where(sizes>large_object_threshold)
            # index
            # a3 = np.isin(labeled_array, index)
            # labeled_array3, num_features3 = label(a3)
            # sizes3 = ndi_sum(a3, labeled_array3, range(num_features3 + 1))
            
            # # Adjust the structure and iterations as per the requirement
            # structure = np.ones((3, 3))  # This defines the neighborhood over which dilation/erosion is applied
            # a3 = binary_dilation(a3, structure=structure, iterations=dilate_erode_it)
            # a = binary_erosion(a3, structure=structure, iterations=dilate_erode_it)

            # if debug:
            #     plt.figure(); plt.imshow(a); plt.title('# 5: fixing possibly unconnected component');
            
            
        
            
            
    return a, erode_dilate

def process_frame2(frame,slice_num, green_threshold=200, erode=2, dilate=1, large_object_threshold=100,dilate_erode_it=5, debug=False):
    """Identifies a segmentation in a 2D slice 
    
    
    Frame must be a 3D array with 3rd channel <> RGB
    
    Can be used as a separate function to debug. Make sure to load 'frame' as numpy array wrapped around PIL.Image.  
    """
    
    # frame to RGB
    aa=frame.astype(np.uint8)
    
    
    a = aa.copy() 
    if debug: 
        # show original
        plt.close('all'); 
        plt.figure(); plt.imshow(a); plt.title('# 0: original');
        


    # remove all values where green channel is less than 200 
    indices=a[:,:,1]<green_threshold
    a[indices] = 0  
    if debug:
        plt.figure(); plt.imshow(a); plt.title('# 1: after remove green channel values less than threshold ');

    # remove all remaining values where R and B channels are greater than 200 
    indices=np.logical_and(a[:,:,0]>green_threshold, a[:,:,2]>green_threshold) 
    a[indices] = 0
    if debug:
        plt.figure(); plt.imshow(a); plt.title('# 2: after removing R and B channel values greater then threshold');


    # make into a mask (only green channel)
    a=a[:,:,1] 
    a[a>0]=1 
    if debug: 
        plt.figure(); plt.imshow(a); plt.title('# 3: after binary seg');
    a = a.astype(bool) # make into a bool (required by below functions)

    # fill holes
    a = binary_fill_holes(a)
    if debug: 
        plt.figure(); plt.imshow(a); plt.title('# 4: after hole filling');
        
    
    a,erode_dilate = check_if_component_changes_significantly(a,slice_num,large_object_threshold,dilate_erode_it, debug)
            

    # erode 
    if erode > 0 and erode_dilate:
        a = binary_erosion(a,iterations=erode)
        if debug:
            plt.figure(); plt.imshow(a); plt.title('# 5: after erosion');

    # dilate 
    if dilate > 0 and erode_dilate: 
        a = binary_dilation(a,iterations=dilate)
        if debug:
            plt.figure(); plt.imshow(a); plt.title('# 6: after dilation'); 


    
    if np.any(a):
        
        ########################
        # count number of objects and choose largest one (since erosion dilation does not work perfectly on T2 images - leaves a lot of unlabelled content)
        ########################
    
    

        # label array with separate components 
        labeled_array, num_features = label(a)
        
        # find size of each component 
        sizes = ndi_sum(a, labeled_array, range(num_features + 1))

        # pick largest 
        index=np.argmax(sizes)
        a=labeled_array==index
        
        if debug:
            plt.figure(); plt.imshow(a); plt.title('# 7: after largest component selected'); 
        
    if debug:
        plt.show();         
        

        

    
    return a


def rotate_png_array_to_nifti(a):
    """Rotates an exported array into correct orientation"""
    return np.moveaxis(np.flipud(a),0,1)

def reshape_nifti(nifti_array):
    """Reshapes a 4D nifti into a 3D array, that will match to the PNG exported array"""
    x,y,z,t = nifti_array.shape
    a=np.moveaxis(nifti_array,-1,-2)
    a=np.reshape(a, (x,y,z*t))
    
    return a

def reshape_nifti_back(a, original_dims):
    
    """Reshapes a 3D array back into original form"""
    x,y,z,t = original_dims
    a=np.reshape(a, (x,y,t,z))
    a=np.moveaxis(a,-1,-2)
    
    return a

def rescale_png_array_to_nifti(png_array, shape):
    
    """Will rescale segmentations to the same size as a reference nifti file"""
    # resize segmentations (derived from annotations) to be the same size as nifti
    if len(shape) == 4:
        xx,yy,zz,tt = shape
    else: 
        xx,yy,zz = shape
        
    out = []
    for i,f in enumerate(np.moveaxis(png_array, -1,0)):
        f = cv2.resize(f, dsize=(yy, xx), interpolation=cv2.INTER_NEAREST)
        out.append(f)
    out = np.moveaxis(np.array(out), 0,-1)  # rotate frames back to original form 
    return out
    
def main(videofile,niftifile, args):
    # splitvideo
    pngdir=avi_to_png(videofile, verbose=False, skip_existing=args.skip_existing)

    # load files
    frames=load_pngs(pngdir)

    # process all annotated frames
    x,y,ch,t=frames.shape
    newframes = np.zeros((x,y,t))
    for i, sl in enumerate(np.moveaxis(frames,-1,0)):
        newframes[:,:,i] = process_frame2(sl,i+1, args.green_threshold,args.erode,args.dilate,args.large_object_threshold,args.dilate_erode_it)
    newframes=np.array(newframes)

    # load nifti 
    assert os.path.exists(niftifile)
    niio = nb.load(niftifile)
    nii = niio.get_fdata()
    
    
    # reshape nifti 
    if nii.ndim ==4: 
        nii = reshape_nifti(nii) 

    # rotate frames to align with nifti 
    newframes2=rotate_png_array_to_nifti(newframes)

    # rescale the image to the same size as a reference nifti shape
    newframes3=rescale_png_array_to_nifti(newframes2, niio.shape)
        
    if niio.ndim == 4:
        if args.dwi: 
            # [quick hack] [in PACS DWI-MRI, TRACEW images do NOT include b0 image, and therefore the original image will have one bvalue less]
            # we perform this quick check/hack here to get around this issue. 
            x,y,slices,bvals = niio.shape 
            if slices*(bvals-1)==newframes3.shape[-1]:
                shape = x,y,slices,bvals-1
            else:
                shape = x,y,slices,bvals
        else: 
            print("warning - processing 4D file that has not been marked as --dwi -> if there is an error make sure to include --dwi as input argument")
            shape = niio.shape 
                        
        # reshape back to original from (x,y,z,t), instead of (x,y,t*z)
        newframes4 = reshape_nifti_back(newframes3, shape)        
            
    else: 
        newframes4 = newframes3

    # add all the bvalue channels of the segmentation together and create segmentation
    if newframes4.ndim == 4:
        seg = np.sum(newframes4, axis=-1)
    else: 
        seg = newframes4
    seg[seg>=1]=1
    seg[seg<1]=0

    # save as segmentation file 
    segmetationo = nb.Nifti1Image(seg, affine=niio.affine, header=niio.header)
    ext = ".nii.gz" if niftifile.endswith(".gz") else ".nii"
    savename=niftifile.replace(ext, "_seg"+ext)
    nb.save(segmetationo, savename)
    print(f"Segmentation saved as {savename}")

if __name__=="__main__":

    
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument("videofile", type=str, help="Path to the video file")
    parser.add_argument("niftifile", type=str, help="Path to the NIfTI file")
    
    # threshold for detecting green values in RGB
    parser.add_argument("--green_threshold", type=float, default=200., help="RGB threshold above which all values are kept (200 is considered default for green values)")
    
    # erosion - dilation will be performed to remove written text from each frame 
    parser.add_argument("--erode", type=int, default=2, help="will erode segmentation image n-times. Default =2")
    parser.add_argument("--dilate", type=int, default=1, help="will dilate segmentation image n-times. Default = 1, used after erosion. i.e. erodes edge of segmentation by 1 voxel.")
    
    # detecting unconnected objects (i.e. bad segmentation )
    parser.add_argument("--large_object_threshold", type=float, default=100., help="Threshold above which we will perform additional checks on the image")
    parser.add_argument("--dilate_erode_it", type=int, default=5, help="if unconnected line is detected - we will erode-dilate over much larger region")
    
    
    # debug features 
    parser.add_argument("--dwi", action="store_true", help="performs extra check on dwi images. (e.g. b0 is not exported in some cases). Is not necessary by default. ")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing png files (i.e. if code was alreary run before)")
    
    
    args = parser.parse_args()

    videofile=args.videofile
    niftifile=args.niftifile
    
    
    
    # basic checks
    assert os.path.exists(videofile)
    assert os.path.exists(niftifile)
    assert videofile.endswith(".avi")
    assert niftifile.endswith(".nii") or niftifile.endswith(".nii.gz")

    main(videofile,niftifile, args)
