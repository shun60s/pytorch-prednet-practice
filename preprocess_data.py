# -*- coding: utf-8 -*-


#
#  make a dataset for prednet, from mp4 files
#



import os
import argparse
import glob
import cv2
import numpy as np
import hickle as hkl




parser = argparse.ArgumentParser(description='make dataset from mp4 files')
parser.add_argument(
    '--mp4-dir',
    default='mp4/',
    metavar='LM',
    help='input folder of mp4')
parser.add_argument(
    '--out-dir',
    default='dataset/',
    metavar='LO',
    help= 'output folder to save dataset')
parser.add_argument(
     '--width',
    default=160, 
    type=int, 
    help='Width of images.')
parser.add_argument(
     '--height',
    default=128, 
    type=int, 
    help='Width of images.')
parser.add_argument(
    '--per-frames',
    type=int,
    default=5,
    help='sample frame number (default: 5)')
parser.add_argument(
    '--grayscale',
    default=False,
    metavar='GS',
    help='convert to grayscale')



args = parser.parse_args()

#
desired_im_sz = (128, 160)


if __name__ == '__main__':
    
    # get mp4file name list
    file_list =  glob.glob( args.mp4_dir + "*.mp4")
    len0= len(file_list)
    train_ratio=0.9
    val_num=3
    len_train= int((len0-val_num)*train_ratio)
    len_test=  len0 - val_num
    
    
    rand_file_list = np.random.choice( file_list, len(file_list),replace=False)
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['train'] = rand_file_list[0: len_train]
    splits['test']  = rand_file_list[len_train : len_test]
    splits['val']   = rand_file_list[len_test:]
    
    
    # load mp4 file
    for split in splits:
        print (split)
        # (1)compute data len
        source_list = []
        c0=0
        for i, infile in enumerate(splits[split]): 
            
            with open( infile, "rt") as fi:
                
                video = cv2.VideoCapture(infile)
                all_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                if args.per_frames == 1:
                    nframe= all_frames_count
                else:
                    nframe= int( (all_frames_count -1) / args.per_frames) + 1
                
                #source_list += [split + '_' + str(i) for l in range(nframe)] #  string にするとhickleでエラーがでる
                source_list += [ i for l in range(nframe)]
                c0 += nframe
                
                video.release()
                
        print ('count all ',c0)
        # (2)concat images
        
        if args.grayscale:
            X = np.zeros((c0,) + (args.height,args.width) + (1,), np.uint8)
        else:
            X = np.zeros((c0,) + (args.height,args.width) + (3,), np.uint8)
        
        c0=0
        for i, infile in enumerate(splits[split]): 
            
            with open( infile, "rt") as fi:
                
                video = cv2.VideoCapture(infile)
                all_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                
                for j in range(all_frames_count):
                    
                    ret, img0 = video.read()
                    # per skip frames
                    if  j %  args.per_frames == 0:
                        
                        if 0:
                            #Bipeadlwalkerの部分を切り出す
                            h, w, c = img0.shape # (400,600,3)
                            y1= int(h * 0.2)
                            y2= int(h)
                            img0 = img0[y1: y2  ,0: int(w/2) ]
                        
                        
                        
                        
                        img0 = cv2.resize(img0, (args.width, args.height))
                        
                        if args.grayscale:
                            # convert to grayscale
                            img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                            #print (img1.shape)
                            #print(img1.dtype)
                        else:
                            img0rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                            img1= img0rgb.astype(np.uint8)
                            #print ('img1.shape', img1.shape)
                        
                        # はじめの10個だけ書かせてみる
                        if c0 < 10:
                            if args.grayscale:
                                cv2.imwrite( os.path.join(args.out_dir, str(i) + '_' + str(j) + '.jpg'), img1)
                            else:
                                cv2.imwrite( os.path.join(args.out_dir, str(i) + '_' + str(j) + '.jpg'), img0)
                        
                        if args.grayscale:
                            X[c0] = np.expand_dims( img1, -1)
                        else:
                            X[c0] = img1
                        
                        c0 +=1
                
                video.release()
                
        print ('count all ',c0)
        # (3) dump out
        hkl.dump(X, os.path.join(args.out_dir, 'X_' + split + '.hkl'))
        #print (source_list)
        hkl.dump(np.array(source_list), os.path.join(args.out_dir, 'sources_' + split + '.hkl'))  # list[数字]ではhickleでエラーがでる
        