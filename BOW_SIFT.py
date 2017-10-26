import argparse
import cv2
import time
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import glob
from random import sample
from IPython import embed

local_feat='SIFT'
fea_det = cv2.FeatureDetector_create(local_feat)
des_ext = cv2.DescriptorExtractor_create(local_feat)
k=1000

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trainingSet", help="Path to Training Set",\
                        default='/local/MI/temporal_action_localization/data/Thumos14/test_frames')
    parser.add_argument('-s',type=int)
    parser.add_argument('-e',type=int)

    return parser.parse_args()

def get_image_paths(train_path,s_,e_):
    '''
    return :
        training_names: video_name
        image_path: path to image(frame)
    '''
    training_names = os.listdir(train_path)
    training_names.sort()
    training_names=training_names[s_:e_]

    image_paths = []
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path=glob.glob(os.path.join(dir,'img_*.jpg'))
        image_paths+=class_path
    print 'totally process {} files'.format(len(image_paths))
    return training_names,image_paths

if __name__=='__main__':
    args=vars(get_parser())
    '''
    all 3 stages:
    0: Read files to list.
    1: Extracting SIFT features.
    2: Make dictionary
    3: Cluster SIFT features to form BOW feature

    '''
    #stage0 has to be done first
    train_path = args["trainingSet"]
    s_=args['s']
    e_=args['e']

    training_names, image_paths = get_image_paths(train_path,s_,e_) 
    
    #stage 1 :
    stage1=0 # 0 if stage1 has done ,else 1 : tobedone
    if stage1:
        print 'stage1 is on, now extracting SIFT features...'
        des_list=[]
        for i,image_path in enumerate(image_paths):
            im=cv2.imread(image_path)
            kpts = fea_det.detect(im)
            kpts, des = des_ext.compute(im, kpts)
            des_list.append((image_path, np.array(des)))
            if len(des_list)%1000==0:
                print time.ctime(),len(des_list)
        joblib.dump(des_list,'{}_{}_{}.pkl'.format(local_feat,s_,e_).lower())

    else:
        print 'stage1 is off, now loading data from local'
        file_name='{}_{}_{}.pkl'.format(local_feat,s_,e_).lower()
        if not os.path.exists(file_name):
            print 'no local file'
        des_list=joblib.load(file_name)


    #stage 2 :   fisrt sample,  then use KNN to cluster * words.
    stage2=1
    if stage2==1:
        print 'now, doing knn, form a dictionary.'
        sample_des=np.array([des_list[i][1] for i in sample(range(len(des_list)),200)])
        descriptors = sample_des[0]
        for descriptor in sample_des[1:]:
            if type(descriptor)==np.ndarray: # whether there contains a errordata or None?
                descriptors = np.vstack((descriptors, descriptor))

        voc, variance = kmeans(descriptors, k, 1)
        joblib.dump(voc,'knn_vectors.pkl')

    else:
        print 'dictionary is loaded from local path'
        voc = joblib.load('knn_vectors.pkl')

    stage3=1
    if stage3==1:
        print 'now, cluster features to form BOW features'
        im_features = np.zeros((len(image_paths), k), "float32")
        for i in range(len(image_paths)):
            if type(des_list[i][1]==np.ndarray()):
                words, distance = vq(des_list[i][1],voc)
            else:
                words = voc[0] # give errordata a uniform representative.
            for w in words:
                im_features[i][w] += 1
            if i%1000==0:
                print time.ctime(),i
        # saving the total feature..
        np.save('feat_{}_{}.npy',im_features)


    # stage 4 is my personal application.
    stage4=1
    if stage4==1:
        video_nframe_file='../metadata/video_nframe_id_test.txt'
        lines=open(video_nframe_file,'rb').readlines()
        lines.sort()
        lines=lines[s_:e_]
        video_nframe_dic={_.split(' ')[0].split('/')[-1]:int(_.split(' ')[1]) for _ in lines}
        len_frames=sum(video_nframe_dic.values())
        assert len_frames==len(image_paths),'length mismatch!'
        videonames=video_nframe_dic.keys();videonames.sort()
        video_feat_dic=dict()
        i=0
        for videoname in videonames:
            nframe=video_nframe_dic[videoname]
            video_feat_dic[videoname]=im_features[i:i+nframe]
            i=nframe
        print 'assign feat done!'
        joblib.dump(video_feat_dic,'video_bow_feat.pkl')



    # stage 4 for classification:
    # Perform Tf-Idf vectorization
    # nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    # idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
    #
    # # Scaling the words
    # stdSlr = StandardScaler().fit(im_features)
    # im_features = stdSlr.transform(im_features)
    #
    # # Train the Linear SVM
    # clf = LinearSVC()
    # clf.fit(im_features, np.array(image_classes))
    #
    # # Save the SVM
    # joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)