#!/usr/bin/env python


from operator import itemgetter
import cv2
import numpy as np
import os
import shutil
import math
import joblib
files = open("output.txt","wb")
image_paths = []
path = "/home/wizkraft/Desktop/Faces/barfi/"
training_names = os.listdir(path)
arr=[]
print 'abc'
total_img=876
training_paths = []
testing_paths = []
names_path = []
for p in training_names:
    training_paths1 = os.listdir("/home/wizkraft/Desktop/Faces/barfi/"+p)
    length=3*len(training_paths1)/4
    k=1
    for j in training_paths1:
        training_paths.append("/home/wizkraft/Desktop/Faces/barfi/"+p+"/"+j)
        names_path.append(p)
        k=k+1

sift = cv2.SIFT()
#print names_path
#print(len(training_paths))

dictionarySize =100
rc=dictionarySize
cc=total_img

tf_idf = [0] * rc
for i in range(rc):
    tf_idf[i] = [0] *cc
#print tf_idf[269][99]
ifindex=[None]*dictionarySize
#fprint ifindex
for h  in range(dictionarySize):
  #print ''
  ifindex[h]=[]
BOW = cv2.BOWKMeansTrainer(dictionarySize)

for p in training_paths:
    image = cv2.imread(p)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
    kp, dsc= sift.detectAndCompute(gray, None)
    print dsc
    if dsc is None:
      print
    else:
     BOW.add(dsc)
dictionary = BOW.cluster()


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   
flann = cv2.FlannBasedMatcher(index_params,search_params)
sift2 = cv2.DescriptorExtractor_create("SIFT")
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)
print np.shape(dictionary)



def feature_extract(pth):
    im = cv2.imread(pth, 1)
    gray = cv2.cvtColor(im, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #f.write("\nMaybe someday, he will promote me to a real file.\n")
    #f.write(bowDiction.compute(gray, sift.detect(gray)))
    #print pth
    #print bowDiction.compute(gray, sift.detect(gray))

    arr=bowDiction.compute(gray, sift.detect(gray))
    #print arr
    #print 'fuck'
    print bowDiction.compute(gray, sift.detect(gray))
    for i in range(dictionarySize):
     ifindex[i].append ({'image':pth,'score':arr[0][i]})
    print ifindex[i]

    return bowDiction.compute(gray, sift.detect(gray))

train_desc = []
train_labels = []
i = 1
t=0
for p in training_paths:
    i=t/54+1
    train_desc.extend(feature_extract(p))
    train_labels.append(i)
    #print i
    #print p
    
    t=t+1
##lets make tf-idf a twodim mat with rownames clusters and coulmn names images
##totalnumberof trainining images=270

N=total_img
Ni=0
for i in range(rc):
 Ni=0

 for p in ifindex[i]:
  if p['score']>0.0:
   Ni=Ni+1
 u=0
 print Ni
  
 for j in training_paths:
   Nd=1
   Ndi=0
   for p in ifindex[i]:
    if p['image']==j:
     Ndi=p['score']
   #print i,u
   if Ni>0:
   
     tf_idf[i][u]=(Ndi/Nd)*math.log(N/Ni)
   else:
     tf_idf[i][u]=0
   print tf_idf[i][u]
   u=u+1
count=0
svm = cv2.SVM()
svm.train(np.array(train_desc), np.array(train_labels))
i=0
j=0

confusion = np.zeros((5,5))
def classify(pth):
    feature = feature_extract(pth)
    p = svm.predict(feature)
    print pth
    print  p

def classify_voc(pth):
  final_score=[0]*total_img
  image = cv2.imread(pth, 1)
  gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  arr=bowDiction.compute(gray, sift.detect(gray))
  u=0
  for  p in training_paths:
   score=0
   for i in range(dictionarySize):
    score=score+ tf_idf[i][u]*arr[0][i]
   final_score[u]={'score':score,'path':p}

   u=u+1
  final_score = sorted(final_score, key=itemgetter('score'), reverse=True)
  
  for i in range(30):

   z,x,y=final_score[i]['path'].split('/');
   files.write(y)
   files.write(" ")
   files.write(x)
   files.write("\n")
print bowDiction

    
classify_voc("/home/wizkraft/Pictures/1.jpg")
