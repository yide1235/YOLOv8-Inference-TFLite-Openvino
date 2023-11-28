import numpy as np
import math

#make first global id 0, or else need to change add to global dataset
soft_similarity_threshold=8
hard_similarity_threshold=3
global_vec_threshold=5
global_face_threshold=5
maturity_threshold=30
zero_tol=8
conf_cutoff_lower=0.7
conf_cutoff_upper=0.9
secondary_euclidean_threshold=8
global_search_percentage=2.4

global next_id
next_id=0
global next_global_id
next_global_id=0
global global_people_list
global_people_list=[]
global global_people_data
global_people_data=np.transpose(np.array([[] for _ in range(34)]))
temp_people_list=[]

def binary_search(arr,value):
    left=0
    right=len(arr)-1
    
    if right<0:
        return 0
    
    while left<=right:
        mid=(left+right)//2
        if arr[mid]==value:
            return mid
        if arr[mid]<value:
            left=mid+1
        else:
            right=mid-1
            
    return mid

def filter_global_data(arr,lookup,percentage):
    index_matched=[]
    started=False
    for i in range(len(lookup)):
        if lookup[i]==0:
            continue
        lower=lookup[i]*(1-percentage)
        upper=lookup[i]*(1+percentage)
        lower_ind=binary_search(arr[:,i*2+1],lower)
        upper_ind=binary_search(arr[:,i*2+1],upper)
        if len(index_matched)==0 and started==False:
            index_matched=arr[lower_ind:upper_ind+1,i*2]
            started=True
        else:
            index_matched=list(set(index_matched).intersection(arr[lower_ind:upper_ind+1,i*2]))
    return index_matched


class Frame_data:
    def __init__(self, frame_id, data, confidence, bounding_box, face_box):
        self.id = frame_id
        self.data = data
        self.conf = confidence
        self.bounding_box=bounding_box
        self.face_box=face_box
        

class Global_person:
    def __init__(self,person):
        self.global_id=person.global_id
        self.frame_ids=person.frame_ids
        self.face_frame=person.face_frame
        self.bounding_box=person.bounding_box
        self.face_box=person.face_box
        self.face_score=person.face_score


class Person:
    def __init__(self,temp_id,vec,conf,frame_id,maturity_threshold,conf_cutoff_lower,conf_cutoff_upper,bounding_box,face_box):
        self.temp_id=temp_id
        self.global_id=None
        self.frame_ids=[frame_id]
        self.face_frame=[]
        self.bounding_box=[bounding_box]
        self.face_box=[face_box]
        self.face_score=0
        self.mature=False
        self.maturity_cutoff=maturity_threshold
        self.conf_cutoff_lower=conf_cutoff_lower
        self.conf_cutoff_upper=conf_cutoff_upper
        self.data=[[0] for _ in range(len(vec))]
        self.mean=[0]*len(vec)
        self.all_data=[[0] for _ in range(len(vec))]
        self.all_conf=[[0] for _ in range(len(vec))]
        self.sd=[0]*len(vec)
        
        for i in range(len(conf)):
            if conf[i]>=conf_cutoff_upper and vec[i]!=0:
                self.all_data[i]=[vec[i]]
                self.data[i]=[vec[i]]
                self.all_conf[i]=[conf[i]]
                self.mean[i]=vec[i]
            elif conf[i]>=conf_cutoff_lower and vec[i]!=0:
                self.all_data[i]=[vec[i]]
                self.all_conf[i]=[conf[i]]
                self.mean[i]=vec[i]
                
        
    def add(self,vec,conf,frame_id,bounding_box,face_box,vec_threshold,face_threshold,global_search_percentage):
        if self.mature==False:
            for i in range(len(vec)):
                if vec[i] != 0:
                    if conf[i]>=self.conf_cutoff_upper:
                        if self.all_data[i][0]!=0:
                            self.all_data[i].append(vec[i])
                            self.all_conf[i].append(conf[i])
                        else:
                            self.all_data[i]=[vec[i]]
                            self.all_conf[i]=[conf[i]]
                        if self.data[i][0]!=0:
                            self.data[i].append(vec[i])
                        else:
                            self.data[i]=[vec[i]]
                        
                    elif conf[i]>=conf_cutoff_lower:
                        if self.all_data[i][0]!=0:
                            self.all_data[i].append(vec[i])
                            self.all_conf[i].append(conf[i])
                        else:
                            self.all_data[i]=[vec[i]]
                            self.all_conf[i]=[conf[i]]
                            
            self.bounding_box.append(bounding_box)
            self.face_box.append(bounding_box)
            self.frame_ids.append(frame_id)
            self.update_mean()
            self.check_maturity(vec_threshold,face_threshold,global_search_percentage)
                    
    def update_mean(self):
        for i in range(len(self.data)):
            self.mean[i]=np.mean(self.data[i])
            self.sd[i]=np.std(self.data[i])
                    
    def check_maturity(self,vec_threshold,face_threshold,global_search_percentage):
        if self.mature!=True:
            mature=True
            for i in range(len(self.all_conf)):
                if len(self.all_data[i]) < 30*math.exp(2.5*(1-np.mean(self.all_conf[i]))):
                    mature=False
            if mature==True:
                self.mature=True
                self.update_mean()
                self.get_global_id(vec_threshold,face_threshold,global_search_percentage)
            return mature
        else:
            return True
            
    def check_similarity(self, candidate_vec, conf, threshold):
        for i in range(len(self.mean)):
            if candidate_vec[i]==0 or self.mean[i]==0:
                continue
            if abs(self.mean[i]-candidate_vec[i])>threshold and conf[i]>self.conf_cutoff_lower:
                return False
        return True
    
    def get_global_id(self,vec_threshold,face_threshold,global_search_percentage):
        good=[]
        score=[]
        global global_people_data
        global global_people_list
        
        if len(global_people_list)==0:
            self.global_id=get_next_global_id()
            self.add_to_global_dataset()
            return
            
        
        found_ids=filter_global_data(global_people_data,self.mean,global_search_percentage)
        
        found=[]
        for i in found_ids:
            found.append(global_people_list[i])
        
        for person in found:
            face_similarity=run_face_check(person.face_frame[0],self.face_frame[0])
            if face_similarity < face_threshold:
                good.append(person)
                score.append(face_similarity)
                
        if len(good)==0:
            self.global_id=get_next_global_id()
            self.add_to_global_dataset()
        elif len(good)==1:
            self.global_id=good[0].global_id
        else:
            self.global_id=good[score.index(max(score))].global_id
            
    def add_to_global_dataset(self):
        global global_people_data
        global global_people_list            
        
        if len(global_people_list)>self.global_id:
            global_people_list[self.global_id]=Global_person(self)
        elif len(global_people_list)==self.global_id:
            global_people_list.append(Global_person(self))
        
        for i in range(len(self.mean)):
            ind=binary_search(global_people_data[:,i*2+1],self.mean[i])
            global_people_data[:,i*2]=np.insert(global_people_data[:,i*2],ind,self.global_id)
            global_people_data[:,i*2+1]=np.insert( global_people_data[:,i*2+1],ind,self.mean[i])
    


def get_next_temp_id():
    global next_id
    id_val=next_id
    next_id=next_id+1
    return id_val

def get_next_global_id():
    global next_global_id
    id_val=next_global_id
    next_global_id=next_global_id+1
    return id_val

def run_face_check(face1_frame,face2_frame):
        return next_id % 2 == 0
    
def secondary_check(vec,conf,person,threshold):
    score=get_euclidean_score(vec,conf,person)
    if score<threshold:
        return True
    else:
        return False

def get_euclidean_score(vec,conf,person):
    vec1=person.mean
    count1=0
    dist1=0
    for i in range(len(vec)):
        if vec[i] !=0 and conf[i] >= conf_cutoff_lower:
            if vec1[i] !=0:
                count1=count1+1
                dist1=dist1+(vec[i]-vec1[i])**2
                
    dist1=dist1/count1
    dist1=dist1/(1.05**count1)
    
    return dist1

def euclidean_check(vec, conf, found):
    scores=[]
    for i in range(len(found)):
        scores.append(get_euclidean_score(vec,conf,found[i]))
        
    best_ind=scores.index(max(scores))
    return found[best_ind]
    
def sort_frame_data(frame,people_list,soft_similarity_threshold,hard_similarity_threshold,maturity_threshold,vec_threshold,face_threshold,secondary_euclidean_threshold,global_search_percentage):
    for i in range(len(frame.data)):
        found=[]
        person_vec=frame.data[i]
        person_conf=frame.conf[i]
        count=0
        for j in range(len(person_vec)):
            if person_vec[j]>0 and frame.conf[i][j]>=conf_cutoff_lower:
                count=count+1
        if count<zero_tol:
            continue
        for person in people_list:
            hard_similarity=person.check_similarity(person_vec,person_conf,hard_similarity_threshold)
            if hard_similarity==True:
                found.append(person)
            else:
                soft_similarity=person.check_similarity(person_vec,person_conf,soft_similarity_threshold)
                print("soft,",soft_similarity)
                if soft_similarity==True:
                    if secondary_check(person_vec,frame.conf[i],person,secondary_euclidean_threshold)==True:
                        print("secondary,True")
                        found.append(person)
        if len(found)==0:
            print("Found 0")
            next_id=get_next_temp_id()
            people_list.append(Person(next_id,person_vec,frame.conf[i],frame.id,maturity_threshold,conf_cutoff_lower,conf_cutoff_upper,frame.bounding_box[i],frame.face_box[i])) 
        elif len(found)==1:
            print("Found 1")
            found[0].add(person_vec,frame.conf[i],frame.id,frame.bounding_box[i],frame.face_box[i],vec_threshold,face_threshold,global_search_percentage)
        else:
            print("Found 2+")
            best=euclidean_check(person_vec,frame.conf[i],found)
            best.add(person_vec,frame.conf[i],frame.id,frame.bounding_box[i],frame.face_box[i],vec_threshold,face_threshold,global_search_percentage)
    return people_list
            