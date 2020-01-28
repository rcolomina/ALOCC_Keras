import cv2
import numpy as np
        

def find_list_files(pattern_filename, path):
    '''
    Find list of files following a pattern filename within a path
    This returns a list of results, void list [] if items not found
    '''
    import os,fnmatch
    result = []
    for root, dirs, files in os.walk(path): # os walk is a generator
        for name in files:
            if fnmatch.fnmatch(name, pattern_filename):
                result.append(os.path.join(root, name))
    return result

# Goal: To build numpy array (num_samples,win_edge1,win_edge2)

class LandsatDataLoader:
    def __init__(self, input_folder):
        self.X_train = None
        self.input_files  = [f for f in find_list_files("LS*png",input_folder)]
        print("Number of input files loaded:",len(self.input_files))
        
    def _load_image(self,ifile):
        image = cv2.imread(ifile)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    def load_data(self, randomization=False):
        #num_samples = len(input_files)
        #print(len(self.input_files))
        list_samples = []
        
        for ifile in self.input_files: 
            list_samples.append(self._load_image(ifile)) 


        if randomization:
            seed=42
            print("Applying Randomization on list of input samples seed {}".format(seed))
            import random
            random.seed(seed)
            random.shuffle(list_samples)
            
        X_train = np.array(list_samples)
        self.X_train = X_train
        return X_train

    def load_data_and_filenames(self):
        filenames = []
        list_samples = []
        for ifile in self.input_files:             
            list_samples.append(self._load_image(ifile))
            filenames.append(ifile)
            
        X_train = np.array(list_samples)
        self.X_train = X_train
        return X_train, filenames
        
    
    def normal_sample(self):
        pass

    def abnormal_sample(self):
        pass
    
if __name__ == '__main__':
    input_folder = '/QCOLT/QCOLT_DEV_OPS//TDS_NOVELTY_DETECTION/EXP_02//nominal_chips/'
    dataLoader = LandsatDataLoader(input_folder)
    data = dataLoader.load_data()
    print(data.shape)
