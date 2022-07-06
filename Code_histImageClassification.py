import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils import compute_class_weight
from torch.utils.tensorboard import SummaryWriter

'''
'''
CONFIGURATION = {
    "datasetPath" : "E:\\AI\\APPS\\BreaKHis_v1",
    "modelPath":"Model"
}

'''
'''
class DataHandler:
    def __init__(self,CONFIGURATION):
        self.CONFIGURATION = CONFIGURATION
        self.trainDF, self.valDF, self.testDF = self.retrieveDatasetInfo()
    
    '''
    '''
    def retrieveDatasetInfo(self):
        trainPath = os.path.join(self.CONFIGURATION["datasetPath"],"trainDF.csv")
        valPath = os.path.join(self.CONFIGURATION["datasetPath"],"valDF.csv")
        testPath = os.path.join(self.CONFIGURATION["datasetPath"],"testDF.csv")
        
        if os.path.isfile(trainPath) and os.path.isfile(valPath) and os.path.isfile(testPath):
            trainDF = pd.read_csv(trainPath,index_col=0)
            valDF = pd.read_csv(valPath,index_col=0)
            testDF = pd.read_csv(testPath,index_col=0)

        else:
            trainPat, testPat, valPat = self.splitPatients()
            trainDF = self.genDataDF(trainPat,"Train")
            valDF = self.genDataDF(valPat,"Validation")
            testDF = self.genDataDF(testPat,"Test")
            
            trainDF.to_csv(trainPath)
            valDF.to_csv(valPath)
            testDF.to_csv(testPath)

        classes = ['benign', 'malignant']
        trainDF['label'] = trainDF.category.apply(lambda x: classes.index(x))
        valDF['label'] = valDF.category.apply(lambda x: classes.index(x))
        testDF['label'] = testDF.category.apply(lambda x: classes.index(x))

        trainDF = trainDF.sample(frac=1).reset_index(drop=True)
        valDF = valDF.sample(frac=1).reset_index(drop=True)
        testDF = testDF.sample(frac=1).reset_index(drop=True)

        return trainDF, valDF, testDF

    '''
        Initially used in this repo: https://github.com/prakashchhipa/Magnification-Prior-Self-Supervised-Method/blob/main/src/data/prepare_data.py
    '''
    def splitPatients(self):
        
        root = os.path.join(self.CONFIGURATION["datasetPath"],"BreaKHis_v1/histology_slides/breast")
        benign_list = ['/benign/SOB/adenosis/','/benign/SOB/fibroadenoma/', '/benign/SOB/phyllodes_tumor/','/benign/SOB/tubular_adenoma/']
        malignant_list = ['/malignant/SOB/lobular_carcinoma/', '/malignant/SOB/papillary_carcinoma/', '/malignant/SOB/ductal_carcinoma/', '/malignant/SOB/mucinous_carcinoma/']

        count = 0
        patient_list = list()

        for benign_type_dir in benign_list:
            p_dir_path = root + benign_type_dir
            for p_id in os.listdir(p_dir_path):
                patient_list.append(p_dir_path + p_id)
                count +=1

        for malignant_type_dir in malignant_list:
            p_dir_path = root + malignant_type_dir
            for p_id in os.listdir(p_dir_path):
                patient_list.append(p_dir_path + p_id)
                count +=1
        
        category = list()
        for patient_path in patient_list:
            category.append(patient_path.split('/')[-1].split('_')[1])
            
        trainPat, testPat, trainCategory, testCategory = train_test_split(patient_list,category,stratify=category,test_size=0.20,random_state=0)
        trainPat, valPat, trainCategory, valCategory = train_test_split(trainPat,trainCategory,stratify=trainCategory,test_size=0.25,random_state=0)
        return trainPat, testPat, valPat

    '''
    '''        
    def genDataDF(self,patientList,mode):
        content = list()
        
        for patPath in patientList:
            patientEx = Path(patPath).parts[-1]
            category = Path(patPath).parts[-4]
        
            for imgPath in list(Path(patPath).glob("**/*.png")):
                magn = imgPath.parts[-2]
                content.append({
                    "path":imgPath,
                    "patientEx":patientEx,
                    "magn":magn,
                    "category":category,
                    "portion":mode
                })
            
        dataDF = pd.DataFrame.from_dict(content)
        return dataDF

'''
'''
class BCDataset(torch.utils.data.Dataset):
    def __init__(self, datasetDF):
        self.datasetDF = datasetDF
        self.transform =  transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize([224,224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    '''
    '''    
    def __len__(self):
        return len(self.datasetDF)

    '''
    '''    
    def __getitem__(self, index):
        node = self.datasetDF.iloc[index]
        img = cv2.imread(str(node.path))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        if "Augmentation" in self.datasetDF.columns:
            if node.Augmentation != "None":
                img = self.augmentor(img,node.Augmentation)
                  
        img = self.transform(img)
        label = node.label
        
        return img, np.array(int(label))
    
    '''
    '''
    def augmentor(self,image,method):
        if method == "HF":
            return cv2.flip(image, 1)
        elif method == "RR":
            angle = 45
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

'''
'''
class DLModel(nn.Module):
    def __init__(self):
        super(DLModel, self).__init__()
        resnet = torchvision.models.mobilenet_v2(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feHidden = 2048
        
        self.feFinal = nn.Sequential(
            nn.Linear(1280 * 7 * 7, feHidden),
            nn.BatchNorm1d(feHidden, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(feHidden, feHidden//4),
            nn.BatchNorm1d(feHidden//4, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(feHidden//4,1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        batchSize = features.shape[0]
        featuresDim = features.shape[1]
        features = features.view(batchSize,-1)
        out = self.feFinal(features)
        return out


'''
'''
'''
'''
def epochTrain(model,device,trainDataGen,optimizer,criterion,report=100):
    model.train()

    epochLoss = 0
    dataCounter = 0
    
    yTrueList = list()
    yPredList = list()

    for batchIndex, (X,y) in enumerate(trainDataGen):
        dataCounter += X.size(0)

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        output = model(X)
        
        output = output.view(-1, )
        
        loss = criterion(output, y.float())
        
        
        loss.backward()
        optimizer.step()

        epochLoss += loss.item() * X.size(0)
        
        pred = (output >= 0.5).long()

        yTrueList.extend(y)
        yPredList.extend(pred)

        if (batchIndex + 1) % report == 0:
            print("[INFO]       {}/{} samples have passed...".format(dataCounter,len(trainDataGen.dataset)))
    
    epochLoss = epochLoss/len(trainDataGen)
    
    yTrue = torch.stack(yTrueList, dim=0)
    yPred = torch.stack(yPredList, dim=0)
    score = f1_score(yTrue.cpu().data.squeeze().numpy(), yPred.cpu().data.squeeze().numpy())

    return epochLoss, score

'''
'''
def epochVal(model,device,criterion,valDataGen):
    model.eval()

    epochLoss = 0
    yTrueList = list()
    yPredList = list()

    with torch.no_grad():
        for X, y in valDataGen:
            X = X.to(device)
            y = y.to(device)

            output = model(X)
            output = output.view(-1, )
            
            loss = criterion(output, y.float())
            epochLoss += loss.item() * X.size(0)
            
            pred = (output >= 0.5).long()    

            yTrueList.extend(y)
            yPredList.extend(pred)
            
    yTrue = torch.stack(yTrueList, dim=0)
    yPred = torch.stack(yPredList, dim=0)

    score = f1_score(yTrue.cpu().data.squeeze().numpy(), yPred.cpu().data.squeeze().numpy())

    epochLoss = epochLoss/len(valDataGen)
    return epochLoss, score

'''
'''
def passAugmentation(df):
    methods = ["HF","RR"]
    
    augDFList = list()
    for m in methods:
        augDF = df.copy()
        augDF["Augmentation"] = m
        augDFList.append(augDF)
        
    augDFList.append(df)
    
    return pd.concat(augDFList,ignore_index=True)

'''
'''
def deepModelDevelopment(CONFIGURATION,de,modelPath,modelState):
    os.makedirs(modelPath,exist_ok=True)
    writerPath = os.path.join(modelPath,"runs")
    trainWriter = SummaryWriter(os.path.join(writerPath,"train"))
    valWriter = SummaryWriter(os.path.join(writerPath,"val"))

    trainDF = de.trainDF
    valDF = de.valDF
    trainDF["Augmentation"] = "None"
    trainDF = passAugmentation(trainDF)
    
    trainDataset = BCDataset(trainDF)
    valDataset = BCDataset(valDF)

    trainDataGen = torch.utils.data.DataLoader(trainDataset, batch_size=8, shuffle=True)
    valDataGen = torch.utils.data.DataLoader(valDataset, batch_size=8, shuffle=False)

    model = DLModel()
    
    device = torch.device("cuda")
    model = model.to(device)
    
    weights = compute_class_weight(y=trainDF.label.values, class_weight="balanced", classes=[0,1])    
    class_weights = torch.FloatTensor(weights)
    class_weights = class_weights.cuda()

    print("[INFO]       Class Weights:")
    print(class_weights)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    sinceLastBest = 1
    minLoss = 99999999
    
    for epoch in range(1,100+1):
        trainLoss, trainScore = epochTrain(model,device,trainDataGen,optimizer,criterion)
        valLoss, valScore = epochVal(model,device,criterion,valDataGen)

        print("[INFO]       Epoch {} ---> Training Loss = {:.4} - Training Accuracy {:.4} -Validation Loss = {:.4} - Validation Accuracy = {:.4}".format(epoch,trainLoss,trainScore,valLoss,valScore))
        
        trainWriter.add_scalar("Loss",trainLoss,epoch)
        trainWriter.add_scalar("Accuracy",trainScore,epoch)
        valWriter.add_scalar("Loss",valLoss,epoch)
        valWriter.add_scalar("Accuracy",valScore,epoch)
        
        sinceLastBest += 1
        
        if valLoss < minLoss:
            print("[INFO]       Model saved!")
            torch.save(model.state_dict(),modelState)
            sinceLastBest = 1
            minLoss = valLoss

        if sinceLastBest > 5:
            break
    
    print("[INFO]   Model trained!")
    trainWriter.flush()
    valWriter.flush()

'''
'''
def deployModel(CONFIGURATION,de,modelPath,modelState,mode):
    model = DLModel()
        
    device = torch.device("cuda")
    model = model.to(device)
    model.load_state_dict(torch.load(modelState))
    model.eval()          

    if mode == "Test":
        
        testDF = de.testDF
 
        print("[INFO]   Perfoming predictions on Test Dataset:")
        testDataset = BCDataset(testDF)
        testDataGen = torch.utils.data.DataLoader(testDataset, batch_size=16, shuffle=False)

        yTrueList = list()
        yPredList = list()

        with torch.no_grad():
            for X, y in testDataGen:
                X = X.to(device)
                y = y.to(device).view(-1, )
                
                output = model(X)
                
                pred = (output >= 0.5).long()

                yTrueList.extend(y)
                yPredList.extend(pred)

        yTrue = torch.stack(yTrueList, dim=0)
        yPred = torch.stack(yPredList, dim=0)

        testDF["predictions"] = yPred.cpu().data.squeeze()

        print(classification_report(y_pred=yPred.cpu().data.squeeze().numpy(),y_true=yTrue.cpu().data.squeeze().numpy()))
        print("[INFO]       Confusion Matrix:")
        print(confusion_matrix(y_pred=yPred.cpu().data.squeeze().numpy(),y_true=yTrue.cpu().data.squeeze().numpy()))
        
        totalPreds = testDF.groupby('patientEx').apply(lambda x: (x.label==x.predictions).sum()).reset_index(name='count')
        counts = testDF.groupby(["patientEx"])["patientEx"].count().reset_index(name="count")
        
        totalPreds = totalPreds.set_index("patientEx")
        totalPreds.columns=["correct"]
        counts = counts.set_index("patientEx")
        predDF = pd.merge(totalPreds, counts, left_index=True, right_index=True)
        
        print("[INFO]       Patient - Level Accuracy:")
        predDF["patientAc"] = predDF["correct"] / predDF["count"]
        
        recognitionRate = predDF["patientAc"].sum()/len(predDF)
        print("[INFO]       Recognition Rate: {}".format(recognitionRate))

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=predDF.index.values,
            y=predDF["correct"],
            name='Correct Predictions',
            marker_color='lightseagreen'
        ))
        fig.add_trace(go.Bar(
            x=predDF.index.values,
            y=predDF["count"],
            name='Total Images',
            marker_color='blue'
        ))

        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        fig.show()

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=predDF.index.values,
            y=predDF["patientAc"],
            name='Patient Accuracy',
            marker_color='lightseagreen'
        ))

        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        fig.show()


'''
'''
def getDeepModelInfo(CONFIGURATION,de):
    modelPath = os.path.join(CONFIGURATION["modelPath"],"DeepModel")
    modelState = os.path.join(modelPath,"DeepModel.pt")

    if not os.path.isfile(modelState):
        print("[INFO]   There is no model - Need development:")
        deepModelDevelopment(CONFIGURATION,de,modelPath,modelState)
    else:
        deployModel(CONFIGURATION,de,modelPath,modelState,"Test")



'''
'''
def main():
    de = DataHandler(CONFIGURATION)
    getDeepModelInfo(CONFIGURATION,de)

if __name__ == "__main__":
    main()