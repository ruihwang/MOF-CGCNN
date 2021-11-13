from mof import MOF_CGCNN
import csv

##read data
with open('./dataset_label/train_65bar.csv') as f:
    readerv = csv.reader(f)
    train = [row for row in readerv]
with open('./dataset_label/val_65bar.csv') as f:
    readerv = csv.reader(f)
    val = [row for row in readerv]
with open('./dataset_label/test_65bar.csv') as f:
    readerv = csv.reader(f)
    test = [row for row in readerv]
#file path
root = './dataset_cif'
#create a class
import time
time_start=time.time()
mof = MOF_CGCNN(cuda=True, root_file=root,trainset = train[:1000], valset=val,testset=test,epoch = 100,lr=0.002,optim='Adam',batch_size=24,h_fea_len=480,n_conv=5,lr_milestones=[200],weight_decay=1e-7,dropout=0.2)
# train the model
mof.train_MOF()
time_end=time.time()
print('totally cost',time_end-time_start)
