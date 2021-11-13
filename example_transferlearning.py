from mof import MOF_CGCNN
import csv

##read data
with open('./dataset_label/TL_5000_5bar.csv') as f:
    readerv = csv.reader(f)
    train = [row for row in readerv]
with open('./dataset_label/val_5bar.csv') as f:
    readerv = csv.reader(f)
    val = [row for row in readerv]
with open('./dataset_label/test_5bar.csv') as f:
    readerv = csv.reader(f)
    test = [row for row in readerv]



root = './dataset_cif'
mof = MOF_CGCNN(cuda=True, root_file=root,trainset = train, valset=val,testset=test,epoch = 800,lr=0.001,optim='Adam',batch_size=24,h_fea_len=480,n_conv=5,lr_milestones=[50],weight_decay=1e-7,dropout=0.2)
mof.transfer_learning(modelpath='./best_65bar.pth.tar',fix_layer_lr = 0.0008, flex_layer_lr = 0.001)

