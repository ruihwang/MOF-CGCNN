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



with open('./dataset_label/hmof.csv') as f:
    readerv = csv.reader(f)
    pred = [row for row in readerv]


## dataset
train_root = './dataset_cif'
pred_root = './hmof_cif'
mof = MOF_CGCNN(cuda=True,works=40, root_file=train_root,trainset = train[:1], valset=val[:1],testset=test[:1],epoch = 1,lr=0,optim='Adam',batch_size=24,h_fea_len=428,n_conv=5,lr_milestones=[160],weight_decay=5e-8,dropout=0.2)
#pred_experiment
mof.pred_MOF(pred_root,pred,'./best_65bar.pth.tar')

