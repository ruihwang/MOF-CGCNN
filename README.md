## Metal Organic Frameworks Crystal Graph Convolutional Neural Networks (MOF-CGCNN)

We developed a novel method, MOF-CGCNN, to efficiently and accurately predict the methane the volumetric uptakes at 65 bar for MOFs. Two major modifications were made to the original CGCNN algorithm.The new pooling method mainly depends on the SBUs to describe the local chemical environment around the metal sites. Considering certain adsorbates dominated by cage window sites, we incorporated certain intrinsic structural features, e.g., PLD, LPD, φ, and AV, to the CGCNN algorithm.

The package is built on an existing code [CGCNN](https://github.com/txie-93/cgcnn)


--------------------------------------------------[Files to be prepared]-----------------------------------------------------------------

You need to prepare two files, a label file(e.g.,  `./dataset_label/pred_experiment.csv`), and a structure file(e.g.,  `./dataset_cif/1.cif`)

------------------------------------------------[label file]-----------------------------------------------------------------------------
a CSV file with multiple columns. The first column recodes a unique `ID` for each MOF. From the second column to the fifth column are 
the structural characteristics of each MOF. 

For example:  `./dataset_label/pred_experiment.csv`
MAF-38,0.932119,0.7097,9.12980,4.48043,76,264
'ID', accessible volume,void fraction,the largest pore diameter,the pore limiting diameter,target property
##Zeo++ software package
The MOF structural features were calculated by the Zeo++ software package. 
For example:
network -ha -vol 0.0 0.0 5000 MAF-38.cif  
network -ha -res MAF-38.cif
---------------------------------------------------------------------------------------------------------------------------------------

-------------------------------------------------[structure file]----------------------------------------------------------------------
`ID.cif`: a CIF file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.
Attention! The CIF file need to meet the following rules:
1, Standard CIF files 
2, Partial occupancy issues is not allowed in CIF files
3, The MOFs crystals in CIF file should not contain solvent molecules (free solvent molecules, and bound solvent molecules)
---------------------------------------------------------------------------------------------------------------------------------------


We provide some examples for training the MOF-CGCNN model under 65 bar, predicting some reports MOFs using the trained model, transferring learning  and screening a hypothetical MOFs database.

`python example_train_65barmodel.py`  ---------->        train the MOF-CGCNN model under 65 bar
`python example_predExperiment.py`    ---------->        predict some reports MOFs using the trained model
`python example_transferlearning.py`  ---------->        transfer learning
`python example_screen.py`            ---------->        screen a hypothetical MOFs database

After training, you will get multiple files.  The most important ones are:

- `model_best.pth.tar`: stores the MOF-CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the MOF-CGCNN model at the last epoch.
- `val_results.csv`: stores the `ID`, target values, and predicted values for each crystal in validation set
- `test_results.csv`: stores the `ID`, target values, and predicted values for each crystal in test 

#Attention
1   If the result predicted by the trained 65bar model is negative, we default it to 0 cm3/cm3.

## License
MOF-CGCNN is released under the MIT License.

## Citation
Please consider citing our paper if you use this code in your work
