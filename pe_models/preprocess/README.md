1. download LIDC from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
2. install pylidc and make sure the .pylidcrc file exists in your HOME https://pylidc.github.io/install.html
3. review the constants.py to make sure things are being saved to the right place
4. run the main method of preprocess/lidc.py to set up the csv and hdf5
5. make sure dataset type is set to lidc-window in the experiment config file
