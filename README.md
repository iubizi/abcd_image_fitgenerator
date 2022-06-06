# abcd_image_fitgenerator

abcd_image_fitgenerator

The original file size (after nii compression) is `150GB`, the file is too large to fit in memory, so use the fit generator to read it incrementally.

programs and notes have been desensitized.

```
#!/bin/bash

cd ~
cd abcd
source activate base
source activate py38tf23



clear



rm -f ~/abcd_data_pre.log

rm -rf ~/abcd/histogram
rm -rf ~/abcd/visualization



rm -f ~/abcd_train.log
rm -f ~/abcd_test.log

rm -rf ~/abcd/train
rm -rf ~/abcd/test
rm -rf ~/abcd/model

rm -rf ~/abcd/__pycache__





python abcd_main.py -t pre -s T1w | tee -a -i ~/abcd_data_pre.log

python abcd_main.py -t pre -s T2w | tee -a -i ~/abcd_data_pre.log





# single line comment
# cmd转储（非slurm/pbs下的cmd转储）

python abcd_main.py -t train -n stb_c1 -s T1w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n stb_p1 -s T1w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n stb1 -s T1w | tee -a -i ~/abcd_train.log

python abcd_main.py -t train -n si_c1 -s T1w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n si_p1 -s T1w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n si1 -s T1w | tee -a -i ~/abcd_train.log

python abcd_main.py -t train -n sa_c1 -s T1w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n sa_p1 -s T1w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n sa1 -s T1w | tee -a -i ~/abcd_train.log



python abcd_main.py -t test -n stb_c1 -s T1w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n stb_p1 -s T1w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n stb1 -s T1w | tee -a -i ~/abcd_test.log

python abcd_main.py -t test -n si_c1 -s T1w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n si_p1 -s T1w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n si1 -s T1w | tee -a -i ~/abcd_test.log

python abcd_main.py -t test -n sa_c1 -s T1w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n sa_p1 -s T1w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n sa1 -s T1w | tee -a -i ~/abcd_test.log





python abcd_main.py -t train -n stb_c1 -s T2w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n stb_p1 -s T2w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n stb1 -s T2w | tee -a -i ~/abcd_train.log

python abcd_main.py -t train -n si_c1 -s T2w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n si_p1 -s T2w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n si1 -s T2w | tee -a -i ~/abcd_train.log

python abcd_main.py -t train -n sa_c1 -s T2w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n sa_p1 -s T2w | tee -a -i ~/abcd_train.log
python abcd_main.py -t train -n sa1 -s T2w | tee -a -i ~/abcd_train.log



python abcd_main.py -t test -n stb_c1 -s T2w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n stb_p1 -s T2w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n stb1 -s T2w | tee -a -i ~/abcd_test.log

python abcd_main.py -t test -n si_c1 -s T2w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n si_p1 -s T2w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n si1 -s T2w | tee -a -i ~/abcd_test.log

python abcd_main.py -t test -n sa_c1 -s T2w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n sa_p1 -s T2w | tee -a -i ~/abcd_test.log
python abcd_main.py -t test -n sa1 -s T2w | tee -a -i ~/abcd_test.log
```
