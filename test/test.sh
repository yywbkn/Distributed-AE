set -ex

python test.py --dataroot ./COVID_data_test/ --name covid_KL_train --model asynKL --phase test   --eval  --results_dir results/KL_test01

