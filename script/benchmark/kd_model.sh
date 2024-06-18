python3 lat_acc_check_cifar100.py --path save/student_model/S:vgg8_T:vgg11_cifar100_kd_r:0.1_a:0.9_b:0.0_2/vgg8_last.pth --save kd_11 --iter 60
python3 lat_acc_check_cifar100.py --path save/student_model/S:vgg8_T:vgg13_cifar100_kd_r:0.1_a:0.9_b:0.0_2/vgg8_last.pth --save kd_13 --iter 60
python3 lat_acc_check_cifar100.py --path save/student_model/S:vgg8_T:vgg16_cifar100_kd_r:0.1_a:0.9_b:0.0_2/vgg8_last.pth --save kd_16 --iter 60
python3 lat_acc_check_cifar100.py --path save/student_model/S:vgg8_T:vgg19_cifar100_kd_r:0.1_a:0.9_b:0.0_2/vgg8_last.pth --save kd_19 --iter 60