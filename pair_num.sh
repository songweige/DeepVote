# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 1

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 2

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 3

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 4

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 5

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 6

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 7

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 9

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 8

# python infer.py -g 1 -o ../results/ -m base -r True -if 0 -ie 399 -n plain_res -i /home/songweig/LockheedMartin/data/MVS/dem_6_18 -np 10

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair1.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair2.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair3.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair4.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair5.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair6.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair7.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair8.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair9.npy -g ../../data/DSM/dem_6_18.npy

python plot_diff.py -i ../results/plain_res/reconstruction/dem_6_18_fold0_399_pair10.npy -g ../../data/DSM/dem_6_18.npy