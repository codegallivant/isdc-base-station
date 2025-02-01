rm -rf data/input
mkdir data/input
scp -r manas@192.168.1.21:/home/manas/realsense-pub-interface/realsense_processor/frames data/input
mv data/input/frames data/input
