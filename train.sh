set -e
arch=resnet50 #resnet101
GSoP_mode=1 #2
batchsize=384
attdim=128
modeldir=ImageNet1k-$arch-GSoP$GSoP_mode
dataset=/put/your/dataset/path/here

if [ ! -e $modeldir/*.pth.tar ]; then

if [ ! -d  "$modeldir" ]; then
   
mkdir $modeldir

fi
cp train.sh $modeldir

python main.py -a $arch\
               -p 100\
               -j 8\
               -b $batchsize\
               --GSoP_mode $GSoP_mode\
               --attdim $attdim\
               --modeldir $modeldir\
               $dataset
else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

python main.py -a $arch\
               -p 100\
               -j 8\
               -b $batchsize\
               --GSoP_mode $GSoP_mode\
               --attdim $attdim\
               --modeldir $modeldir\
               --resume $checkpointfile\
               $dataset

fi


