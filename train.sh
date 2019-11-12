set -e
arch=resnet50 #resnet101
GSoP_mode=1 #2
att_manner=channel #{channel, position, fusion_avg, fusion_max, fusion_concat}
batchsize=384
attdim=128
modeldir=checkpoints/ImageNet1k-$arch-GSoP$GSoP_mode-${att_manner}
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
               --att_manner ${att_manner}\
               --attdim $attdim\
               --modeldir $modeldir\
               $dataset
else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

python main.py -a $arch\
               -p 100\
               -j 8\
               -b $batchsize\
               --GSoP_mode checkpoints/$GSoP_mode\
               --att_manner ${att_manner}\
               --attdim $attdim\
               --modeldir $modeldir\
               --resume $checkpointfile\
               $dataset

fi


