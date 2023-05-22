echo "======= TRANSFER MAIN.PY ======="
scp /Users/quentinlao/Documents/GitHub/Data-augmentation-time-series/SBTimeSeries/test/main.py hpc:/home/users/qlao/SBTimeSeries/test/
for i in {1..5}
do
   echo ""
done
echo "======= EXCEUTION MAIN.PY ======="
ssh hpc 'conda activate tf-gpu; cd SBTimeSeries/test; python main.py'
for i in {1..5}
do
   echo ""
done
echo "========= GET BACK DATA ========="
rm -r SBTimeSeries/test/plot
scp -r hpc:/home/users/qlao/SBTimeSeries/test/plot /Users/quentinlao/Documents/GitHub/Data-augmentation-time-series/SBTimeSeries/test
