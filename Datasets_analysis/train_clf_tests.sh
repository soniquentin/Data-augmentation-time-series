for i in {1..20}
do
	echo "ITERATION $i"
	echo ""
	python train_clf.py --mode single --train On
	python train_clf.py --mode multi --train On
	python train_clf.py --mode conditionned_single --train On
done
