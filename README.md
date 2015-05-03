# Treebank

Several steps of general usage:

1) Extract features for dependency pairs (dep_features.py)


python dep_features.py infilename outdirname param

where 
- infilename is a csv formatted file with dependency markup (with "check" column for training your own model or without it for test data)
- outdirname - output directory for file with extracted features
- param: 
	"GS_head" - only manually checked markup ("check" column)
	"head" - for parser markup
	"all" - extract every sort of markup (useful for training your own model)



2) Train model / evaluate model  (model.py)

python model.py -c classifier -i inputfile -e n_estimators -t 'train_eval' 

3) Process corpus with loaded model (model.py)

python model.py -i testinputfile -m modelfilename -o output -t 'test'


4) Process corpus with trained model (model.py)

python model.py -i traininputfile -c classifiername -o output -t 'train_test' -f testinputfile






