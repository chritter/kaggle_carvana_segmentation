requirements:
	pip install -r requirements

data:
	mkdir ~/.kaggle
	echo '{"username":"christianhritter","key":$(KAGGLE_SECRET_ACCESS_KEY)}' > ~/.kaggle/kaggle.json
	chmod 600 ~/.kaggle/kaggle.json
	mkdir carvana-image-masking-challenge
	kaggle competitions download -c carvana-image-masking-challenge  -p carvana-image-masking-challenge
	cd carvana-image-masking-challenge; unzip -q "*.zip"
