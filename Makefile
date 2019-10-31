requirements:
	pip install -r requirements

data:
	mkdir ~/.kaggle
	echo '{"username":"christianhritter","key":$(KAGGLE_SECRET_ACCESS_KEY)}' > ~/.kaggle/kaggle.json
	chmod 600 ~/.kaggle/kaggle.json
	mkdir carvana-image-masking-challenge
	kaggle competitions download -c carvana-image-masking-challenge  -p carvana-image-masking-challenge
	cd carvana-image-masking-challenge; unzip -q "*.zip"

install_anaconda:
	mkdir -p /opt
	cd /opt
	wget -q --no-check-certificate https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O anaconda.sh
	echo "1046228398cf2e31e4a3b0bf0b9c9fd6282a1d7c  anaconda.sh | sha1sum -c -
	bash anaconda.sh -b -p /opt/conda
	rm anaconda.sh
	#export PATH=/opt/conda/bin:$(PATH)
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
	bash


create_env:
	conda create -n carvana python=3.6
	source activate carvana


