* **IMPORTANT: You need a CUDA compatible GPU**

* Verify dependencies
```
./setup.sh
```

* Install requirements
```
pyenv install anaconda3-2020.02
pyenv virtualenv anaconda3-2020.02 cuda_captcha
pyenv activate cuda_captcha
pyenv local cuda_captcha
conda install --revision=18
```