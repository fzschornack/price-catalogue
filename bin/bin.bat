pip freeze > requirements.txt &
pip install -r .\requirements.txt -t lib &
C:\Users\fzsch\spark-3.3.0-bin-hadoop3\bin\spark-submit --master local[*] --driver-memory 10G --py-files .\lib\lib.zip .\price_catalogue\price_catalogue.py > .\log\log.txt 2>&1

# conda activate price-catalogue & C:\Users\fzsch\spark-3.3.0-bin-hadoop3\bin\spark-submit --master local[*] --driver-memory 10G price_catalogue.py > ..\log\log.txt 2>&1
