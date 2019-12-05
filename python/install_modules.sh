pip install -r requirements.txt
export WELD_HOME=$(cd .. && pwd)

cd pyweld;  python setup.py develop; cd ..

cd grizzly; python setup.py develop; cd ..

#cd numpy; python setup.py develop; cd ..
cd numpy-0.1.0; python setup.py develop; cd ..
