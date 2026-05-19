pip install google-cloud-storage # not needed if you are not using GCS
pip install deepspeed
pip install matplotlib

pip install --upgrade ai2thor ai2thor-colab
sudo apt-get -y install libvulkan1
sudo apt update
sudo apt install xserver-xorg
sudo apt install xorg xvfb

sudo ai2thor-xorg start
Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset & export DISPLAY=:0
# kill $(pgrep -f "Xvfb :0")
# sudo ai2thor-xorg stop
