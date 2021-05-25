from matplotlib import pyplot as plt
import skimage.io
import cnn_predict



img = skimage.io.imread('data/varoa.jpg')
net = cnn_predict.load_net('log/varoa_net_03.pth')
out = cnn_predict.predict(net, img)
print(out)