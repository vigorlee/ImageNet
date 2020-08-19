import numpy as np
import matplotlib.pyplot as plt

'''
画图函数plot_images
参数介绍：
- images：包含多张图片数据的序列。
- labels：包含图片对应标签的序列(序列中的元素需要是0,1,2,...,9这样的正整数)。
'''

class_names = ['cassette_player', 'chain_saw', 'church', 'French_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'springer', 'tench']
def plot_images(images, labels, class_names ):
    fig, axes = plt.subplots(3, 5, figsize=(12, 6))
    axes = axes.flatten()
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img)
        ax.set_title(class_names[np.argmax(label)])
        ax.axis('off')
    plt.tight_layout()
    plt.show()