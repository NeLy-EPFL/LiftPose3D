
def multi_outsu():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    from skimage import data
    from skimage.filters import threshold_multiotsu

    import cv2

    # Setting the font size for all plots.
    matplotlib.rcParams['font.size'] = 9

    # # The input image.
    # image = data.camera()

    image = cv2.imread('/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_5_clipped/AG/1/LV/5_AG_1_LV_00001.tiff')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image,classes=3, nbins=5)

    image[image > thresholds[1]] = 10


    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    # Plotting the original image.
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    # Plotting the Multi Otsu result.
    ax[2].imshow(regions, cmap='jet')
    ax[2].set_title('Multi-Otsu result')
    ax[2].axis('off')

    plt.subplots_adjust()

    plt.show()

if __name__ == '__main__':

    multi_outsu()