import os
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
import spectral
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score


###########################################################################################
def loadData(name): ## customize data and return data label and class_name
    data_path = os.path.join(os.getcwd(),'datasets')
    if name == 'Tangdaowan':
        data = loadmat(os.path.join(data_path, 'QUH-Tangdaowan.mat'))['Tangdaowan']
        labels = loadmat(os.path.join(data_path, 'QUH-Tangdaowan_GT.mat'))['TangdaowanGT']
        class_name = [     "Rubber track",    "Flaggingv",    "Sandy",    "Asphalt",    "Boardwalk",    "Rocky shallows",    "Grassland",
    "Bulrush",    "Gravel road",    "Ligustrum vicaryi",    "Coniferous pine",    "Spiraea",    "Bare soil",    "Buxus sinica",    "Photinia serrulata",
    "Populus",    "Ulmus pumila L",    "Seawater"]

    elif name == 'Qingyun':
        data = loadmat(os.path.join(data_path, 'QUH-Qingyun.mat'))['Chengqu']
        labels = loadmat(os.path.join(data_path, 'QUH-Qingyun_GT.mat'))['ChengquGT']
        class_name = ["Trees", "Concrete building", "Car", "Ironhide building", "Plastic playground", "Asphalt road"]

    elif name == 'Pingan':
        data = loadmat(os.path.join(data_path, 'QUH-Pingan.mat'))['Haigang']
        labels = loadmat(os.path.join(data_path, 'QUH-Pingan_GT.mat'))['HaigangGT']
        class_name = ["Ship", "Seawater", "Trees"," Concrete structure building", "Floating pier", "Asphalt road", "Brick houses",
                      "Steel houses"," Wharf construction land", "Car", "Road"]
    return data, labels, class_name

###########################################################################################
'''
def get_non_zero_indices(class_map):
    # Flatten the class map and get the indices of non-zero values
    non_zero_indices = np.argwhere(class_map != 0)
    
    # Flatten the class map to get the corresponding pixel values
    labels = class_map[non_zero_indices[:, 0], non_zero_indices[:, 1]]
    
    # Convert indices to a list of tuples for easier use
    non_zero_indices = [tuple(idx) for idx in non_zero_indices]
    
    return non_zero_indices, np.array(labels.tolist()) - 1
'''
###########################################################################################
def get_img_indexes (class_map, removeZeroindexes = True):
    """
    Get indices of elements in the class map.
    
    Parameters:
    class_map (numpy array): The class map (2D array).
    removeZero (bool): If True, return indices of non-zero elements, 
                       otherwise return indices of all elements.
    
    Returns:
    tuple: (indices, labels)
           - indices: List of tuples representing the indices of the selected elements.
           - labels: Array of labels corresponding to the indices.
    """
    if removeZeroindexes:
        # Get indices of non-zero values
        indices = np.argwhere(class_map != 0)
    else:
        # Get indices of all elements (including zeros)
        indices = np.argwhere(class_map != None)
    
    # Flatten the class map to get the corresponding pixel values (labels)
    labels = class_map[indices[:, 0], indices[:, 1]]
    
    # Convert indices to a list of tuples for easier use
    indices = [tuple(idx) for idx in indices]
    
    return indices, np.array(labels.tolist()) - 1

def createImageCubes(X, indices, windowSize):
    """
    Extract patches centered at given indices from the hyperspectral image 
    after applying zero padding.
    
    Parameters:
    X (numpy array): Hyperspectral image of shape (N, M, P)
    indices (list of tuples): List of indices where patches should be extracted
    windowSize (int): Window size, the patch will be of size (windowSize, windowSize)
    
    Returns:
    list: List of image patches extracted from the padded hyperspectral image
    """
    # Calculate margin based on window size
    margin = windowSize // 2
    
    # Apply zero padding to the hyperspectral image
    N, M, P = X.shape
    X_padded = np.zeros((N + 2 * margin, M + 2 * margin, P))
    
    # Offsets to place the original image in the center of the padded image
    x_offset = margin
    y_offset = margin
    X_padded[x_offset:N + x_offset, y_offset:M + y_offset, :] = X
    
    # Extract patches centered at the provided indices
    patches = []
    
    for idx in indices:
        i, j = idx
        i = i + margin
        j = j + margin
        # Get patch boundaries, ensuring the patch is centered at (i, j)
        i_min = i - margin  # Centered on the index, accounting for padding
        i_max = i_min + windowSize
        j_min = j - margin
        j_max = j_min + windowSize
        
        # Extract the patch
        patch = X_padded[i_min:i_max, j_min:j_max, :]
        

        patches.append(patch)
    
    return np.array(patches)

###########################################################################################
def splitTrainTestSet(X, y, testRatio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, 
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

###########################################################################################    
def applyPCA(X, numComponents=15, normalization = True):
    """PCA and processing
    Args:
        X (ndarray M*N*C): data needs DR
        numComponents (int, optional):  number of reserved components(Defaults to 15, 0 for no PCA).
        norm: normalization or not
    Returns:
        newX: processed data
        pca:
    """

    if numComponents == 0:
        newX = np.reshape(X, (-1, X.shape[2]))
    else:
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents)   ##PCA and normalization
        newX = pca.fit_transform(newX)
    if normalization:
        newX = minmax_scale(newX, axis=1)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], -1))
    return newX

###########################################################################################


def predict_by_batching(model, input_tensor_idx, batch_size, X, windowSize):
    '''
    Function to to perform predictions by dividing large tensor into small ones 
    to reduce load on GPU
    
    Parameters
    ----------
    model: The model itself with pre-trained weights.
    input_tensor: Tensor of diemnsion batches x windowSize x windowSize x channels x 1.
    batch_size: integer value smaller than batches .

    Returns
    -------
    Predicetd labels
    '''
    
    num_samples = len(input_tensor_idx)
    k = 0
    predictions = []
    for i in tqdm(range(0, num_samples, batch_size), desc="Progress"):
        k+=1
        
        batch = createImageCubes(X, input_tensor_idx[i:i + batch_size], windowSize)
        batch_predictions = model.predict(batch, verbose=0)
        predictions.append(batch_predictions)
        
    Y_pred_test = np.concatenate(predictions, axis=0)
  
    return Y_pred_test

###########################################################################################
def get_class_map(model, X, label, window_size):
    indexes, labels = get_img_indexes(label, removeZeroindexes = False)
    
    y_pred = predict_by_batching(model, indexes, 10000, X, window_size)
    
    y_pred = (np.argmax(y_pred, axis=1)).astype(np.uint8)
    
    Y_pred = np.reshape(y_pred, label.shape) + 1
    
   
    return Y_pred

###########################################################################################
def img_display(data = None, rgb_band = None, classes = None,class_name = None,title = None, 
                figsize = (7,7),palette = spectral.spy_colors):
    if data is not None:
        im_rgb = np.zeros_like(data[:,:,0:3])
        im_rgb = data[:,:,rgb_band]
        im_rgb = im_rgb/(np.max(np.max(im_rgb,axis = 1),axis = 0))*255
        im_rgb = np.asarray(im_rgb,np.uint8)
        fig, rgbax = plt.subplots(figsize = figsize)
        rgbax.imshow(im_rgb)
        rgbax.set_title(title)
        rgbax.axis('off')
        
    elif classes is not None:
        rgb_class = np.zeros((classes.shape[0],classes.shape[1],3))
        for i in np.unique(classes):
            rgb_class[classes==i]=palette[i]
        rgb_class = np.asarray(rgb_class, np.uint8)
        _,classax = plt.subplots(figsize = figsize)
        classax.imshow(rgb_class)
        classax.set_title(title)
        classax.axis('off')

###################################################################
def display_history(history):
    # Retrieve loss and accuracy data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # Create a figure with 2 horizontal subplots
    plt.figure(figsize=(12, 5))
    
    # Subplot for training and validation loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot for training and validation accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    # Show the combined figure
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    
    # Get training history
    val_acc = history.history['val_accuracy']
    best_epoch = np.argmax(val_acc)
    best_val = val_acc[best_epoch]
    
    # Plot Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], 'y', label='Train Acc')
    plt.plot(val_acc, 'r', label='Val Acc')
    
    # Mark best epoch
    plt.axvline(best_epoch, color='k', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.scatter(best_epoch, best_val, color='black')
    plt.text(best_epoch, best_val, f"{best_val:.2f}", fontsize=10, color='black', va='bottom')
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        
