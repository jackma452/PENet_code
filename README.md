<strong>PE-Net: a parallel encoding enhanced network for medical image segmentation</strong>

<strong>Abstract</strong>

In recent years, hybrid medical image segmentation methods combining Convolutional Neural Networks (CNNs) and Transformers have achieved remarkable results by leveraging CNNs’ local context modeling and Transformers’ long-range context modeling. However, these models often overlook the spatial and channel redundancy in CNN-based feature extraction, leading to errors in long-range context modeling and inaccuracies in lesion segmentation. Additionally, increasing network depth can cause shallow feature overshadowing and gradient vanishing, resulting in blurred lesion boundaries. To address these challenges, this paper proposes the Parallel Encoder-Enhanced Medical Image Segmentation Network (PE-Net). PE-Net bridges the CNN branch and the proposed FE-Transformer branch, reducing spatial and channel redundancy and enhancing the accuracy of global–local feature fusion. The DCM module computes cross-attention on multi-scale features to extract deep features, while the CGAfusion module integrates shallow boundary information with deep features via skip connections, improving information flow and mitigating feature overshadowing and gradient issues. Experiments on ISIC 2018 and CVC-ClinicDB datasets validate the effectiveness of PE-Net. On ISIC 2018, it achieves a Dice Similarity Coefficient (DSC) of 92.46% and a Mean Intersection over Union (mIoU) of 88.95%, while on CVC-ClinicDB, it achieves a DSC of 90.75% and an mIoU of 89.84%, outperforming state-of-the-art methods.

<strong>Keywords</strong>

Medical image segmentation,Feature fusion,Feature enhancement,Transformer,Convolutional neural network

<strong>Codes</strong>

We would upload our code here as soon as possible, please wait.
