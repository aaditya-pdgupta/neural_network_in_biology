# Introduction

Neural networks, also known as artificial neural networks (ANNs) or simulated network (SNNs) are sub-set of machine learning and are heart of deep learning algorithms. A standard neural network (NN) consists odf many simple, connected processors called neuron , each producing a sequence of real valued activations. An artificial neuron recieves a signals and then processes them and can signal neurons connected to it.

ANNs are comprised of a node layers, containing an input layer, one or more hidden layers and an output layer. Each node or artificial neuron, connects to another has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the next layer of the network.

<p align="center">
  <img src="../main/Pictures/artificial_neural_network.jpg" width="250" height="250"/>
</p>

The aim of constructing ANNs is to create artificial intelligence inspired by the working of human brain, even though the latter is not yet fully understood. They are based on the computers and man's brain abilities. In a similar way, the main asset of neural network is the ability of their neurons to take part in an analysis while working simultaneously but independently from each other.Artificial neural networks are also good at analysing large sets of unlabeled, often high-dimensional data-where it may be difficult to determine a prior which questions are most relevant and rewarding to ask

The tools for machine learning with neural networks were developed long ago, most of them during the second half of the last century. In 1943, McCulloch and Pitts analysed how networks of neurons can process information.


# Multi-layer Neural Network

A multi-layer neural network contains more than one layer of artificial neurons or nodes. They differ widely in design. It is important to note that while single-layer neural networks were useful early in the evolution of AI, the vast majority of networks used today have a multi-layer model.

<p align="center">
  <img src="../main/Pictures/multi.jpg" width="250" height="250"/>
</p>

Multi-layer neural networks can be set up in numerous ways. Typically, they have at least one iput layer, which send weighted inputs series of hidden layers, and an output layer at the end. These more sophisticated setups are also assosicated with nonlinear builds using sidmoids and other function to direct the firing or activation of artificial neurons. A fully connected multi-layer neural network is called as  Multilayer Perceptron(MLP). In above figure we have one hidden layer including one input layer and one output layer.  An MLP is a typical example of feedforward artificial neural network.

The number of layers and the number of neurons are reffered to as hyperparameters of a neural network and these need tuning. Cross validation techniques must be used to find ideal values. An example of a multi-layer is shown in figure. It contains a layer of n input neuron $I_{i}$ (i = 1,...,n), a hidden layer( $H_{m}$ ) of m neuron on $H_{j}$ with the activation function $f_{Hj}$ (j = 1,...,m) and output layer of n neuron $O_{k}$ with the activation function $f_{Ok}$ (k = 1,....,n). The activation functions can be different for different layers. The connection between the neurons are weighed with different values, $v_{ij}$ are the weights between the hidden layer and the output layer. Using these weights, the networks propagates the external signal through the layers prodeucing the output signal which is of the form $I_{i}$. 

$$O_{k} = f_{ok}(net_{ok}) = f_{ok}\left(\sum_{j=1}^{m+1} w_{kj} f_{h} (net_{Hj}\right) = f_{ok}\left(\sum_{j=1}^{m+1} w_{kj} f_{Hj} \left(\sum_{i=1}^{n+1} v_{ji}I_{i}\right)\right)$$

This type of ANN, which simply propagates the input through all the layers, is called a feed-forward multi-layer ANN. The example discussed here contains only oner hideen layer. The network architecture can be extended to contain as many hidden layer as necessary.

### ANN learning

The learning process for an ANN is the process through which the weights of the network are determined. This is achieved by adjusting the weights until the certain criteria are satisfied. The ANN is presented with a training dataset which contains the input vector and a target associated with each input vector. The target is the desired output. The weights of the ANN are adjusted iteratively such taht the difference between the actual output of the ANN and the target is minimized. The most common supervised learning method is based on the gradient descent learning rule. The method optimises the network weights such that a certain objective function E is minimized by calculating the gradient of E in the weight space and moving the weight vector along the negative gradient. The binary crossentropy (BCE) is used for training. The BCE calculates the loss by computing the following average:

$$Loss = -\frac{1}{N}\sum_{i}^{N}O_{i} . log \hat{O} + (1-O_{i})log(1-\hat{O_{i}})$$

where, $\hat{O_{i}}$ is the i-th scalar value in the output, $O_{i}$ is the corresponding target value, and N is the number of scalar values in the model output. For each iteration (usually called epoch), the gradient descent weight optimisation contains two phases:
* feed-forward pass in which the output of the network is calculated with the current value of the weights, activation function and bias
* backward propagation in which the errors of the output signal are propagated back from the output layer towards the input layer and the weights are adjusted as a function of the back-propagated errors.

to summarise, the supervised learning process implies the following steps:

(i) initialisation of the weights 

(ii) initialisation of the loss functions

(iii) for each training pattern

      (a) calculate $H_{jp}$ and $O_{kp}$ (feed-forward phase)
      
      (b) calculate the output error and the hidden layer error
      
      (c) adjust the weights $w_{kj}$ and $v_{kj} (black-propagation phase)
      
(iv) test the stopping criteria; if this is not met then the process continues from the step(ii)

As stopping criteria, common choices are a maximum number of epochs, a minimum value of the error function evaluated for the training data set, and the over-fitting point. The initialisation of the weights, the learning rate and the momentum are very important for the convergence and the efficiency of the learning process. 



# Application in biology

Neural network has been applied widely in biology since the 1980s, and has advanced tremendously in the field of reasearch, and others lately. Baldi and Brunak used application in biology to explain the theory of neural network.The spectrum of apllications of artificial neural network is very wide. ANNs is used for the diagonis of different diseases in both palnt and animal caused due to different factors. The capacity of ANNs to analyze large amounts of data and detect patterns warrants application in analysis of medical images, calssification of tumors and prediction of survival have some how made easy resarch in medical biology.

<p align="center">
  <img src="../main/Pictures/bioinformatics.jpg" width="300" height="300"/>
</p>

Neural networks have also been actively used in many bioinformatics applications such as DNA sequence, prediction, protein secondary structure prediction, gene expresssion profiles classification and analysis of gene expression patterns. The concepts of neural network used in pattern classification and signal processing gene procesing have been sucessfully applied in bioinformatics. Neural Likewise, in medical science ANNs have been extensively applied in diagosis, electronic signal analysis, radiology etc. ANNs have been used by many authors for clinical reserach. Application of ANNs are increasing in medical data mining. In agriculture Artififcial neural networks are one of the most popular tools for high production efficiency combined with a high quality products. ANNs can replace the classical methods of modelling many issuses, and are one of the main alternatives to classical mathematical models.networks have also been applied to the analysis of gene expression patterns as an alternative to hierarchial cluster methods. Neural networks have been used in agriculture for selection of appropriate net of plant during sudden and quick changes of environmental condition and predict the result of it. Here, we use ANN to classify that the Yeast is poisonous or edible to eat based on its properties.

# Classification of Mushrooms

### Introduction

A mushroom is the fleshy, spore-bearing fruiting body of a fungus, typically produced above ground, on soil, or on its food source. Mushroom is one of the fungi types food which include nutritrional content and health benefits. It has differnt parts such as cap, scale, gills, tubes, pores, ring, stipe, stalk, scales and volva.

<p align="center">
  <img src="../main/Pictures/mushroom_pic.jpg" width="300" height="300"/>
</p>

* Gills, Pores, or Teeth: These structures appear under the mushroom's cap. They look similar to a fish's gills.
* Ring: The ring (sometimes called the annulus) is the remaining structure of the partial veil after the gills have pushed through.
* Stem or Stipe: The stem is the tall structure that holds the cap high above the ground.
* Volva: The volva is the protective veil that remains after the mushroom sprouted up from the ground. As the fungus grows, it breaks through the volva.
* Spores: Microscopic seeds acting as reproductive agents; they are usually released into the air and fall on a substrate to produce a new mushroom.

### Motivation

Mushroom specis can be both edible and non-edible. The non-edible are toxic and poisonous. The identification of edible and poisonous mushrooms among its existing species is a must due to its high demand for people's everyday meal and major advantage on medical science. It is very difficult to distinguish between edible and poisonous mushrooms due to their similar apperance. Distinguishing edible from poisonous mushroom species needs meticulous care to detail; there is no single trait by which poisonous mushrooms can be recognized, nor one by which all edible mushrooms can be recognized. So, ANNs will be able to classify if a mushroom is poisonous or not using data mining as one of the approaches for obtaining computer assisted knowledge. In this, we use the training dataset that contain the mushroom images that classify it into edible and poisonous. Here our approach aim is to classifies and predict for the class (groups) of mushrooms when submit the features of the mushrooms to different techniques of machine learning.

<p align="center">
  <img src="../main/Pictures/different_types_of_mushroom.jpg" width="300" height="300"/>
</p>

### Data

A dataset that was originally contributed to the UCI Machine Learning repository nearly 30 years ago, mushroom hunting (otherwise known as "shrooming") is used to for the ANN purpose. It helps to learn which features (inputs) spell certain death and which are most palatable in this dataset of mushroom characteristics.This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy. 

* In this dataset, we consider the cap-shape in different letter notation such as bell-b, conical-c, convex-x, flat-f, knobbed-k and sunnksen as s. We also differntiate the different features of cap surface as fibrous-f, grooves-g, scaly-y, smooth-k. 
* Likewise, we also adrressed the different feautures of cap colaor as brown-n, buff-b, cinnamon-c, gray-g, green-r, pink-p, purple-u, red-e, white-w and yellow-y. 
* For odor, we have almond-a, anise-I, creosote-c, fishy-y, foul-f, musty-m, none-n, pungent-p and spicy-s. On the basis of gill-attachement we have denoted attached-a, descending-d, free-f and notched-n. Gill-spacing is denoted as close-c, crowded-w and distant-d. 
* In same way, gill-size features are denoted as broad-b and narrow-n. Gill color for black-k, brown-n, buff-b,, chocolate-h, gary-g, green-r, orange-o, pink-p, purple-u, red-e, white-w and yellow-y. 
* Similarly, for stalk shape feature different noatation is used such as enlarging-e and tapering-t. For different features of stalk-root we denoted bulbos-b, club-c, cup-u, equal-e, rhizomorphs-z, rooted-r and missing-?.
*  We also have a different features under stalk-surface-below-ring which is denoted as fibrous-f, scale-y, silky-k and smooth-s. Continously for stalk-color-above-ring features we denote brown-n, buff-b, cinnamon-c, gray-g, orange-o, pink-p, red-e, white-w and yellow-y. 
*  For veil-type, we denote partial-p a-and universal-u. Similarly for veil-color we denote brown-n, orange-o, white-w and yellow-y. At last for ring-number none-n, one-o and two-t. 
 
### Separation Using Machine Learning

<p align="center">
  <img src="../main/classify_yeast/losses.jpg" width="400" height="400" />
  <img src="../main/classify_yeast/roc.jpg" width="400" height="400" /> 
</p>

