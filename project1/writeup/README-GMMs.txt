This project may seem a bit confusing. GMMs are an unsupervised learning technique and yet we are using them (or here just single gaussians)
on labeled data. 
--> we are training different GMMs (or gaussians) on different color models, but those color models can be in different clusters (light-dark-...) so we need to discover where those clusters are. The GMMs allow doing this. We a priori pick a the number of mixtures in one color model (guesstimate).
Each color model has a GMM associated with it. So the entire algorithm is supervised learning on top of unsupervised learning (like deep learning).

Also don't forget that GMMs are just a generalization of k-means!