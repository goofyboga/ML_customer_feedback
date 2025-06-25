# Technical Summary

This project investigated four classification algorithms for customer feedback categorisation, including KNN, Random Forest, and Neural Networks. Through our evaluation, we found that 
Random Forest worked better on majority classes, while simpler models like KNN struggled with high-dimensional and unbalanced data. After doing feature selection and tuning hyperparameters, 
the Neural Network gave the best results across all classes. It had the highest macro and weighted F1 scores among all models. In the end, we chose the Neural Network because it had better 
generalization, more stable training, and handled class imbalance more effectively. In the future, we could improve further by using ensemble methods and advanced sampling techniques like 
ADASYN to help with minority classes.
